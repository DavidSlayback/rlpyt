import psutil
import time
import torch
import math
from collections import deque

from rlpyt.runners.base import BaseRunner
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter

class EpisodicRlRunner(BaseRunner):
    """
    Implements startup, logging, and agent checkpointing functionality, to be
    called in the `train()` method of the subclassed runner.  Subclasses will
    modify/extend many of the methods here.

    Args:
        algo: The algorithm instance.
        agent: The learning agent instance.
        sampler: The sampler instance.
        n_episodes (int): Total number of episodes to run in training loop.
        seed (int): Random seed to use, if ``None`` will generate randomly.
        affinity (dict): Hardware component assignments for sampler and algorithm.
        log_interval_steps (int): Number of environment steps between logging to csv.
    """

    _eval = False

    def __init__(
            self,
            algo,
            agent,
            sampler,
            n_episodes,
            seed=None,
            affinity=None,
            log_interval_steps=1e5,
            ):
        n_steps = int(n_episodes)
        log_interval_steps = int(log_interval_steps)
        affinity = dict() if affinity is None else affinity
        save__init__args(locals())
        self.min_itr_learn = getattr(self.algo, 'min_itr_learn', 0)

    def startup(self):
        """
        Sets hardware affinities, initializes the following: 1) sampler (which
        should initialize the agent), 2) agent device and data-parallel wrapper (if applicable),
        3) algorithm, 4) logger.
        """
        p = psutil.Process()
        try:
            if (self.affinity.get("master_cpus", None) is not None and
                    self.affinity.get("set_affinity", True)):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
            f"{cpu_affin}.")
        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        self.rank = rank = getattr(self, "rank", 0)
        self.world_size = world_size = getattr(self, "world_size", 1)
        examples = self.sampler.initialize(
            agent=self.agent,  # Agent gets initialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size,
        )
        self.itr_batch_size = self.sampler.batch_spec.size * world_size
        logger.log(f"Running {self.n_episodes} iterations of minibatch RL.")
        n_itr = self.get_n_itr()
        self.agent.to_device(self.affinity.get("cuda_idx", None))
        if world_size > 1:
            self.agent.data_parallel()
        self.algo.initialize(
            agent=self.agent,
            n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=examples,
            world_size=world_size,
            rank=rank,
        )
        self.initialize_logging()
        return n_itr