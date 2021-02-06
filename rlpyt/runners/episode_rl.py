import psutil
import time
import torch
import math
from collections import deque

from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.runners.base import BaseRunner
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter

class EpisodicRlRunner(MinibatchRl):
    def train(self):
        """
        Performs startup, then loops by alternating between
        ``sampler.obtain_samples()`` and ``algo.optimize_agent()``, logging
        diagnostics at the specified interval.
        """
        n_itr = self.startup()
        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                if self.transfer and self.transfer_iter == itr:
                    self.sampler.transfer(self.transfer_arg)  # Transfer if doing
                    self._traj_infos.clear()  # Clear trajectory information
                    self._transfer_start(itr, opt_info)
                self.agent.sample_mode(itr)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    self.log_diagnostics(itr)
                if self.n_episodes is not None and self._cum_completed_trajs >= self.n_episodes:
                    break
        self.shutdown()