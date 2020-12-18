
import torch
from rlpyt.models.running_mean_std import RunningMeanStdModel, RunningReward
from rlpyt.utils.tensor import select_at_indexes
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.algos.utils import (discount_return, generalized_advantage_estimation,
    valid_from_done)

# Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase
OptInfo = namedtuple("OptInfo", ["loss", "pi_loss", "value_loss", "gradNorm", "entropy", "perplexity"])
AgentTrain = namedtuple("AgentTrain", ["dist_info", "value"])


class PolicyGradientAlgo(RlAlgorithm):
    """
    Base policy gradient / actor-critic algorithm, which includes
    initialization procedure and processing of data samples to compute
    advantages.
    """

    bootstrap_value = True  # Tells the sampler it needs Value(State')
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False,
            examples=None, world_size=1, rank=0):
        """
        Build the torch optimizer and store other input attributes. Params
        ``batch_spec`` and ``examples`` are unused.
        """
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        self.agent = agent
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.rank = rank
        self.world_size = world_size
        self.ret_rms = None if self.normalize_rewards is None else \
            RunningReward(None, *self.rew_clip, self.rew_min_var)
        # self.ret_rms = None if self.normalize_rewards is None else \
        #     RunningMeanStdModel((), *self.rew_clip, self.rew_min_var)

    def process_returns(self, samples):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples.  Uses either discounted returns (if ``self.gae_lambda==1``)
        or generalized advantage estimation.  Mask out invalid samples
        according to ``mid_batch_reset`` or for recurrent agent.  Optionally,
        normalize advantages.
        """
        reward, done, value, bv = (samples.env.reward, samples.env.done,
            samples.agent.agent_info.value, samples.agent.bootstrap_value)
        done = done.type(reward.dtype)
        if self.normalize_rewards is not None:  # Normalize and clip rewards before computing advantage
            if self.normalize_rewards == 'return':
                return_ = discount_return(reward, done, 0., self.discount)  # NO boostrapping of value
                reward = self.ret_rms(reward, center=False)
            else:
                reward = self.ret_rms(reward)

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(
                reward, value, done, bv, self.discount, self.gae_lambda)

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        return return_, advantage, valid

OptInfoOC = namedtuple("OptInfo", ["loss", "pi_loss", "q_loss", "beta_loss", "pi_omega_loss", "gradNorm", "entropy", "pi_omega_entropy"])
AgentTrainOC = namedtuple("AgentTrain", ["dist_info", "q", "beta", "inter_option_dist_info"])
class OCAlgo(RlAlgorithm):
    """
    Base option-critic algorithm, which includes
    initialization procedure and processing of data samples to compute
    advantages.
    """

    bootstrap_value = True  # Tells the sampler it needs Value(State')
    opt_info_fields = tuple(f for f in OptInfoOC._fields)  # copy

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False,
            examples=None, world_size=1, rank=0):
        """
        Build the torch optimizer and store other input attributes. Params
        ``batch_spec`` and ``examples`` are unused.
        """
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        self.agent = agent
        self.n_opt = self.agent.n_opt
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.rank = rank
        self.world_size = world_size
        self.ret_rms = None if self.normalize_rewards is None else \
            RunningReward(None, *self.rew_clip, self.rew_min_var)
        # self.ret_rms = None if self.normalize_rewards is None else \
        #     RunningMeanStdModel((), *self.rew_clip, self.rew_min_var)

    def process_returns(self, samples):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples.  Uses either discounted returns (if ``self.gae_lambda==1``)
        or generalized advantage estimation.  Mask out invalid samples
        according to ``mid_batch_reset`` or for recurrent agent.  Optionally,
        normalize advantages.
        """
        reward, done, q, v, termination, o, prev_o, pi_omega, bv = (samples.env.reward, samples.env.done,
                                                                    samples.agent.agent_info.q,
                                                                    samples.agent.agent_info.value,
                                                                    samples.agent.agent_info.termination,
                                                                    samples.agent.agent_info.o,
                                                                    samples.agent.agent_info.prev_o,
                                                                    samples.agent.agent_info.dist_info_omega,
                                                                    samples.agent.bootstrap_value)
        done = done.type(reward.dtype)
        q_o = select_at_indexes(o, q)
        if self.normalize_rewards is not None:  # Normalize and clip rewards before computing advantage
            if self.normalize_rewards == 'return':
                return_ = discount_return(reward, done, 0., self.discount)  # NO boostrapping of value
                reward = self.ret_rms(reward, center=False)
            else:
                reward = self.ret_rms(reward)

        valid_o = torch.ones_like(done)  # Options: If reset, no termination gradient, no deliberation cost
        valid_o[prev_o == -1] = 0.
        reward[torch.logical_and(valid_o.bool(), termination)] -= self.delib_cost

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - q_o
        else:
            advantage, return_ = generalized_advantage_estimation(
                reward, q_o, done, bv, self.discount, self.gae_lambda)

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        q_prev_o = select_at_indexes(prev_o, q)
        termination_advantage = q_prev_o - v + self.delib_cost

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        if self.normalize_termination_advantage:
            valid_mask = valid_o > 0
            adv_mean = termination_advantage[valid_mask].mean()
            adv_std = termination_advantage[valid_mask].std()
            termination_advantage[:] = (termination_advantage - adv_mean) / max(adv_std, 1e-6)

        return return_, advantage, valid, termination_advantage, valid_o