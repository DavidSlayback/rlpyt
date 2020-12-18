import numpy as np
import torch

from rlpyt.agents.base import (AgentStep, BaseAgent, OCAgentMixin, RecurrentOCAgentMixin, AlternatingRecurrentOCAgentMixin, AlternatingOCAgentMixin)
from rlpyt.agents.pg.base import AgentInfoOC, AgentInfoOCRnn
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.utils.tensor import select_at_indexes

class GaussianOCAgentBase(BaseAgent):
    """
    Agent for option-critic algorithm using Gaussian action distribution.
    """

    def __call__(self, observation, prev_action, prev_reward):
        """Performs forward pass on training data, for algorithm."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, q, beta, pi = self.model(*model_inputs)
        # Need gradients from intra-option (DistInfoStd), q_o (q), termination (beta), and pi_omega (DistInfo)
        return buffer_to((DistInfoStd(mean=mu, log_std=log_std), q, beta, DistInfo(prob=pi)), device="cpu")

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        """Extends base method to build Gaussian distribution."""
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        assert len(env_spaces.action.shape) == 1
        assert len(np.unique(env_spaces.action.high)) == 1
        assert np.all(env_spaces.action.low == -env_spaces.action.high)
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            # min_std=MIN_STD,
            # clip=env_spaces.action.high[0],  # Probably +1?
        )
        self.distribution_omega = Categorical(
            dim=self.model_kwargs["option_size"]
        )

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """
        Compute policy's option and action distributions from inputs.
        Calls model to get mean, std for all pi_w, q, beta for all options, pi over options
        Moves inputs to device and returns outputs back to CPU, for the
        sampler.  (no grad)
        """
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, q, beta, pi = self.model(*model_inputs)
        dist_info_omega = DistInfo(prob=pi)
        new_o, terminations = self.sample_option(beta, dist_info_omega)  # Sample terminations and options
        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        dist_info_o = DistInfoStd(mean=select_at_indexes(new_o, mu), log_std=select_at_indexes(new_o, log_std))
        action = self.distribution.sample(dist_info_o)
        agent_info = AgentInfoOC(dist_info=dist_info, q=q, value=select_at_indexes(new_o, q), termination=terminations, inter_option_dist_info=dist_info_omega, prev_o=self._prev_option, o=new_o)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_oc_state(new_o)
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        """
        Compute the value estimate for the environment state, e.g. for the
        bootstrap value, V(s_{T+1}), in the sampler.

        For option-critic algorithms, this is the q(s_{T+1}, prev_o) * (1-beta(s_{T+1}, prev_o)) +
        beta(s_{T+1}, prev_o) * sum_{o} pi_omega(o|s_{T+1}) * q(s_{T+1}, o)
        (no grad)
        """
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _mu, _log_std, q, beta, pi = self.model(*model_inputs)  # [B, nOpt]
        v = (q * pi).mean(-1)  # Weight q value by probability of option. Average value if terminal
        q_prev_o = select_at_indexes(self.prev_option, q)
        beta_prev_o = select_at_indexes(self.prev_option, beta)
        value = q_prev_o * (1 - beta_prev_o) + v * beta_prev_o
        return value.to("cpu")


class GaussianOCAgent(OCAgentMixin, GaussianOCAgentBase):
    pass


class AlternatingGaussianOCAgent(AlternatingOCAgentMixin, GaussianOCAgentBase):
    pass


class RecurrentGaussianOCAgentBase(BaseAgent):

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        """Performs forward pass on training data, for algorithm (requires
        recurrent state input)."""
        # Assume init_rnn_state already shaped: [N,B,H]
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            init_rnn_state), device=self.device)
        mu, log_std, q, beta, pi, next_rnn_state = self.model(*model_inputs)
        # Need gradients from intra-option (DistInfoStd), q_o (q), termination (beta), and pi_omega (DistInfo)
        dist_info, q, beta, dist_info_omega = buffer_to((DistInfoStd(mean=mu, log_std=log_std), q, beta, DistInfo(prob=pi)), device="cpu")
        return dist_info, q, beta, dist_info_omega, next_rnn_state  # Leave rnn_state on device.

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory)
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            # min_std=MIN_STD,
            # clip=env_spaces.action.high[0],  # Probably +1?
        )
        self.distribution_omega = Categorical(
            dim=self.model_kwargs["option_size"]
        )

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu, log_std, q, beta, pi, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        terminations = torch.bernoulli(beta).bool()  # Sample terminations
        dist_info_omega = DistInfo(prob=pi)
        new_o = self.sample_option(terminations, dist_info_omega)
        dist_info = DistInfoStd(mean=mu, log_std=log_std)
        dist_info_o = DistInfoStd(mean=select_at_indexes(new_o, mu), log_std=select_at_indexes(new_o, log_std))
        action = self.distribution.sample(dist_info_o)
        # Model handles None, but Buffer does not, make zeros if needed:
        prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        agent_info = AgentInfoOCRnn(dist_info=dist_info, q=q, value=select_at_indexes(new_o, q),termination=terminations, inter_option_dist_info=dist_info_omega, prev_o=self._prev_option, o=new_o, prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Keep on device.
        self.advance_oc_state(new_o)
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        """
        Compute the value estimate for the environment state using the
        currently held recurrent state, without advancing the recurrent state,
        e.g. for the bootstrap value V(s_{T+1}), in the sampler.

        For option-critic algorithms, this is the q(s_{T+1}, prev_o) * (1-beta(s_{T+1}, prev_o)) +
        beta(s_{T+1}, prev_o) * sum_{o} pi_omega(o|s_{T+1}) * q(s_{T+1}, o)
        (no grad)
        """
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _mu, _log_std, q, beta, pi, _rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        v = (q * pi).mean(-1)  # Weight q value by probability of option. Average value if terminal
        q_prev_o = select_at_indexes(self.prev_option, q)
        beta_prev_o = select_at_indexes(self.prev_option, beta)
        value = q_prev_o * (1 - beta_prev_o) + v * beta_prev_o
        return value.to("cpu")


class RecurrentGaussianOCAgent(RecurrentOCAgentMixin, RecurrentGaussianOCAgentBase):
    pass


class AlternatingRecurrentGaussianOCAgent(AlternatingRecurrentOCAgentMixin,
        RecurrentGaussianOCAgentBase):
    pass