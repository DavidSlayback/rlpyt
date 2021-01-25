
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.oc import ContinuousIntraOptionPolicy, OptionCriticHead_IndependentPreprocessor
from rlpyt.models.running_mean_std import RunningMeanStdModel
from rlpyt.models.utils import Dummy

class MujocoOCFfModel(torch.nn.Module):
    """
    Model commonly used in Mujoco locomotion agents: an MLP which outputs
    distribution means, separate parameter for learned log_std, and separate
    MLPs for state-option-value estimate, termination probabilities. Policy over options
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            option_size,
            hidden_sizes=None,  # None for default (see below).
            hidden_nonlinearity=torch.nn.Tanh,  # Module form.
            mu_nonlinearity=torch.nn.Tanh,  # Module form.
            init_log_std=0.,
            normalize_observation=True,
            norm_obs_clip=10,
            norm_obs_var_clip=1e-6,
            baselines_init=True,  # Orthogonal initialization of sqrt(2) until last layer, then 0.01 for policy, 1 for value
            use_interest=False,  # IOC sigmoid interest functions
            ):
        """Instantiate neural net modules according to inputs."""
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        hidden_sizes = hidden_sizes or [64, 64]
        inits_mu = inits_v = None
        if baselines_init:
            inits_mu = (np.sqrt(2), 0.01)
            inits_v = (np.sqrt(2), 1.)
        # Body for intra-option policy mean
        mu_mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=None,
            nonlinearity=hidden_nonlinearity,
            inits=inits_mu
        )
        # Intra-option policy. Outputs tanh mu if exists, else unactivateed linear. Also logstd
        self.mu = torch.nn.Sequential(mu_mlp, ContinuousIntraOptionPolicy(input_size=mu_mlp.output_size,
                                                                          num_options=option_size,
                                                                          num_actions=action_size,
                                                                          ortho_init=baselines_init,
                                                                          ortho_init_value=inits_mu[-1],
                                                                          init_log_std=init_log_std,
                                                                          mu_nonlinearity=mu_nonlinearity))
        # Option value. Pure linear
        self.q = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=option_size,
            nonlinearity=hidden_nonlinearity,
            inits=inits_v
        )
        # Option termination. MLP with sigmoid at end
        self.beta = torch.nn.Sequential(MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=option_size,
            nonlinearity=hidden_nonlinearity,
            inits=inits_v
        ), torch.nn.Sigmoid())
        # self.log_std = torch.nn.Parameter(init_log_std * torch.ones(action_size))
        # Softmax policy over options
        self.pi_omega = torch.nn.Sequential(MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=option_size,
            nonlinearity=hidden_nonlinearity,
            inits=inits_v
        ), torch.nn.Softmax(-1))
        # Per-option sigmoid interest functions
        self.pi_omega_I = torch.nn.Sequential(MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=option_size,
            nonlinearity=hidden_nonlinearity,
            inits=inits_v
        ), torch.nn.Sigmoid()) if use_interest else Dummy(option_size)
        if normalize_observation:
            self.obs_rms = RunningMeanStdModel(observation_shape)
            self.norm_obs_clip = norm_obs_clip
            self.norm_obs_var_clip = norm_obs_var_clip
        self.normalize_observation = normalize_observation
        self.use_interest = use_interest

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute mean, log_std, q-value, and termination estimates from input state. Infers
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], with T=1,B=1 when not given. Used both in sampler
        and in algorithm (both via the agent).
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.norm_obs_var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)

        obs_flat = observation.view(T * B, -1)
        mu, logstd = self.mu(obs_flat)
        q = self.q(obs_flat)
        log_std = logstd.repeat(T * B, 1, 1)
        beta = self.beta(obs_flat)
        pi = self.pi_omega(obs_flat)
        I = self.pi_omega_I(obs_flat)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, q, beta, pi, I = restore_leading_dims((mu, log_std, q, beta, pi, I), lead_dim, T, B)
        pi = pi * I  # Torch multinomial will normalize
        return mu, log_std, q, beta, pi

    def update_obs_rms(self, observation):
        if self.normalize_observation:
            self.obs_rms.update(observation)

class RefactoredMujocoOcFfModel(torch.nn.Module):
    """
    Model commonly used in Mujoco locomotion agents: an MLP which outputs
    distribution means, separate parameter for learned log_std, and separate
    MLPs for state-option-value estimate, termination probabilities. Policy over options
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            option_size,
            hidden_sizes=None,  # None for default (see below).
            hidden_nonlinearity=torch.nn.Tanh,  # Module form.
            mu_nonlinearity=torch.nn.Tanh,  # Module form.
            init_log_std=0.,
            normalize_observation=True,
            norm_obs_clip=10,
            norm_obs_var_clip=1e-6,
            baselines_init=True,  # Orthogonal initialization of sqrt(2) until last layer, then 0.01 for policy, 1 for value
            use_interest=False,  # IOC sigmoid interest functions
            use_diversity=False,  # TDEOC q entropy output
            use_attention=False,
            ):
        """Instantiate neural net modules according to inputs."""
        super().__init__()
        from functools import partial
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        hidden_sizes = hidden_sizes or [64, 64]
        inits_mu = inits_v = None
        if baselines_init:
            inits_mu = (np.sqrt(2), 0.01)
            inits_v = (np.sqrt(2), 1.)
        body_mlp_class = partial(MlpModel, hidden_sizes=hidden_sizes, output_size=None, nonlinearity=hidden_nonlinearity, inits=inits_v)
        self.model = OptionCriticHead_IndependentPreprocessor(
            input_size=input_size,
            input_module_class=body_mlp_class,
            output_size=action_size,
            num_options=option_size,
            intra_option_policy='continuous',
            intra_option_kwargs={'init_log_std': init_log_std, 'mu_nonlinearity': mu_nonlinearity},
            input_module_kwargs={},
            use_interest=use_interest,
            use_diversity=use_diversity,
            use_attention=use_attention,
            orthogonal_init=baselines_init,
            orthogonal_init_base=inits_v[1],
            orthogonal_init_pol=inits_mu[1]
        )
        if normalize_observation:
            self.obs_rms = RunningMeanStdModel(observation_shape)
            self.norm_obs_clip = norm_obs_clip
            self.norm_obs_var_clip = norm_obs_var_clip
        self.normalize_observation = normalize_observation
        self.use_interest = use_interest
        self.use_diversity = use_diversity
        self.use_attention = use_attention

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute mean, log_std, q-value, and termination estimates from input state. Infers
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], with T=1,B=1 when not given. Used both in sampler
        and in algorithm (both via the agent).
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.norm_obs_var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)

        obs_flat = observation.view(T * B, -1)
        (mu, logstd), beta, q, pi_I, q_ent = self.model(obs_flat)
        log_std = logstd.repeat(T * B, 1, 1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, q, beta, pi, q_ent = restore_leading_dims((mu, log_std, q, beta, pi_I, q_ent), lead_dim, T, B)
        return mu, log_std, q, beta, pi

    def update_obs_rms(self, observation):
        if self.normalize_observation:
            self.obs_rms.update(observation)
