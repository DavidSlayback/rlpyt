import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from rlpyt.models.utils import layer_init, EpislonGreedyLayer, View

class DiscreteIntraOptionPolicy(nn.Module):
    """Option-Critic intra-option policy model for discrete action spaces

    Args:
        input_size (int): Number of inputs
        num_options (int): Number of options for inter-option policy
        num_actions (int): Number of actions for intra-option policy
        ortho_init (bool): Whether to use orthogonal initialization instead of default
        ortho_init_value (float): If using orthogonal init, initialization std to use for (standard, value, policy) networks
    Returns:
        pi_w (Tensor): Logits for input to a Categorical distribution
    """
    def __init__(self,
                 input_size,
                 num_options,
                 num_actions,
                 ortho_init=False,
                 ortho_init_value=1e-2):
        super().__init__()
        pi_w = layer_init(nn.Linear(input_size, num_options * num_actions), ortho_init_value) if ortho_init else nn.Linear(input_size, num_options * num_actions)
        self.pi_w = nn.Sequential(pi_w, View((num_options, num_actions)))

    def forward(self, x):
        return self.pi_w(x)

class ContinuousIntraOptionPolicy(nn.Module):
    """Option-Critic intra-option policy model for continuous action spaces

    Args:
        input_size (int): Number of inputs
        num_options (int): Number of options for inter-option policy
        num_actions (int): Number of actions for intra-option policy
        ortho_init (bool): Whether to use orthogonal initialization instead of default
        ortho_init_value (float): If using orthogonal init, initialization std to use for (standard, value, policy) networks,
        init_log_std (float): Initial value for log_std parameter,
        mu_nonlinearity (torch.nn.Module): Nonlinearity applied to mu output. Identity by default
    Returns:
        pi_mu (Tensor): Logits for input to a Categorical distribution
        log_std (Tensor): Current log_std for each option-action pair
    """
    def __init__(self,
                 input_size,
                 num_options,
                 num_actions,
                 ortho_init=True,
                 ortho_init_value=1e-2,
                 init_log_std=0.,
                 mu_nonlinearity=torch.nn.Identity):
        super().__init__()
        pi_mu = layer_init(nn.Linear(input_size, num_options * num_actions), ortho_init_value) if ortho_init else nn.Linear(input_size, num_options * num_actions)
        self.pi_mu = nn.Sequential(pi_mu, View((num_options, num_actions)), mu_nonlinearity())
        self.log_std = torch.nn.Parameter(init_log_std * torch.ones((num_options, num_actions)))  # State-independent vector of log_std

    def forward(self, x):
        return self.pi_mu(x), self.log_std


"""
Interest option critic init: 1. for all except 0.01 for intra-option policy. TDEOC is same
Zhang uses 1. for all in master branch, but 1e-3 for all in ppoc
"""

class SingleOptionGaussian(nn.Module):
    """Single option Gaussian network. Needs inputs from action and termination bodies (tend to be separate)

    Args:
        input_size (int): Number of inputs
        num_actions (int): Number of actions for intra-option policy
        ortho_init (bool): Whether to use orthogonal initialization instead of default
        inits (tuple of floats): If using orthogonal init, initialization std to use for (standard, value, policy) networks
        init_log_std (float): Initial log std parameter
    """
    def __init__(self,
                 input_size,
                 num_actions,
                 ortho_init=False,
                 inits=(np.sqrt(2), 1., 1e-2),
                 init_log_std=0.
                 ):
        super().__init__()
        self.beta = layer_init(nn.Linear(input_size, 1), inits[0]) if ortho_init else nn.Linear(input_size, 1)
        self.std = nn.Parameter(torch.full((num_actions,), init_log_std))
        self.pi = layer_init(nn.Linear(input_size, num_actions), inits[2]) if ortho_init else nn.Linear(input_size, num_actions)

    def forward(self, x_pi, x_beta):
        phi_pi = self.pi(x_pi)
        mean = torch.tanh(self.fc_pi(phi_pi))
        std = F.softplus(self.std).expand(mean.size(), -1)
        beta = F.sigmoid(self.beta(x_beta))
        """
        def compute_pi_bar(self, options, action, mean, std):
            options = options.unsqueeze(-1).expand(-1, -1, mean.size(-1))
            mean = mean.gather(1, options).squeeze(1)
            std = std.gather(1, options).squeeze(1)
            dist = torch.distributions.Normal(mean, std)
            pi_bar = dist.log_prob(action).sum(-1).exp().unsqueeze(-1)
            return pi_bar"""

        return {
            'mean': mean,
            'std': std,
            'beta': beta,
        }

class CategoricalOCHead(nn.Module):
    """Option-Critic Head model for discrete action spaces

    Args:
        input_size (int): Number of inputs
        num_options (int): Number of options for inter-option policy
        num_actions (int): Number of actions for intra-option policy
        pi_omega (string): 'epsilon' for E-greedy policy over options (uses model q output), 'softmax' for separately learned softmax policy,
        pi_omega_epsilon (float): Epsilon for the above policy, if using
        ortho_init (bool): Whether to use orthogonal initialization instead of default
        inits (tuple of floats): If using orthogonal init, initialization std to use for (standard, value, policy) networks
    """
    def __init__(self,
                 input_size,
                 num_options,
                 num_actions,
                 pi_omega='epsilon',
                 pi_omega_epsilon=0.01,
                 ortho_init=False,
                 inits=(np.sqrt(2), 1., 1e-2)
                 ):
        super().__init__()
        if ortho_init: assert len(inits) == 3
        self.Q = layer_init(nn.Linear(input_size, num_options), inits[1]) if ortho_init else nn.Linear(input_size, num_options)
        pi_w = layer_init(nn.Linear(input_size, num_options * num_actions), inits[2]) if ortho_init else nn.Linear(input_size, num_options * num_actions)
        self.pi_w = nn.Sequential(pi_w, View((num_options, num_actions)), nn.LogSoftmax(-1))
        beta = layer_init(nn.Linear(input_size, num_options), inits[0]) if ortho_init else nn.Linear(input_size, num_options)
        self.beta = nn.Sequential(beta, nn.Sigmoid())
        if pi_omega == 'epsilon':
            assert isinstance(pi_omega_epsilon, float)
            self.pi_omega = EpislonGreedyLayer(pi_omega_epsilon)
        else:
            pi_omega = layer_init(nn.Linear(input_size, num_options), inits[2]) if ortho_init else nn.Linear(input_size, num_options)

    def forward(self, input):
        pass

class GaussianOCHead(nn.Module):
    """Option-Critic Head model for continuous action spaces

    Args:
        input_size (int): Number of inputs
        num_options (int): Number of options for inter-option policy
        num_actions (int): Number of actions for intra-option policy
        pi_omega (string): 'epsilon' for E-greedy policy over options (uses model q output), 'softmax' for separately learned softmax policy,
        pi_omega_epsilon (float): Epsilon for the above policy, if using
        ortho_init (bool): Whether to use orthogonal initialization instead of default
        inits (tuple of floats): If using orthogonal init, initialization std to use for (standard, value, policy) networks
    """
    def __init__(self,
                 input_size,
                 num_options,
                 num_actions,
                 pi_omega='epsilon',
                 pi_omega_epsilon=0.01,
                 ortho_init=False,
                 inits=(np.sqrt(2), 1., 1e-2)
                 ):
        super().__init__()
        if ortho_init: assert len(inits) == 3
        self.Q = layer_init(nn.Linear(input_size, num_options), inits[1]) if ortho_init else nn.Linear(input_size, num_options)
        pi_w = layer_init(nn.Linear(input_size, num_options * num_actions), inits[2]) if ortho_init else nn.Linear(input_size, num_options * num_actions)
        self.pi_w = nn.Sequential(pi_w, View((num_options, num_actions)), nn.LogSoftmax(-1))
        beta = layer_init(nn.Linear(input_size, num_options), inits[0]) if ortho_init else nn.Linear(input_size, num_options)
        self.beta = nn.Sequential(beta, nn.Sigmoid())
        if pi_omega == 'epsilon':
            assert isinstance(pi_omega_epsilon, float)
            self.pi_omega = EpislonGreedyLayer(pi_omega_epsilon)
        else:
            pi_omega = layer_init(nn.Linear(input_size, num_options), inits[2]) if ortho_init else nn.Linear(input_size, num_options)

    def forward(self, input):
        pass