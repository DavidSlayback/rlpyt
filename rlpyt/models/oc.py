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
        return torch.softmax(self.pi_w(x), dim=-1)

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