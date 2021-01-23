from rlpyt.models.discrete import OneHotLayer
from rlpyt.models.mlp import MlpModel
import torch.nn as nn
from torch.jit import script as tscr
import numpy as np
from typing import Tuple, List
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

class POMDPFfModel(nn.Module):
    """ Basic feedforward actor-critic model for discrete state space

    Args:
        input_classes (int): number of possible input states
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        inits: tuple(ints): Orthogonal initialization for base, value, and policy (or None for standard init)
        nonlinearity (nn.Module): Nonlinearity applied in MLP component
    """
    def __init__(self,
                 input_classes: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 output_size: [int, None] = None,
                 inits: [(float, float, float), None] = (np.sqrt(2), 1., 0.01),
                 nonlinearity: nn.Module = nn.ReLU
                 ):
        super().__init__()
        self._obs_ndim = 1
        self.base_model = tscr(OneHotLayer(input_classes))
        self.v = tscr(MlpModel(input_classes, hidden_sizes, 1, nonlinearity, inits[:-1] if inits is not None else inits))
        self.pi = tscr(nn.Sequential(MlpModel(input_classes, hidden_sizes, output_size, nonlinearity, inits[0::2] if inits is not None else inits), nn.Softmax(-1)))

    def forward(self, observation, prev_action, prev_reward):
        """ Compute action probabilities and value estimate

        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        obs_flat = self.base_model(observation.view(T * B, -1))  # Onehot
        pi, v = self.pi(obs_flat), self.v(obs_flat).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v

class POMDPRnnModel(nn.Module):
    def __init__(self):
        super().__init__()
    pass

class POMDPOcModel(nn.Module):
    def __init__(self):
        super().__init__()
    pass