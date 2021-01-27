from rlpyt.models.discrete import OneHotLayer
from rlpyt.models.mlp import MlpModel
from rlpyt.models.oc import OptionCriticHead_IndependentPreprocessor, OptionCriticHead_SharedPreprocessor
from functools import partial
import torch
import torch.nn as nn
from torch.jit import script as tscr
import numpy as np
from typing import Tuple, List
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from rlpyt.utils.collections import namedarraytuple
RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work
class POMDPFfModel(nn.Module):
    """ Basic feedforward actor-critic model for discrete state space

    Args:
        input_classes (int): number of possible input states
        output_size: (int): Action space
        hidden_sizes (list): can be empty list for none (linear model).
        inits: tuple(ints): Orthogonal initialization for base, value, and policy (or None for standard init)
        nonlinearity (nn.Module): Nonlinearity applied in MLP component
    """
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 inits: [(float, float, float), None] = (np.sqrt(2), 1., 0.01),
                 nonlinearity: nn.Module = nn.ReLU
                 ):
        super().__init__()
        self._obs_ndim = 0
        self.preprocessor = tscr(OneHotLayer(input_classes))
        self.v = tscr(MlpModel(input_classes, hidden_sizes, 1, nonlinearity, inits[:-1] if inits is not None else inits))
        self.pi = tscr(nn.Sequential(MlpModel(input_classes, hidden_sizes, output_size, nonlinearity, inits[0::2] if inits is not None else inits), nn.Softmax(-1)))

    def forward(self, observation, prev_action, prev_reward):
        """ Compute action probabilities and value estimate

        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        obs_flat = self.preprocessor(observation.view(T * B))  # Onehot
        pi, v = self.pi(obs_flat), self.v(obs_flat).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v

class POMDPRnnModel(nn.Module):
    """ Basic recurrent actor-critic model for discrete state space

    Args:
        input_classes (int): number of possible input states
        output_size: (int): Action space
        rnn_type (str): Type of rnn layer to use 'gru' or 'lstm'
        rnn_size (int): Size of rnn layer
        hidden_sizes (list): can be empty list for none (linear model).
        inits: tuple(ints): Orthogonal initialization for base, value, and policy (or None for standard init)
        nonlinearity (nn.Module): Nonlinearity applied in MLP component
    """
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 rnn_type: str = 'lstm',
                 rnn_size: int = 128,
                 inits: [(float, float, float), None] = (np.sqrt(2), 1., 0.01),
                 nonlinearity: nn.Module = nn.ReLU
                 ):
        super().__init__()
        self._obs_ndim = 0
        self.rnn_type = rnn_type
        rnn_class = nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.preprocessor = tscr(OneHotLayer(input_classes))
        self.rnn = rnn_class(input_classes + output_size + 1, rnn_size)
        self.v = tscr(MlpModel(rnn_size, hidden_sizes, 1, nonlinearity, inits[:-1] if inits is not None else inits))
        self.pi = tscr(nn.Sequential(MlpModel(rnn_size, hidden_sizes, output_size, nonlinearity, inits[0::2] if inits is not None else inits), nn.Softmax(-1)))

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """ Compute action probabilities and value estimate

        NOTE: Rnn concatenates previous action and reward to input
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        # Convert init_rnn_state appropriately
        if init_rnn_state is not None:
            if self.rnn_type == 'gru':
                init_rnn_state = init_rnn_state.h  # namedarraytuple -> h
            else:
                init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        oh = self.preprocessor(observation)  # Leave in TxB format for lstm
        rnn_input = torch.cat([
            oh.view(T,B,-1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        rnn_out, h = self.rnn(rnn_input, init_rnn_state)
        rnn_out = rnn_out.view(T*B, -1)
        pi, v = self.pi(rnn_out), self.v(rnn_out).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=h[0], c=h[1] if self.rnn_type == 'lstm' else None)
        return pi, v, next_rnn_state

class POMDPOcFfModel(nn.Module):
    """ Basic feedforward option-critic model for discrete state space

    Args:
        input_classes (int): number of possible input states
        output_size (int): Size of action space
        option_size (int): Number of options
        hidden_sizes (list): can be empty list for none (linear model).
        inits: tuple(ints): Orthogonal initialization for base, value, and policy (or None for standard init)
        nonlinearity (nn.Module): Nonlinearity applied in MLP component
    """
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 option_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 inits: [(float, float, float), None] = (np.sqrt(2), 1., 0.01),
                 hidden_nonlinearity=torch.nn.Tanh,  # Module form.
                 use_interest=False,  # IOC sigmoid interest functions
                 use_diversity=False,  # TDEOC q entropy output
                 use_attention=False,
                 ):
        super().__init__()
        self._obs_ndim = 0
        self.preprocessor = tscr(OneHotLayer(input_classes))
        body_mlp_class = partial(MlpModel, hidden_sizes=hidden_sizes, output_size=None, nonlinearity=hidden_nonlinearity, inits=inits[:-1])  # MLP with no head (and potentially no body)
        # Seperate mlp processors for each head
        self.model = tscr(OptionCriticHead_IndependentPreprocessor(
            input_size=input_classes,
            input_module_class=body_mlp_class,
            output_size=output_size,
            option_size=option_size,
            intra_option_policy='discrete',
            use_interest=use_interest,
            use_diversity=use_diversity,
            use_attention=use_attention,
            orthogonal_init=True,
            orthogonal_init_base=inits[1],
            orthogonal_init_pol=inits[2]
        ))
        #self.v = tscr(MlpModel(input_classes, hidden_sizes, 1, nonlinearity, inits[:-1] if inits is not None else inits))
        #self.pi = tscr(nn.Sequential(MlpModel(input_classes, hidden_sizes, output_size, nonlinearity, inits[0::2] if inits is not None else inits), nn.Softmax(-1)))

    def forward(self, observation, prev_action, prev_reward):
        """ Compute per-option action probabilities, termination probabilities, value, option probabilities, and value entropy (if using)

        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        obs_flat = self.preprocessor(observation.view(T * B))  # Onehot
        pi, beta, q, pi_omega, q_ent = self.model(obs_flat)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, beta, q, pi_omega, q_ent = restore_leading_dims((pi, beta, q, pi_omega, q_ent), lead_dim, T, B)
        return pi, beta, q, pi_omega

class POMDPOcRnnModel(nn.Module):
    def __init__(self):
        super().__init__()
    pass

if __name__ == "__main__":
    test = POMDPFfModel(input_classes=10, output_size=2)
    print(3)