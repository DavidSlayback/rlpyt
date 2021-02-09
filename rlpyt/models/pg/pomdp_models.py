from rlpyt.models.discrete import OneHotLayer
from rlpyt.models.utils import apply_init, O_INIT_VALUES, get_rnn_class
from rlpyt.models.mlp import MlpModel, layer_init
from rlpyt.models.oc import OptionCriticHead_IndependentPreprocessor, OptionCriticHead_SharedPreprocessor
from functools import partial
import torch
import torch.nn as nn
from torch.jit import script as tscr
import numpy as np
from typing import Tuple, List
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from rlpyt.utils.collections import namedarraytuple, namedtuple
RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work
DualRnnState = namedarraytuple("DualRnnState", ["pi", "v"])
# GruState = namedarraytuple("GruState", ["h"])

class ScriptedRNN(nn.Module):
    """ Workaround from PyTorch issue #32976 for scripting RNNs

    Call LSTM/GRU's forward from within a scripted Module
    """
    def __init__(self, rnn_instance):
        super().__init__()
        self.scripted_lstm = rnn_instance

    def forward(self, input):
        return self.scripted_lstm(input)

class POMDPFfModel(nn.Module):
    """ Basic feedforward actor-critic model for discrete state space

    Args:
        input_classes (int): number of possible input states
        output_size: (int): Action space
        hidden_sizes (list): can be empty list for none (linear model).
        inits: tuple(ints): Orthogonal initialization for base, value, and policy (or None for standard init)
        nonlinearity (nn.Module): Nonlinearity applied in MLP component
        shared_processor (bool): Whether to share model processor (MLP) between heads. Onehot is shared anyway.
          **Unshared outperforms shared in sample-efficiency, but is slower
    """
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 inits: [(float, float, float), None] = (np.sqrt(2), 1., 0.01),
                 nonlinearity: nn.Module = nn.ReLU,
                 shared_processor: bool = False
                 ):
        super().__init__()
        self._obs_ndim = 0
        if shared_processor:
            self.preprocessor = tscr(nn.Sequential(OneHotLayer(input_classes), MlpModel(input_classes, hidden_sizes, None, nonlinearity, inits[:-1] if inits is not None else inits)))
            self.v = tscr(layer_init(nn.Linear(hidden_sizes[-1], 1), inits[1]) if inits else nn.Linear(hidden_sizes[-1], 1))
            self.pi = tscr(nn.Sequential(layer_init(nn.Linear(hidden_sizes[-1], output_size), inits[1]) if inits else nn.Linear(hidden_sizes[-1], output_size), nn.Softmax(-1)))
        else:
            self.preprocessor = tscr(OneHotLayer(input_classes))
            self.v = tscr(MlpModel(input_classes, hidden_sizes, 1, nonlinearity, inits[:-1] if inits is not None else inits))
            self.pi = tscr(nn.Sequential(MlpModel(input_classes, hidden_sizes, output_size, nonlinearity, inits[0::2] if inits is not None else inits), nn.Softmax(-1)))

    def forward(self, observation, prev_action, prev_reward):
        """ Compute action probabilities and value estimate

        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_dim)
        obs_flat = self.preprocessor(observation.view(T * B))  # Onehot
        pi, v = self.pi(obs_flat), self.v(obs_flat).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        return pi, v

# NOTE: In baselines, LSTMS are only supported in shared part of network (not necessarily principled though)
# Enumerate types of POMDP w/RNN. Include one-hot in these
class POMDPRnnShared0Rnn(nn.Module):
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 rnn_type: str = 'gru',
                 rnn_size: int = 256,
                 hidden_sizes: [List, Tuple] = None,
                 baselines_init: bool = True,
                 layer_norm: bool = False
                 ):
        super().__init__()
        self._obs_dim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.rnn = rnn_class(input_classes + output_size + 1, rnn_size)  # Concat action, reward
        self.body = MlpModel(rnn_size, hidden_sizes, None, nn.ReLU, None)
        self.pi = nn.Sequential(nn.Linear(self.body.output_size, output_size), nn.Softmax(-1))
        self.v = nn.Linear(self.body.output_size, 1)
        if baselines_init:
            self.rnn.apply(apply_init); self.body.apply(apply_init)
            self.pi.apply(partial(apply_init, gain=O_INIT_VALUES['pi']))
            self.v.apply(partial(apply_init, gain=O_INIT_VALUES['v']))
        self.body, self.pi, self.v = tscr(self.body), tscr(self.pi), tscr(self.v)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_dim)
        if init_rnn_state is not None and self.rnn_is_lstm: init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        oh = self.preprocessor(observation)  # Leave in TxB format for lstm
        rnn_input = torch.cat([
            oh.view(T,B,-1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        rnn_out, next_rnn_state = self.rnn(rnn_input, init_rnn_state)
        rnn_out = rnn_out.view(T*B, -1)
        rnn_out = self.body(rnn_out)
        pi, v = self.pi(rnn_out), self.v(rnn_out).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        if self.rnn_is_lstm: next_rnn_state = RnnState(next_rnn_state)
        return pi, v, next_rnn_state

class POMDPRnnShared1Rnn(nn.Module):
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 rnn_type: str = 'gru',
                 rnn_size: int = 256,
                 hidden_sizes: [List, Tuple] = None,
                 baselines_init: bool = True,
                 layer_norm: bool = False
                 ):
        super().__init__()
        self._obs_dim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.body = MlpModel(input_classes, hidden_sizes, None, nn.ReLU, None)
        self.rnn = rnn_class(self.body.output_size + output_size + 1, rnn_size)  # Concat action, reward
        self.pi = nn.Sequential(nn.ReLU(), nn.Linear(rnn_size, output_size), nn.Softmax(-1))
        self.v = nn.Sequential(nn.ReLU(), nn.Linear(rnn_size, 1))
        if baselines_init:
            self.rnn.apply(apply_init); self.body.apply(apply_init)
            self.pi.apply(partial(apply_init, gain=O_INIT_VALUES['pi']))
            self.v.apply(partial(apply_init, gain=O_INIT_VALUES['v']))
        self.body, self.pi, self.v = tscr(self.body), tscr(self.pi), tscr(self.v)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_dim)
        if init_rnn_state is not None and self.rnn_is_lstm: init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        oh = self.preprocessor(observation)  # Leave in TxB format for lstm
        features = self.body(oh)
        rnn_input = torch.cat([
            features.view(T,B,-1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        rnn_out, next_rnn_state = self.rnn(rnn_input, init_rnn_state)
        rnn_out = rnn_out.view(T*B, -1)
        pi, v = self.pi(rnn_out), self.v(rnn_out).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        if self.rnn_is_lstm: next_rnn_state = RnnState(next_rnn_state)
        return pi, v, next_rnn_state

class POMDPRnnUnshared0Rnn(nn.Module):
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 rnn_type: str = 'gru',
                 rnn_size: int = 256,
                 hidden_sizes: [List, Tuple] = None,
                 baselines_init: bool = True,
                 layer_norm: bool = False
                 ):
        super().__init__()
        self._obs_dim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.rnn = rnn_class(input_classes + output_size + 1, rnn_size)  # Concat action, reward
        pi_inits = (O_INIT_VALUES['base'], O_INIT_VALUES['pi']) if baselines_init else None
        v_inits = (O_INIT_VALUES['base'], O_INIT_VALUES['v']) if baselines_init else None
        self.pi = nn.Sequential(MlpModel(rnn_size, hidden_sizes, output_size, nn.ReLU, pi_inits), nn.Softmax(-1))
        self.v = nn.Sequential(MlpModel(rnn_size, hidden_sizes, 1, nn.ReLU, v_inits))
        if baselines_init:
            self.rnn.apply(apply_init)
        self.pi, self.v = tscr(self.pi), tscr(self.v)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_dim)
        if init_rnn_state is not None and self.rnn_is_lstm: init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        oh = self.preprocessor(observation)  # Leave in TxB format for lstm
        rnn_input = torch.cat([
            oh.view(T,B,-1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        rnn_out, next_rnn_state = self.rnn(rnn_input, init_rnn_state)
        rnn_out = rnn_out.view(T*B, -1)
        pi, v = self.pi(rnn_out), self.v(rnn_out).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        if self.rnn_is_lstm: next_rnn_state = RnnState(next_rnn_state)
        return pi, v, next_rnn_state

class POMDPRnnUnshared1Rnn(nn.Module):
    """Special case, rnn after processing for each head

    Going to handle the hidden state by adding an extra dimension
    """
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 rnn_type: str = 'gru',
                 rnn_size: int = 256,
                 hidden_sizes: [List, Tuple] = None,
                 baselines_init: bool = True,
                 layer_norm: bool = False
                 ):
        super().__init__()
        self._obs_dim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.body_pi = MlpModel(input_classes, hidden_sizes, None, nn.ReLU, None)
        self.body_v = MlpModel(input_classes, hidden_sizes, None, nn.ReLU, None)
        self.rnn_pi = rnn_class(self.body_pi.output_size + output_size + 1, rnn_size)  # Concat action, reward
        self.rnn_v = rnn_class(self.body_v.output_size + output_size + 1, rnn_size)
        self.pi = nn.Sequential(nn.ReLU(), nn.Linear(rnn_size, output_size), nn.Softmax(-1))  # Need to activate after lstm
        self.v = nn.Sequential(nn.ReLU(), nn.Linear(rnn_size, 1))
        if baselines_init:
            self.body_pi.apply(apply_init); self.body_v.apply(apply_init)
            self.rnn_pi.apply(apply_init); self.rnn_v.apply(apply_init)
            self.pi.apply(partial(apply_init, O_INIT_VALUES['pi']))
            self.v.apply(partial(apply_init, O_INIT_VALUES['v']))
        self.body_pi, self.body_v, self.pi, self.v = tscr(self.body_pi), tscr(self.body_v), tscr(self.pi), tscr(self.v)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_dim)
        # init_rnn_state is either tuple(h,c), h, or None.
        # Typical h is nlayers*ndir x B x hsize. I can't stack in 1st 2 dims without
        # if init_rnn_state is not None:
        #     if self.rnn_is_lstm:
        #         h_pi, h_v = init_rnn_state.h.chunk(2, dim=-1)
        #         c_pi, c_v = init_rnn_state.c.chunk(2, dim=-1)
        #         init_rnn_pi, init_rnn_v = (h_pi, c_pi), (h_v, c_v)
        #     else:
        #         init_rnn_pi, init_rnn_v = init_rnn_state.chunk(2, dim=-1)
        # else:
        #     init_rnn_pi, init_rnn_v = None, None
        if init_rnn_state is not None:
            if self.rnn_is_lstm:
                init_rnn_pi, init_rnn_v = tuple(init_rnn_state)  # DualRnnState -> RnnState_pi, RnnState_v
                init_rnn_pi, init_rnn_v = tuple(init_rnn_pi), tuple(init_rnn_v)
            else:
                init_rnn_pi, init_rnn_v = tuple(init_rnn_state)  # DualRnnState -> h, h
        else:
            init_rnn_pi, init_rnn_v = None, None
        o_flat = self.preprocessor(observation.view(T*B))
        b_pi, b_v = self.body_pi(o_flat), self.body_v(o_flat)
        rnn_input_pi = torch.cat([b_pi.view(T,B,-1),prev_action.view(T, B, -1),prev_reward.view(T, B, 1),], dim=2)
        rnn_input_v = torch.cat([b_v.view(T, B, -1), prev_action.view(T, B, -1), prev_reward.view(T, B, 1), ], dim=2)
        rnn_pi, next_rnn_state_pi = self.rnn_pi(rnn_input_pi, init_rnn_pi)
        rnn_v, next_rnn_state_v = self.rnn_pi(rnn_input_v, init_rnn_v)
        rnn_pi = rnn_pi.view(T*B, -1); rnn_v = rnn_v.view(T*B, -1)
        pi, v = self.pi(rnn_pi), self.v(rnn_v).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        # if self.rnn_is_lstm:
        #     next_rnn_state = RnnState(h=torch.cat((next_rnn_state_pi[0], next_rnn_state_v[0]), dim=-1),
        #                               c=torch.cat((next_rnn_state_pi[1], next_rnn_state_v[1]), dim=-1))
        # else:
        #     next_rnn_state = torch.cat((next_rnn_state_pi, next_rnn_state_v), dim=-1)
        if self.rnn_is_lstm:
            next_rnn_state = DualRnnState(RnnState(*next_rnn_state_pi), RnnState(*next_rnn_state_v))
        else:
            next_rnn_state = DualRnnState(next_rnn_state_pi, next_rnn_state_v)
        return pi, v, next_rnn_state


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
        shared_processor (bool): Whether to share model processor (MLP) between heads. Onehot is shared anyway
        rnn_placement (int): 0 for right after one-hot, 1 for right after MLP
    """
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 rnn_type: str = 'gru',
                 rnn_size: int = 128,
                 inits: [(float, float, float), None] = (np.sqrt(2), 1., 0.01),
                 shared_processor: bool = False,
                 rnn_placement: int = 1,
                 layer_norm: bool = False
                 ):
        super().__init__()
        if shared_processor and rnn_placement == 0: self.model = POMDPRnnShared0Rnn(input_classes, output_size, rnn_type, rnn_size, hidden_sizes, inits is not None, layer_norm)
        elif shared_processor and rnn_placement == 1: self.model = POMDPRnnShared1Rnn(input_classes, output_size, rnn_type, rnn_size, hidden_sizes, inits is not None, layer_norm)
        elif not shared_processor and rnn_placement == 0: self.model = POMDPRnnUnshared0Rnn(input_classes, output_size, rnn_type, rnn_size, hidden_sizes, inits is not None, layer_norm)
        elif not shared_processor and rnn_placement == 1: self.model = POMDPRnnUnshared1Rnn(input_classes, output_size, rnn_type, rnn_size, hidden_sizes, inits is not None, layer_norm)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        return self.model(observation, prev_action, prev_reward, init_rnn_state)

class POMDPOcFfModel(nn.Module):
    """ Basic feedforward option-critic model for discrete state space

    Args:
        input_classes (int): number of possible input states
        output_size (int): Size of action space
        option_size (int): Number of options
        hidden_sizes (list): can be empty list for none (linear model).
        inits: tuple(ints): Orthogonal initialization for base, value, and policy (or None for standard init)
        shared_processor (bool): Whether to share MLP among heads or reuse
        hidden_nonlinearity (nn.Module): Nonlinearity applied in MLP component
        use_interest (bool): Sigmoid interest functions
        use_diversity (bool): Q entropy output
        use_attention (bool): Attention masking of state **NOT IMPLEMENTED**
    """
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 option_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 inits: [(float, float, float), None] = (np.sqrt(2), 1., 0.01),
                 shared_processor: bool = True,
                 hidden_nonlinearity=torch.nn.ReLU,  # Module form.
                 use_interest=False,  # IOC sigmoid interest functions
                 use_diversity=False,  # TDEOC q entropy output
                 use_attention=False,
                 ):
        super().__init__()
        self._obs_ndim = 0
        self.preprocessor = tscr(OneHotLayer(input_classes))
        body_mlp_class = partial(MlpModel, hidden_sizes=hidden_sizes, output_size=None, nonlinearity=hidden_nonlinearity, inits=inits[:-1])  # MLP with no head (and potentially no body)
        if shared_processor:
            # Same mlp for all heads
            self.model = tscr(nn.Sequential(body_mlp_class(input_classes), OptionCriticHead_SharedPreprocessor(
                input_size=hidden_sizes[-1],
                output_size=output_size,
                option_size=option_size,
                intra_option_policy='discrete',
                use_interest=use_interest,
                use_attention=use_attention,
                use_diversity=use_diversity,
                orthogonal_init=True,
                orthogonal_init_base=inits[1],
                orthogonal_init_pol=inits[2]
            )))
        else:
            # Seperate mlp processors for each head (though if using diversity, q entropy and q share mlp
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
    """ Basic feedforward option-critic model for discrete state space

    Args:
        input_classes (int): number of possible input states
        output_size (int): Size of action space
        option_size (int): Number of options
        hidden_sizes (list): can be empty list for none (linear model).
        rnn_type (str): Either 'gru' or 'lstm'
        rnn_size (int): Number of units in recurrent layer
        inits: tuple(ints): Orthogonal initialization for base, value, and policy (or None for standard init)
        nonlinearity (nn.Module): Nonlinearity applied in MLP component
        use_interest (bool): Use an interest function (IOC)
        use_diversity (bool): Use termination diversity objective (TDEOC)
        use_attention(bool): Use attention mechanism (AOC)
    """
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 option_size: int,
                 rnn_type: str = 'gru',
                 rnn_size: int = 128,
                 rnn_placement: int = 1,
                 hidden_sizes: [List, Tuple, None] = None,
                 inits: [(float, float, float), None] = (np.sqrt(2), 1., 0.01),
                 shared_processor: bool = False,
                 hidden_nonlinearity=torch.nn.ReLU,  # Module form.
                 use_interest: bool = False,  # IOC sigmoid interest functions
                 use_diversity: bool = False,  # TDEOC q entropy output
                 use_attention: bool = False,
                 ):
        super().__init__()
        self._obs_ndim = 0
        self.preprocessor = tscr(OneHotLayer(input_classes))
        self.rnn_type = rnn_type
        rnn_class = nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.rnn = rnn_class(input_classes + output_size + 1, rnn_size)  # At some point, want to put option in here too
        body_mlp_class = partial(MlpModel, hidden_sizes=hidden_sizes, output_size=None, nonlinearity=hidden_nonlinearity, inits=inits[:-1])  # MLP with no head (and potentially no body)
        # Seperate mlp processors for each head
        self.model = tscr(OptionCriticHead_IndependentPreprocessor(
            input_size=rnn_size,
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
        pi, beta, q, pi_omega, q_ent = self.model(rnn_out)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, beta, q, pi_omega, q_ent = restore_leading_dims((pi, beta, q, pi_omega, q_ent), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        if self.rnn_type == 'gru':
            next_rnn_state = GruState(h=h)
        else:
            next_rnn_state = RnnState(*h)
        return pi, beta, q, pi_omega, next_rnn_state