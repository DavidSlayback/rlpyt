from rlpyt.models.discrete import OneHotLayer
from rlpyt.models.utils import apply_init, O_INIT_VALUES, get_rnn_class
from rlpyt.models.mlp import MlpModel, layer_init
from rlpyt.models.oc import OptionCriticHead_IndependentPreprocessor, OptionCriticHead_SharedPreprocessor, OptionCriticHead_IndependentPreprocessorWithRNN
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
IocRnnState = namedarraytuple("IocRnnState", ["pi", "beta", "q", "pi_omega", "interest"])
OcRnnState = namedarraytuple("OcRnnState", ["pi", "beta", "q", "pi_omega"])

ALL, NONE = 'All', 'None'
ac_previous_input_names = ['Pi', 'V']
def ac_name_to_int(acname: str) -> int:
    """Convert string name input to int flag to pass to submodels"""
    i = 0  # 0 if None
    i += (ac_previous_input_names[0] in acname)  # 1, 3 if pi
    i += (2*ac_previous_input_names[1] in acname)  # 2, 3 if v
    i = 3 if acname == ALL else i  # 3 if All
    return i
oc_previous_input_names = ['Pi', 'Beta', 'Q', 'PIo', 'Int']
def oc_name_to_array(ocname: str) -> np.ndarray:
    i = np.zeros(5, dtype=bool)
    for j, name in enumerate(oc_previous_input_names):
        if name in ocname: i[j] = True
    if ocname == 'All': i[:] = True
    return i

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
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
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
                 layer_norm: bool = False,
                 prev_action: int = 2,
                 prev_reward: int = 2,
                 ):
        super().__init__()
        self._obs_dim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        rnn_input_size = input_classes
        if prev_action: rnn_input_size += output_size  # Use previous action as input
        if prev_reward: rnn_input_size += 1  # Use previous reward as input
        self.rnn = rnn_class(rnn_input_size, rnn_size)  # Concat action, reward
        self.body = MlpModel(rnn_size, hidden_sizes, None, nn.ReLU, None)
        self.pi = nn.Sequential(nn.Linear(self.body.output_size, output_size), nn.Softmax(-1))
        self.v = nn.Linear(self.body.output_size, 1)
        if baselines_init:
            self.rnn.apply(apply_init); self.body.apply(apply_init)
            self.pi.apply(partial(apply_init, gain=O_INIT_VALUES['pi']))
            self.v.apply(partial(apply_init, gain=O_INIT_VALUES['v']))
        self.body, self.pi, self.v = tscr(self.body), tscr(self.pi), tscr(self.v)
        self.p_a = prev_action > 0
        self.p_r = prev_reward > 0

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_dim)
        if init_rnn_state is not None and self.rnn_is_lstm: init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        oh = self.preprocessor(observation)  # Leave in TxB format for lstm
        inp_list = [oh.view(T,B,-1)] + ([prev_action.view(T, B, -1)] if self.p_a else []) + ([prev_reward.view(T, B, 1)] if self.p_r else [])
        rnn_input = torch.cat(inp_list, dim=2)
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
                 layer_norm: bool = False,
                 prev_action: int = 2,
                 prev_reward: int = 2,
                 ):
        super().__init__()
        self._obs_dim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.body = MlpModel(input_classes, hidden_sizes, None, nn.ReLU, None)
        rnn_input_size = self.body.output_size
        if prev_action: rnn_input_size += output_size  # Use previous action as input
        if prev_reward: rnn_input_size += 1  # Use previous reward as input
        self.rnn = rnn_class(rnn_input_size, rnn_size)  # Concat action, reward
        self.pi = nn.Sequential(nn.ReLU(), nn.Linear(rnn_size, output_size), nn.Softmax(-1))
        self.v = nn.Sequential(nn.ReLU(), nn.Linear(rnn_size, 1))
        if baselines_init:
            self.rnn.apply(apply_init); self.body.apply(apply_init)
            self.pi.apply(partial(apply_init, gain=O_INIT_VALUES['pi']))
            self.v.apply(partial(apply_init, gain=O_INIT_VALUES['v']))
        self.body, self.pi, self.v = tscr(self.body), tscr(self.pi), tscr(self.v)
        self.p_a = prev_action > 0
        self.p_r = prev_reward > 0

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_dim)
        if init_rnn_state is not None and self.rnn_is_lstm: init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        oh = self.preprocessor(observation)  # Leave in TxB format for lstm
        features = self.body(oh)
        inp_list = [features.view(T,B,-1)] + ([prev_action.view(T, B, -1)] if self.p_a else []) + ([prev_reward.view(T, B, 1)] if self.p_r else [])
        rnn_input = torch.cat(inp_list, dim=2)
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
                 layer_norm: bool = False,
                 prev_action: int = 2,
                 prev_reward: int = 2,
                 ):
        super().__init__()
        self._obs_dim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        rnn_input_size = input_classes
        if prev_action: rnn_input_size += output_size  # Use previous action as input
        if prev_reward: rnn_input_size += 1  # Use previous reward as input
        self.rnn = rnn_class(rnn_input_size, rnn_size)  # Concat action, reward
        pi_inits = (O_INIT_VALUES['base'], O_INIT_VALUES['pi']) if baselines_init else None
        v_inits = (O_INIT_VALUES['base'], O_INIT_VALUES['v']) if baselines_init else None
        self.pi = nn.Sequential(MlpModel(rnn_size, hidden_sizes, output_size, nn.ReLU, pi_inits), nn.Softmax(-1))
        self.v = nn.Sequential(MlpModel(rnn_size, hidden_sizes, 1, nn.ReLU, v_inits))
        if baselines_init:
            self.rnn.apply(apply_init)
        self.pi, self.v = tscr(self.pi), tscr(self.v)
        self.p_a = prev_action > 0
        self.p_r = prev_reward > 0

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_dim)
        if init_rnn_state is not None and self.rnn_is_lstm: init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        oh = self.preprocessor(observation)  # Leave in TxB format for lstm
        inp_list = [oh.view(T,B,-1)] + ([prev_action.view(T, B, -1)] if self.p_a else []) + ([prev_reward.view(T, B, 1)] if self.p_r else [])
        rnn_input = torch.cat(inp_list, dim=2)
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
                 layer_norm: bool = False,
                 prev_action: int = 3,
                 prev_reward: int = 3,
                 ):
        super().__init__()
        self._obs_dim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.body_pi = MlpModel(input_classes, hidden_sizes, None, nn.ReLU, None)
        self.body_v = MlpModel(input_classes, hidden_sizes, None, nn.ReLU, None)
        rnn_input_size_pi = self.body_pi.output_size + (prev_action in [1,3]) * output_size + (prev_reward in [1,3])
        rnn_input_size_v = self.body_v.output_size + (prev_action in [2,3]) * output_size + (prev_reward in [2,3])
        self.rnn_pi = rnn_class(rnn_input_size_pi, rnn_size)  # Concat action, reward
        self.rnn_v = rnn_class(rnn_input_size_v, rnn_size)
        self.pi = nn.Sequential(nn.ReLU(), nn.Linear(rnn_size, output_size), nn.Softmax(-1))  # Need to activate after lstm
        self.v = nn.Sequential(nn.ReLU(), nn.Linear(rnn_size, 1))
        if baselines_init:
            self.body_pi.apply(apply_init); self.body_v.apply(apply_init)
            self.rnn_pi.apply(apply_init); self.rnn_v.apply(apply_init)
            self.pi.apply(partial(apply_init, O_INIT_VALUES['pi']))
            self.v.apply(partial(apply_init, O_INIT_VALUES['v']))
        self.body_pi, self.body_v, self.pi, self.v = tscr(self.body_pi), tscr(self.body_v), tscr(self.pi), tscr(self.v)
        self.p_a = prev_action
        self.p_r = prev_reward

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_dim)
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
        p_a, p_r = prev_action.view(T, B, -1), prev_reward.view(T, B, 1)
        pi_inp_list = [b_pi.view(T,B,-1)] + ([p_a] if self.p_a in [1,3] else []) + ([p_r] if self.p_r in [1,3] else [])
        v_inp_list = [b_pi.view(T, B, -1)] + ([p_a] if self.p_a in [2,3] else []) + ([p_r] if self.p_r in [2, 3] else [])
        rnn_input_pi = torch.cat(pi_inp_list, dim=2)
        rnn_input_v = torch.cat(v_inp_list, dim=2)
        rnn_pi, next_rnn_state_pi = self.rnn_pi(rnn_input_pi, init_rnn_pi)
        rnn_v, next_rnn_state_v = self.rnn_v(rnn_input_v, init_rnn_v)
        rnn_pi = rnn_pi.view(T*B, -1); rnn_v = rnn_v.view(T*B, -1)
        pi, v = self.pi(rnn_pi), self.v(rnn_v).squeeze(-1)
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
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
        shared_processor (bool): Whether to share model processor (MLP) between heads. Onehot is shared anyway
        rnn_placement (int): 0 for right after one-hot, 1 for right after MLP
        layer_norm (bool): True for layer-normalized GRU/LSTM
        prev_action (int): Flag to set which rnns get previous action. 0 means neither, 1 means pi, 2 means v, 3 means pi and v
        prev_reward (int): Flag to set which rnns get previous reward. 0 means neither, 1 means pi, 2 means v, 3 means pi and v
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
                 layer_norm: bool = False,
                 prev_action: str = 'All',
                 prev_reward: str = 'All',
                 ):
        super().__init__()
        pa = ac_name_to_int(prev_action)
        pr = ac_name_to_int(prev_reward)
        if shared_processor and rnn_placement == 0: self.model = POMDPRnnShared0Rnn(input_classes, output_size, rnn_type, rnn_size, hidden_sizes, inits is not None, layer_norm, pa, pr)
        elif shared_processor and rnn_placement == 1: self.model = POMDPRnnShared1Rnn(input_classes, output_size, rnn_type, rnn_size, hidden_sizes, inits is not None, layer_norm, pa, pr)
        elif not shared_processor and rnn_placement == 0: self.model = POMDPRnnUnshared0Rnn(input_classes, output_size, rnn_type, rnn_size, hidden_sizes, inits is not None, layer_norm, pa, pr)
        elif not shared_processor and rnn_placement == 1: self.model = POMDPRnnUnshared1Rnn(input_classes, output_size, rnn_type, rnn_size, hidden_sizes, inits is not None, layer_norm, pa, pr)

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
                baselines_init=True,
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
                baselines_init=True,
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

class POMDPOcRnnShared0Model(nn.Module):
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 option_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 rnn_type: str = 'gru',
                 rnn_size: int = 256,
                 baselines_init: bool = True,
                 layer_norm: bool = False,
                 use_interest: bool = False,  # IOC sigmoid interest functions
                 use_diversity: bool = False,  # TDEOC q entropy output
                 use_attention: bool = False,
                 prev_action: np.ndarray = np.ones(5, dtype=bool),
                 prev_reward: np.ndarray = np.ones(5, dtype=bool),
                 prev_option: np.ndarray = np.zeros(5, dtype=bool)
                 ):
        super().__init__()
        self._obs_ndim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.p_a, self.p_o, self.p_r = prev_action.any().item(), prev_option.any().item(), prev_reward.any().item()
        rnn_input_size = input_classes + (output_size * self.p_a) + (option_size * self.p_o) + self.p_r
        self.rnn = rnn_class(rnn_input_size, rnn_size)  # Concat action, reward
        self.body = MlpModel(rnn_size, hidden_sizes, None, nn.ReLU, None)
        self.oc = tscr(OptionCriticHead_SharedPreprocessor(
            input_size=self.body.output_size,
            output_size=output_size,
            option_size=option_size,
            intra_option_policy='discrete',
            use_interest=use_interest,
            use_diversity=use_diversity,
            use_attention=use_attention,
            baselines_init=baselines_init))
        if baselines_init:
            self.rnn.apply(partial(apply_init, gain=O_INIT_VALUES['lstm']))
            self.body.apply(apply_init)
        self.body = tscr(self.body)

    def forward(self, observation, prev_action, prev_reward, prev_option, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        if init_rnn_state is not None and self.rnn_is_lstm: init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        o = self.preprocessor(observation)
        inp_list = [o.view(T,B,-1)] + ([prev_action.view(T, B, -1)] if self.p_a else []) + ([prev_reward.view(T, B, 1)] if self.p_r else []) + ([prev_option.view(T, B, -1)] if self.p_o else [])
        rnn_input = torch.cat(inp_list, dim=2)
        rnn_out, next_rnn_state = self.rnn(rnn_input, init_rnn_state)
        rnn_out = rnn_out.view(T*B, -1)
        features = self.body(rnn_out)
        pi, beta, q, pi_omega, q_ent = self.oc(features)
        pi, beta, q, pi_omega, q_ent = restore_leading_dims((pi, beta, q, pi_omega, q_ent), lead_dim, T, B)
        if self.rnn_is_lstm: next_rnn_state = RnnState(next_rnn_state)
        return pi, beta, q, pi_omega, next_rnn_state


class POMDPOcRnnShared1Model(nn.Module):
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 option_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 rnn_type: str = 'gru',
                 rnn_size: int = 256,
                 baselines_init: bool = True,
                 layer_norm: bool = False,
                 use_interest: bool = False,  # IOC sigmoid interest functions
                 use_diversity: bool = False,  # TDEOC q entropy output
                 use_attention: bool = False,
                 prev_action: np.ndarray = np.ones(5, dtype=bool),
                 prev_reward: np.ndarray = np.ones(5, dtype=bool),
                 prev_option: np.ndarray = np.zeros(5, dtype=bool)
                 ):
        super().__init__()
        self._obs_ndim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.body = MlpModel(input_classes, hidden_sizes, None, nn.ReLU, None)
        self.p_a, self.p_o, self.p_r = prev_action.any().item(), prev_option.any().item(), prev_reward.any().item()
        rnn_input_size = self.body.output_size + (output_size * self.p_a) + (option_size * self.p_o) + self.p_r
        self.rnn = rnn_class(rnn_input_size, rnn_size)  # Concat action, reward
        self.oc = tscr(OptionCriticHead_SharedPreprocessor(
            input_size=rnn_size,
            output_size=output_size,
            option_size=option_size,
            intra_option_policy='discrete',
            use_interest=use_interest,
            use_diversity=use_diversity,
            use_attention=use_attention,
            baselines_init=baselines_init))
        if baselines_init:
            self.rnn.apply(partial(apply_init, gain=O_INIT_VALUES['lstm']))
            self.body.apply(apply_init)
        self.body = tscr(self.body)

    def forward(self, observation, prev_action, prev_reward, prev_option, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        if init_rnn_state is not None and self.rnn_is_lstm: init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        o = self.preprocessor(observation.view(T * B))
        features = self.body(o)
        inp_list = [features.view(T,B,-1)] + ([prev_action.view(T, B, -1)] if self.p_a else []) + ([prev_reward.view(T, B, 1)] if self.p_r else []) + ([prev_option.view(T, B, -1)] if self.p_o else [])
        rnn_input = torch.cat(inp_list, dim=2)
        rnn_out, next_rnn_state = self.rnn(rnn_input, init_rnn_state)
        rnn_out = rnn_out.view(T*B, -1)
        pi, beta, q, pi_omega, q_ent = self.oc(rnn_out)
        pi, beta, q, pi_omega, q_ent = restore_leading_dims((pi, beta, q, pi_omega, q_ent), lead_dim, T, B)
        if self.rnn_is_lstm: next_rnn_state = RnnState(next_rnn_state)
        return pi, beta, q, pi_omega, next_rnn_state


class POMDPOcRnnUnshared0Model(nn.Module):
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 option_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 rnn_type: str = 'gru',
                 rnn_size: int = 256,
                 baselines_init: bool = True,
                 layer_norm: bool = False,
                 use_interest: bool = False,  # IOC sigmoid interest functions
                 use_diversity: bool = False,  # TDEOC q entropy output
                 use_attention: bool = False,
                 prev_action: np.ndarray = np.ones(5, dtype=bool),
                 prev_reward: np.ndarray = np.ones(5, dtype=bool),
                 prev_option: np.ndarray = np.zeros(5, dtype=bool)
                 ):
        super().__init__()
        self._obs_ndim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.p_a, self.p_o, self.p_r = prev_action.any().item(), prev_option.any().item(), prev_reward.any().item()
        rnn_input_size = input_classes + (output_size * self.p_a) + (option_size * self.p_o) + self.p_r
        self.rnn = rnn_class(rnn_input_size, rnn_size)  # Concat action, reward
        body_mlp_class = partial(MlpModel, hidden_sizes=hidden_sizes, output_size=None, nonlinearity=nn.ReLU, inits=None)
        self.oc = tscr(OptionCriticHead_IndependentPreprocessor(
            input_size=rnn_size,
            input_module_class=body_mlp_class,
            output_size=output_size,
            option_size=option_size,
            intra_option_policy='discrete',
            use_interest=use_interest,
            use_diversity=use_diversity,
            use_attention=use_attention,
            baselines_init=baselines_init))
        if baselines_init:
            self.rnn.apply(partial(apply_init, gain=O_INIT_VALUES['lstm']))

    def forward(self, observation, prev_action, prev_reward, prev_option, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        if init_rnn_state is not None and self.rnn_is_lstm: init_rnn_state = tuple(init_rnn_state)  # namedarraytuple -> tuple (h, c)
        o = self.preprocessor(observation)
        inp_list = [o.view(T,B,-1)] + ([prev_action.view(T, B, -1)] if self.p_a else []) + ([prev_reward.view(T, B, 1)] if self.p_r else []) + ([prev_option.view(T, B, -1)] if self.p_o else [])
        rnn_input = torch.cat(inp_list, dim=2)
        rnn_out, next_rnn_state = self.rnn(rnn_input, init_rnn_state)
        rnn_out = rnn_out.view(T*B, -1)
        pi, beta, q, pi_omega, q_ent = self.oc(rnn_out)
        pi, beta, q, pi_omega, q_ent = restore_leading_dims((pi, beta, q, pi_omega, q_ent), lead_dim, T, B)
        if self.rnn_is_lstm: next_rnn_state = RnnState(next_rnn_state)
        return pi, beta, q, pi_omega, next_rnn_state


class POMDPOcRnnUnshared1Model(nn.Module):
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 option_size: int,
                 hidden_sizes: [List, Tuple, None] = None,
                 rnn_type: str = 'gru',
                 rnn_size: int = 256,
                 baselines_init: bool = True,
                 layer_norm: bool = False,
                 use_interest: bool = False,  # IOC sigmoid interest functions
                 use_diversity: bool = False,  # TDEOC q entropy output
                 use_attention: bool = False,
                 prev_action: np.ndarray = np.ones(5, dtype=bool),
                 prev_reward: np.ndarray = np.ones(5, dtype=bool),
                 prev_option: np.ndarray = np.zeros(5, dtype=bool)
                 ):
        super().__init__()
        self._obs_ndim = 0
        self.rnn_is_lstm = rnn_type != 'gru'
        self.preprocessor = tscr(OneHotLayer(input_classes))
        rnn_class = get_rnn_class(rnn_type, layer_norm)
        self.p_a, self.p_o, self.p_r = prev_action, prev_option, prev_reward
        body_mlp_class = partial(MlpModel, hidden_sizes=hidden_sizes, output_size=None, nonlinearity=nn.ReLU, inits=None)
        self.oc = OptionCriticHead_IndependentPreprocessorWithRNN(
            input_size=input_classes,
            input_module_class=body_mlp_class,
            rnn_module_class=rnn_class,
            output_size=output_size,
            option_size=option_size,
            rnn_size=rnn_size,
            intra_option_policy='discrete',
            use_interest=use_interest,
            use_diversity=use_diversity,
            use_attention=use_attention,
            baselines_init=baselines_init,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_option=prev_option
        )

    def forward(self, observation, prev_action, prev_reward, prev_option, init_rnn_state):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        if init_rnn_state is not None:
            if self.rnn_is_lstm:
                init_rnn_pi, init_rnn_beta, init_rnn_q, init_rnn_pi_omega, init_rnn_I = tuple(init_rnn_state)  # DualRnnState -> RnnState_pi, RnnState_v
                init_rnn_pi, init_rnn_q, init_rnn_beta, init_rnn_pi_omega, init_rnn_I = tuple(init_rnn_pi), tuple(init_rnn_q), tuple(init_rnn_beta), tuple(init_rnn_pi_omega), tuple(init_rnn_I)
            else:
                init_rnn_pi, init_rnn_beta, init_rnn_q, init_rnn_pi_omega, init_rnn_I = tuple(init_rnn_state)  # DualRnnState -> h, h
        else:
            init_rnn_pi, init_rnn_beta, init_rnn_q, init_rnn_pi_omega, init_rnn_I = None, None, None, None, None
        o = self.preprocessor(observation.view(T * B))
        pi, beta, q, pi_omega, q_ent, (nrs_pi, nrs_beta, nrs_q, nrs_pi_omega, nrs_I) = \
            self.oc(o, prev_action, prev_reward, prev_option, T, B, (init_rnn_pi, init_rnn_beta, init_rnn_q, init_rnn_pi_omega, init_rnn_I))
        pi, beta, q, pi_omega, q_ent = restore_leading_dims((pi, beta, q, pi_omega, q_ent), lead_dim, T, B)
        if self.rnn_is_lstm:
            next_rnn_state = IocRnnState(pi=RnnState(*nrs_pi), beta=RnnState(*nrs_beta), q=RnnState(*nrs_q), pi_omega=RnnState(*nrs_pi_omega), interest=RnnState(*nrs_I))
        else:
            next_rnn_state = IocRnnState(pi=nrs_pi, beta=nrs_beta, q=nrs_q, pi_omega=nrs_pi_omega, interest=nrs_I)
        return pi, beta, q, pi_omega, next_rnn_state

class POMDPOcRnnModel(nn.Module):
    """ Basic feedforward option-critic model for discrete state space

    Args:
        input_classes (int): number of possible input states
        output_size (int): Size of action space
        option_size (int): Number of options
        hidden_sizes (list): can be empty list for none (linear model).
        rnn_type (str): Either 'gru' or 'lstm'
        rnn_size (int): Number of units in recurrent layer
        shared_processor (bool): Shared MLP for each head or not
        rnn_placement (int): Specifies where in architecture to place rnn
            0: Immediately after one-hot
            1: After MLP. If shared MLP, there's one. If independent MLP, there's one for each head
        inits: tuple(ints): Orthogonal initialization for base, value, and policy (or None for standard init)
        use_interest (bool): Use an interest function (IOC)
        use_diversity (bool): Use termination diversity objective (TDEOC)
        use_attention(bool): Use attention mechanism (AOC)
        layer_norm (bool): True for layer-normalized GRU/LSTM
        prev_action (bool ndarray): Flag to set which rnns get previous action. [pi, beta, q, pio, interest]
        prev_reward (bool ndarray): Flag to set which rnns get previous reward. [pi, beta, q, pio, interest]
        prev_option (bool ndarray): Flag to set which rnns get previous reward. [pi, beta, q, pio, interest]
    """
    def __init__(self,
                 input_classes: int,
                 output_size: int,
                 option_size: int,
                 rnn_type: str = 'gru',
                 rnn_size: int = 256,
                 shared_processor: bool = False,
                 rnn_placement: int = 1,
                 hidden_sizes: [List, Tuple, None] = None,
                 inits: [(float, float, float), None] = (np.sqrt(2), 1., 0.01),
                 use_interest: bool = False,  # IOC sigmoid interest functions
                 use_diversity: bool = False,  # TDEOC q entropy output
                 use_attention: bool = False,
                 layer_norm: bool = True,
                 prev_action: str = 'All',
                 prev_reward: str = 'All',
                 prev_option: str = 'None'
                 ):
        super().__init__()
        prev_action = oc_name_to_array(prev_action)
        prev_reward = oc_name_to_array(prev_reward)
        prev_option = oc_name_to_array(prev_option)
        if shared_processor and rnn_placement == 0:
            self.model = POMDPOcRnnShared0Model(input_classes, output_size, option_size, hidden_sizes, rnn_type, rnn_size,
                                            inits is not None, layer_norm, use_interest, use_diversity, use_attention,
                                                prev_action, prev_reward, prev_option)
        elif shared_processor and rnn_placement == 1:
            self.model = POMDPOcRnnShared1Model(input_classes, output_size, option_size, hidden_sizes, rnn_type, rnn_size,
                                            inits is not None, layer_norm, use_interest, use_diversity, use_attention,
                                                prev_action, prev_reward, prev_option)
        elif not shared_processor and rnn_placement == 0:
            self.model = POMDPOcRnnUnshared0Model(input_classes, output_size, option_size, hidden_sizes, rnn_type, rnn_size,
                                            inits is not None, layer_norm, use_interest, use_diversity, use_attention,
                                                prev_action, prev_reward, prev_option)
        elif not shared_processor and rnn_placement == 1:
            self.model = POMDPOcRnnUnshared1Model(input_classes, output_size, option_size, hidden_sizes, rnn_type, rnn_size,
                                            inits is not None, layer_norm, use_interest, use_diversity, use_attention,
                                                prev_action, prev_reward, prev_option)

    def forward(self, observation, prev_action, prev_reward, prev_option, init_rnn_state):
        return self.model(observation, prev_action, prev_reward, prev_option, init_rnn_state)
