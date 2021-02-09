from typing import Callable

import torch
from torch import nn
from torch.nn import Module, Sequential, Linear
import torch.nn.functional as F
import numpy as np
from functools import partial
from rlpyt.models.utils import layer_init, EpislonGreedyLayer, View, Dummy, O_INIT_VALUES, apply_init
#NORM_EPS = 1e-6
"""
Option components:
- Attention (AOC): In FourRooms (one-hot view), attentions are per-option, elemwise applied to state. [0,1] mask. Initialized randomly
  Extra loss: the sum of absolute differences between attentions (for each option) of adjacent states in a trajectories. Ignore for now, details are missing
  
- Interest (IOC): Pretty simple. Sigmoid interest function which additionally parameterizes pi_omega. 
  - LRS: [main=3e-4, pi_o=1e-4, interest=1e-4, term=5e-7]. Note term, int, pi_o updated once in ppo, not multiple
  - Need to normalize for avg option value

- Diversity-enriched OC (TDEOC): 
  - Pseudo-reward for diversity while learning (DEOC): Bonus reward has entropy of each intra-option policy, entropy of pi_omega, and entropy between intra-option policies. Only the last matters (other 3 are baseline)
    - R_full(s,a) = (1-tau)R(s,a) + (tau)R_bonus(s)
  - Diversity in termination (TDEOC): D(s) is diversity of options at a given state. Compute by standardizing samples of r_bonus(s) in buffer
    - D(s) = (R_bonus(s) - mu_{R_bonus}) / sigma_{R_bonus}
    
  -Augmenting the reward with R_bonus for tasks with very sparse rewards can cause the agent to
    prioritize diversifying options over learning the task. To mitigate this, we avoid the reward
    augmentation step in all our sparse reward tasks including the four-rooms task. Instead
    of standardizing the diversity values, use update a moving sum of all values observed in
    the current run and center the diversity around the moving mean instead. For more than
    three options, the diversity is computed by sampling six pairs of options and averaging
    the respective cross entropy
    
  - The pseudo reward bonus is scaled down depending on the task with the intention of prioritizing
    task reward. The diversity term is calculated using cross entropy, as stated in the Eq. (2).
    We compute the softmax of the action distribution before computing the cross entropy to
    ensure Rbonus remains positive. As mentioned earlier, for the TMaze(continuous) task, we
    avoid augmenting the reward with the diversity term
    
  - Due to discrete action space, diversity is computed by taking the softmax of the logits of the policy network. 
    Rest of the implementation is consistent with the continuous control case discussed above. Due to the sparse reward
    situation, we again do not augment the reward with the diversity term
    
  - LRS: [3e-4 main, 5e-7 term (each epoch)
  - vpred ent shares state processor
    

- Termination Critic (ACTC): 

- PPOC: term is 5e-7, but each epoch
"""

class DiscreteIntraOptionPolicy(Module):
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
                 ortho_init=True,
                 ortho_init_value=O_INIT_VALUES['pi']):
        super().__init__()
        pi = layer_init(Linear(input_size, num_options * num_actions), ortho_init_value) if ortho_init else Linear(input_size, num_options * num_actions)
        self.pi = Sequential(pi, View((num_options, num_actions)))

    def forward(self, x):
        return torch.softmax(self.pi(x), dim=-1)

class ContinuousIntraOptionPolicy(Module):
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
                 ortho_init_value=O_INIT_VALUES['pi'],
                 init_log_std=0.,
                 mu_nonlinearity=nn.Identity):
        super().__init__()
        pi_mu = layer_init(Linear(input_size, num_options * num_actions), ortho_init_value) if ortho_init else Linear(input_size, num_options * num_actions)
        self.pi_mu = Sequential(pi_mu, View((num_options, num_actions)), mu_nonlinearity())
        self.log_std = nn.Parameter(init_log_std * torch.ones((num_options, num_actions)))  # State-independent vector of log_std

    def forward(self, x):
        return self.pi_mu(x), self.log_std

class OptionCriticHead_SharedPreprocessor(Module):
    """ Option-critic end output with optional components. Assumes input from shared state preprocessor

    Args:
        input_size (int): Number of inputs
        output_size (int): Number of outputs (per option)
        option_size (int): Number of options (O)
        intra_option_policy (str): Type of intra-option policy (either discrete or continuous)
        intra_option_kwargs (dict, None): Extra args for intra-option policy (e.g., init_log_std and mu_nonlinearity for continuous)
        use_interest (bool): If true, apply sigmoid interest function to policy over options (pi_omega)
        use_diversity (bool): If true, output a learned per-option Q(s,o) entropy
        use_attention (bool): If true, apply a per-option soft attention mechanism to the incoming state. NOT CURRENTLY IMPLEMENTED
        baselines_init (bool): If true, use orthogonal initialization
        NORM_EPS (float): Normalization constant added to pi_omega normalization to prevent instability.
    Outputs:
        pi(s): Per-option intra-option policy [BxOxA for discrete, BxO (mu, log_std) for continuous]
        beta(s): Per-option termination probability [BxO]
        q(s): Per-option value [BxO]
        pi_omega(s): Softmax policy over options (parameterized by interest function if using) [BxO]
        q_entropy(s): Per-option value entropy (or 0 if not learning) [BxO]

    """
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 option_size: int,
                 intra_option_policy: str,
                 intra_option_kwargs: [dict, None] = None,
                 use_interest: bool = False,
                 use_diversity: bool = False,
                 use_attention: bool = False,
                 baselines_init: bool = True,
                 NORM_EPS: float = 1e-6
                 ):
        super().__init__()
        if intra_option_kwargs is None: intra_option_kwargs = {}
        self.use_interest = use_interest
        self.use_diversity = use_diversity
        self.use_attention = use_attention
        self.NORM_EPS = NORM_EPS
        pi_class = DiscreteIntraOptionPolicy if intra_option_policy == 'discrete' else ContinuousIntraOptionPolicy
        self.pi = pi_class(input_size, option_size, output_size, ortho_init=baselines_init, **intra_option_kwargs)
        self.beta = nn.Sequential(Linear(input_size, option_size), nn.Sigmoid())
        self.q = Linear(input_size, option_size)
        self.q_ent = Linear(input_size, option_size) if use_diversity else Dummy(option_size, out_value=0.)
        self.pi_omega = Sequential(Linear(input_size, option_size), nn.Softmax(-1))
        self.interest = Sequential(Linear(input_size, option_size), nn.Sigmoid()) if use_interest else Dummy(
            option_size)
        if baselines_init:
            init_v, init_pi = O_INIT_VALUES['v'], O_INIT_VALUES['pi']
            self.beta.apply(apply_init); self.q_ent.apply(apply_init); self.interest.apply(apply_init)
            self.pi_omega.apply(partial(apply_init, gain=init_pi))
            self.q.apply(partial(apply_init, gain=init_v))

    def forward(self, x):
        pi_I = self.pi_omega(x) * self.interest(x)  # Interest-parameterized pi_omega
        if self.use_interest:  # Avoid messing with computations if we don't have to
            pi_I.add_(self.NORM_EPS)  # Add eps to avoid instability
            pi_I.div_(pi_I.sum(-1, keepdim=True))  # Normalize so probabilities add to 1
        return self.pi(x), self.beta(x), self.q(x), pi_I, self.q_ent(x)


class OptionCriticHead_IndependentPreprocessor(Module):
    """ Option-critic end output with optional components. Creates separate processor networks for heads

    Args:
        input_size (int): Number of inputs (to torch module class)
        input_module_class (nn.Module): Torch Module class to use for each option-critic head. Must have output_size property
        output_size (int): Number of outputs (per option)
        option_size (int): Number of options (O)
        input_module_kwargs (dict): Additional arguments to provide to input module on construction
        intra_option_policy (str): Type of intra-option policy (either discrete or continuous),
        use_interest (bool): If true, apply sigmoid interest function to policy over options (pi_omega)
        use_diversity (bool): If true, output a learned per-option Q(s,o) entropy
        use_attention (bool): If true, apply a per-option soft attention mechanism to the incoming state. NOT CURRENTLY IMPLEMENTED
        baselines_init (bool): If true, use orthogonal initialization
        NORM_EPS (float): Normalization constant added to pi_omega normalization to prevent instability.
    Outputs:
        pi(s): Per-option intra-option policy [BxOxA for discrete, BxO (mu, log_std) for continuous]
        beta(s): Per-option termination probability [BxO]
        q(s): Per-option value [BxO]
        pi_omega(s): Softmax policy over options (parameterized by interest function if using) [BxO]
        q_entropy(s): Per-option value entropy (or 0 if not learning) [BxO]

    """

    def __init__(self,
                 input_size: int,
                 input_module_class: Callable,
                 output_size: int,
                 option_size: int,
                 intra_option_policy: str,
                 intra_option_kwargs: [dict, None] = None,
                 input_module_kwargs: [dict, None] = None,
                 use_interest: bool = False,
                 use_diversity: bool = False,
                 use_attention: bool = False,
                 baselines_init: bool = True,
                 NORM_EPS: float = 1e-6
                 ):
        super().__init__()
        if input_module_kwargs is None: input_module_kwargs = {}  # Assume module has all necessary arguments
        if intra_option_kwargs is None: intra_option_kwargs = {}
        input_module_kwargs = {**input_module_kwargs, **{'input_size': input_size}}  # Add input size
        intra_option_kwargs = {**intra_option_kwargs}
        self.use_interest = use_interest
        self.use_diversity = use_diversity
        self.use_attention = use_attention
        self.NORM_EPS = NORM_EPS
        pi_class = DiscreteIntraOptionPolicy if intra_option_policy == 'discrete' else ContinuousIntraOptionPolicy
        # Instantiate independent preprocessors for pi, pi_omega, q (and entropy), interest, and termination heads
        pi_proc, pi_omega_proc, q_proc, int_proc, beta_proc = [input_module_class(**input_module_kwargs) for _ in range(5)]
        if baselines_init:
            pi_proc.apply(apply_init); pi_omega_proc.apply(apply_init); q_proc.apply(apply_init); int_proc.apply(apply_init); beta_proc.apply(apply_init)
        input_size = pi_proc.output_size
        self.pi = Sequential(pi_proc, pi_class(input_size, option_size, output_size, ortho_init=baselines_init, **intra_option_kwargs))
        self.beta = Sequential(beta_proc, Linear(input_size, option_size), nn.Sigmoid())
        self.q = Sequential(q_proc, Linear(input_size, option_size))
        self.q_ent = Sequential(q_proc, Linear(input_size, option_size)) if use_diversity else Dummy(option_size,
                                                                                                     out_value=0.)
        self.pi_omega = Sequential(pi_omega_proc, Linear(input_size, option_size), nn.Softmax(-1))
        self.interest = Sequential(int_proc, Linear(input_size, option_size), nn.Sigmoid()) if use_interest else Dummy(
            option_size)
        if baselines_init:
            init_v, init_pi = O_INIT_VALUES['v'], O_INIT_VALUES['pi']
            self.beta[1].apply(apply_init)
            self.pi_omega[1].apply(partial(apply_init, gain=init_pi))
            self.q[1].apply(partial(apply_init, gain=init_v))
            if use_interest: self.interest[1].apply(apply_init)
            if use_diversity: self.q_ent[1].apply(apply_init)

    def forward(self, x):
        pi_I = self.pi_omega(x) * self.interest(x)  # Interest-parameterized pi_omega
        if self.use_interest:  # Avoid messing with computations if we don't have to
            pi_I.add_(self.NORM_EPS)  # Add eps to avoid instability
            pi_I.div_(pi_I.sum(-1, keepdim=True))  # Normalize so probabilities add to 1
        return self.pi(x), self.beta(x), self.q(x), pi_I, self.q_ent(x)

class OptionCriticHead_IndependentPreprocessorWithRNN(nn.Module):
    """ Option-critic end output with optional components. Creates separate processor networks for heads. Rnn

    Args:
        input_size (int): Number of inputs (to torch module class)
        input_module_class (nn.Module): Torch Module class to use for each option-critic head. Must have output_size property
        rnn_module_class (nn.Module): Torch Module class to use for RNN for each option-critic head.
        rnn_size (int)
        output_size (int): Number of outputs (per option)
        option_size (int): Number of options (O)
        input_module_kwargs (dict): Additional arguments to provide to input module on construction
        intra_option_policy (str): Type of intra-option policy (either discrete or continuous),
        use_interest (bool): If true, apply sigmoid interest function to policy over options (pi_omega)
        use_diversity (bool): If true, output a learned per-option Q(s,o) entropy
        use_attention (bool): If true, apply a per-option soft attention mechanism to the incoming state. NOT CURRENTLY IMPLEMENTED
        baselines_init (bool): If true, use orthogonal initialization
        NORM_EPS (float): Normalization constant added to pi_omega normalization to prevent instability.
    Outputs:
        pi(s): Per-option intra-option policy [BxOxA for discrete, BxO (mu, log_std) for continuous]
        beta(s): Per-option termination probability [BxO]
        q(s): Per-option value [BxO]
        pi_omega(s): Softmax policy over options (parameterized by interest function if using) [BxO]
        q_entropy(s): Per-option value entropy (or 0 if not learning) [BxO]

    """

    def __init__(self,
                 input_size: int,
                 input_module_class: Callable,
                 rnn_module_class: Callable,
                 output_size: int,
                 option_size: int,
                 rnn_size: int,
                 intra_option_policy: str,
                 intra_option_kwargs: [dict, None] = None,
                 input_module_kwargs: [dict, None] = None,
                 use_interest: bool = False,
                 use_diversity: bool = False,
                 use_attention: bool = False,
                 baselines_init: bool = True,
                 NORM_EPS: float = 1e-6
                 ):
        super().__init__()
        if input_module_kwargs is None: input_module_kwargs = {}  # Assume module has all necessary arguments
        if intra_option_kwargs is None: intra_option_kwargs = {}
        input_module_kwargs = {**input_module_kwargs, **{'input_size': input_size}}  # Add input size
        intra_option_kwargs = {**intra_option_kwargs}
        self.use_interest = use_interest
        self.use_diversity = use_diversity
        self.use_attention = use_attention
        self.NORM_EPS = NORM_EPS
        pi_class = DiscreteIntraOptionPolicy if intra_option_policy == 'discrete' else ContinuousIntraOptionPolicy
        # Instantiate independent preprocessors for pi, pi_omega, q (and entropy), interest, and termination heads
        self.pi_proc, self.pi_omega_proc, self.q_proc, self.beta_proc = [input_module_class(**input_module_kwargs) for _ in range(4)]
        self.int_proc = input_module_class(**input_module_kwargs) if use_interest else Dummy(option_size)
        if baselines_init:
            self.pi_proc.apply(apply_init); self.pi_omega_proc.apply(apply_init)
            self.q_proc.apply(apply_init); self.int_proc.apply(apply_init); self.beta_proc.apply(apply_init)
        input_size = self.pi_proc.output_size
        self.pi_rnn, self.pi_omega_rnn, self.q_rnn, self.beta_rnn = [rnn_module_class(input_size + output_size + 1, rnn_size) for _ in range(4)]
        self.int_rnn = rnn_module_class(input_size + output_size + 1, rnn_size) if use_interest else None
        if baselines_init:
            lstm_init = partial(apply_init, gain=O_INIT_VALUES['lstm'])
            self.pi_rnn.apply(lstm_init); self.pi_omega_rnn.apply(lstm_init)
            self.q_rnn.apply(lstm_init); self.beta_rnn.apply(lstm_init)
            if use_interest: self.int_rnn.apply(lstm_init)
        self.pi = Sequential(nn.ReLU(), pi_class(rnn_size, option_size, output_size, ortho_init=baselines_init, **intra_option_kwargs))
        self.beta = Sequential(nn.ReLU(), Linear(rnn_size, option_size), nn.Sigmoid())
        self.q = Sequential(nn.ReLU(), Linear(rnn_size, option_size))
        self.q_ent = Sequential(nn.ReLU(), Linear(rnn_size, option_size)) if use_diversity else Dummy(option_size, out_value=0.)
        self.pi_omega = Sequential(nn.ReLU(), Linear(rnn_size, option_size), nn.Softmax(-1))
        self.interest = Sequential(nn.ReLU(), Linear(rnn_size, option_size), nn.Sigmoid()) if use_interest else Dummy(
            option_size)
        if baselines_init:
            init_v, init_pi = O_INIT_VALUES['v'], O_INIT_VALUES['pi']
            self.beta[1].apply(apply_init)
            self.pi_omega[1].apply(partial(apply_init, gain=init_pi))
            self.q[1].apply(partial(apply_init, gain=init_v))
            if use_interest: self.interest[0].apply(apply_init)
            if use_diversity: self.q_ent.apply(apply_init)

    def forward(self, observation, prev_action, prev_reward, T, B, init_rnn_states):
        # pi, beta, q, pi_omega, I hiddens
        # Features
        features = (pi_f, beta_f, q_f, pi_omega_f, I_f) = (self.pi_proc(observation), self.beta_proc(observation), self.q_proc(observation),
                                     self.pi_omega_proc(observation), self.int_proc(observation))
        # Rnns
        rnn_extra = torch.cat([
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        p_rnn_in, b_rnn_in, q_rnn_in, pio_rnn_in = [torch.cat([f.view(T, B, -1), rnn_extra], dim=2) for f in features[:-1]]
        p_s, b_s, q_s, pio_s, i_s = init_rnn_states
        (p_rnn_out, np_s), (b_rnn_out, nb_s), (q_rnn_out, nq_s), (pio_rnn_out, npio_s) = (self.pi_rnn(p_rnn_in, p_s), self.beta_rnn(b_rnn_in, b_s),
                                                        self.q_rnn(q_rnn_in, q_s), self.pi_omega_rnn(pio_rnn_in, pio_s))
        pi, beta, q, pi_omega = self.pi(p_rnn_out.view(T*B,-1)), self.beta(b_rnn_out.view(T*B,-1)), self.q(q_rnn_out.view(T*B,-1)), self.pi_omega(pio_rnn_out.view(T*B, -1))
        if self.use_interest:
            i_rnn_in = torch.cat([I_f.view(T, B, -1), rnn_extra], dim=2)
            i_rnn_out, ni_s = self.int_rnn(i_rnn_in, i_s)
            I = self.interest(i_rnn_out)
            pi_omega = pi_omega * I
            pi_omega.add_(self.NORM_EPS)  # Add eps to avoid instability
            pi_omega.div_(pi_omega.sum(-1, keepdim=True))  # Normalize so probabilities add to 1
        else:
            ni_s = npio_s  # REturn someone else's state, don't bother making one
        q_ent = self.q_ent(q_rnn_out.view(T*B, -1))  # Entropy uses q's processor. Will be appropriate size otherwise
        return pi, beta, q, pi_omega, q_ent, (np_s, nb_s, nq_s, npio_s, ni_s)

class TerminationCriticHead(Module):
    """

    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

"""
Termination Critic (ACTC):
    Predictability Objective: J(P^o) = H(X_f|o)  Minimize distribution over final states X_f given option o
        - Expressed as option: J(P^o) = -E_{x_s}[sum_{x_f}[P^o(x_f|x_s) * log P^o_{pi_omega} (x_f)]] Last term is option's starting distribution induced 
    
    A3C outputs a policy pi and Q(s,a)
    ACTC outputs pi_o and Q(s,o) for each option
    ACTC uses additional network 
        - Takes in 2 input states x_f and x_s (body_xs and body_xf)
        - Outputs:
            - P(x_f|x_s) for each option (probability of final state given start state and optio o ) Rainy Sigmoid
            - beta^o (x_f) (termination function) Rainy bernoulli
            - Marginal P_{pi_omega} (x_f) for each option (general probability of x_f under option o (sum of P(x_f|x_s) for all x_s)?) Rainy sigmoid 
            - B: Per-option baseline
        - Cnn as in A3C, FC1 as in a3c, specialized FC2 for each option
    
    ACTC pseudocode: For all x_i in trajectory
        - Update beta(x_i) via transition gradient: 
        
    RAINY:
        OptionCritic network (as above, but no termination)
        TerminationCritic network
    
"""