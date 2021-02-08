
import torch
from typing import List


#@torch.no_grad()
#@torch.jit.script
def build_packed_sequence_info(dones: torch.Tensor):
    """Build info necessary to make packed sequence

    Args:
        dones: T x B
    """
    is_new_episodes = dones.roll(1,0)
    is_new_episodes[0] = True  # First timestep is start of rollout
    sequence_starts_flat = torch.nonzero(is_new_episodes.T.flatten()).flatten()  # Flatten to T*B. Must transpose first
    lengths = sequence_starts_flat.roll(-1) - sequence_starts_flat
    lengths[-1] = dones.numel() - sequence_starts_flat[-1]  # Last length
    #sorted_lengths = torch.sort(lengths, descending=True)
    #max_length = sorted_lengths.max()
    return sequence_starts_flat, lengths, is_new_episodes

#@torch.no_grad()
#@torch.jit.script
def build_padded_sequence(o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, rnn_state: torch.Tensor, dones: torch.Tensor):
    # a must be one-hot
    T, B, N, H = rnn_state.size()
    flat_ind, lengths, is_new_episodes = build_packed_sequence_info(dones)
    length_list: List[int] = lengths.tolist()
    # Flatten to indices correctly
    o = o.T.reshape(T*B, *o.shape[2:])  # Have to call reshape here
    a = a.view(T*B, -1)
    r = r.view(T*B, -1)
    # Create padded sequences
    padded_o = torch.nn.utils.rnn.pad_sequence(o.split(length_list))
    padded_a = torch.nn.utils.rnn.pad_sequence(a.split(length_list))
    padded_r = torch.nn.utils.rnn.pad_sequence(r.split(length_list))
    # Create corresponding rnn
    init_rnns = rnn_state[is_new_episodes]
    return padded_o, padded_a, padded_r, init_rnns







def select_at_indexes(indexes, tensor):
    """Returns the contents of ``tensor`` at the multi-dimensional integer
    array ``indexes``. Leading dimensions of ``tensor`` must match the
    dimensions of ``indexes``.
    """
    dim = len(indexes.shape)
    assert indexes.shape == tensor.shape[:dim]
    num = indexes.numel()
    t_flat = tensor.view((num,) + tensor.shape[dim:])
    s_flat = t_flat[torch.arange(num), indexes.view(-1)]
    return s_flat.view(tensor.shape[:dim] + tensor.shape[dim + 1:])


def to_onehot(indexes, num, dtype=None):
    """Converts integer values in multi-dimensional tensor ``indexes``
    to one-hot values of size ``num``; expanded in an additional
    trailing dimension."""
    if dtype is None:
        dtype = indexes.dtype
    onehot = torch.zeros(indexes.shape + (num,),
        dtype=dtype, device=indexes.device)
    onehot.scatter_(-1, indexes.unsqueeze(-1).type(torch.long), 1)
    return onehot


def from_onehot(onehot, dim=-1, dtype=None):
    """Argmax over trailing dimension of tensor ``onehot``. Optional return
    dtype specification."""
    indexes = torch.argmax(onehot, dim=dim)
    if dtype is not None:
        indexes = indexes.type(dtype)
    return indexes


def valid_mean(tensor, valid=None, dim=None):
    """Mean of ``tensor``, accounting for optional mask ``valid``,
    optionally along a dimension."""
    dim = () if dim is None else dim
    if valid is None:
        return tensor.mean(dim=dim)
    valid = valid.type(tensor.dtype)  # Convert as needed.
    return (tensor * valid).sum(dim=dim) / valid.sum(dim=dim)


def infer_leading_dims(tensor, dim: int):
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should 
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape


def restore_leading_dims(tensors, lead_dim, T=1, B=1):
    """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
    leading dimensions, which will become [], [B], or [T,B].  Assumes input
    tensors already have a leading Batch dimension, which might need to be
    removed. (Typically the last layer of model will compute with leading
    batch dimension.)  For use in model ``forward()`` method, so that output
    dimensions match input dimensions, and the same model can be used for any
    such case.  Use with outputs from ``infer_leading_dims()``."""
    is_seq = isinstance(tensors, (tuple, list))
    tensors = tensors if is_seq else (tensors,)
    if lead_dim == 2:  # (Put T dim.)
        tensors = tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
    if lead_dim == 0:  # (Remove B=1 dim.)
        assert B == 1
        tensors = tuple(t.squeeze(0) for t in tensors)
    return tensors if is_seq else tensors[0]


def joint_entropy(x1, x2):
    joint = x1 * x2
    return -torch.sum(joint * torch.log(joint), dim=-1)  # Sum over action dimension

# ADapted from https://github.com/mi92/batchdist
class BatchDistance(torch.nn.Module):
    """
    This module allows for the computation of a given
    pair-wise operation (such as a distance) on the batch level.
    Notably, we assume that the operation already allows for a
    batch dimension, however only computes the operation pair-wise.
    """

    def __init__(self, op=joint_entropy, device='cpu', dtype=torch.float64, result_index=0, pair_index=2):
        """
        - op: callable operation which takes two data batches x1,x2
            (both tensor of same dimensionality with the batch dimension
            in the first dimension), and returns a tensor of shape [batch_dim]
            with the pair-wise result of the operation.
            Example: for the operation f, op would return
                [f(x1_1, x2_1), f(x1_2, x2_2), ..., f(x1_n, x2_n)] where n
            refers to the batch dimension.
            In case, the op returns a tuple of results, result_index specifies
            which element refers to the result to use (by default 0, i.e. the first
            element)
        """
        super(BatchDistance, self).__init__()
        self.op = op
        self.device = device
        self.dtype = dtype
        self.result_index = result_index
        self.pair_index = pair_index

    def forward(self, x1, x2=None, **params):
        """
        computes batched distance operation for two batches of data x1 and x2
        (first dimension refering to the batch dimension) and returns the
        matrix of distances.
        """
        # get batch dimension of both data batches
        x2 = x1 if x2 is None else x2
        d1, d2 = x1.size(), x2.size()
        n1, n2 = d1[self.pair_index], d2[self.pair_index]

        # if operation is computed on one dataset, we can skip redundant index pairs
        if x1.size() == x2.size() and torch.equal(x1, x2):
            inds = torch.triu_indices(n1, n2)
            triu = True  # use only upper triangular
        else:
            inds = self._get_index_pairs(n1, n2)  # get index pairs without looping
            triu = False
        # expand data such that pair-wise operation covers all required pairs of
        # instances
        x1_batch = x1[..., inds[0], :]
        x2_batch = x2[..., inds[1], :]

        result = self.op(x1_batch, x2_batch)

        # check if op returns tuple of results and use the result_index'th element:
        if type(result) == tuple:
            result = result[self.result_index]

        # convert flat output to result matrix (e.g. a distance matrix)
        D = torch.zeros(n1, n2, dtype=self.dtype, device=self.device)
        D[..., inds[0], inds[1]] = result.to(dtype=self.dtype)
        if triu:  # mirror upper triangular such that full distance matrix is recovered
            D = self._triu_to_full(D)
        return D

    def _get_index_pairs(self, n1, n2):
        """
        return all pairs of indices of two 1d index tensors
        """
        x1 = torch.arange(n1)
        x2 = torch.arange(n2)

        x1_ = x1.repeat(x2.shape[0])
        x2_ = x2.repeat_interleave(x1.shape[0])
        return torch.stack([x1_, x2_])

    def _triu_to_full(self, D):
        """
        Convert triu (upper triangular) matrix to full, symmetric matrix.
        Assumes square input matrix D
        """
        diagonal = torch.eye(D.shape[0],
                             dtype=torch.float64,
                             device=self.device)
        diagonal = diagonal * torch.diag(D)  # eye matrix with diagonal entries of D
        D = D + D.T - diagonal  # add transpose minus diagonal values to convert
        # upper triangular to full matrix
        return D

EPS=1e-20
def batch_pairwise_joint_entropy_mean(x1: torch.Tensor, pair_dim: int = 2, eps: float = EPS):
    x1 = torch.clamp(x1, min=eps, max=1.0)
    x2 = x1
    d1, d2 = x1.size(), x2.size()
    n1, n2 = d1[pair_dim], d2[pair_dim]
    inds = torch.triu_indices(n1, n2, offset=1)  # Offset is to avoid pairing rows with themselves
    x1_batch = x1[..., inds[0], :]
    x2_batch = x2[..., inds[1], :]
    joint = x1_batch * x2_batch
    joint = -torch.sum(joint * torch.log(joint), dim=-1)  # Sum over action dimension
    return joint.mean(pair_dim)  # Mean over pairwise dimension (gives mean joint entropy for all combinations)

@torch.jit.script
def batch_pairwise_joint_entropy_mean_T_B(x1: torch.Tensor, pair_dim: int = 2, eps: float = EPS):
    x1 = torch.clamp(x1, min=eps, max=1.0)
    x2 = x1
    d1, d2 = x1.size(), x2.size()
    n1, n2 = d1[pair_dim], d2[pair_dim]
    inds = torch.triu_indices(n1, n2, offset=1)
    x1_batch = x1[:,:, inds[0], :]
    x2_batch = x2[:,:, inds[1], :]
    joint = x1_batch * x2_batch
    joint = -torch.sum(joint * torch.log(joint), dim=-1)  # Sum over action dimension
    return joint.mean(pair_dim)  # Mean over pairwise dimension (gives mean joint entropy for all combinations)
def batch_pairwise_cross_entropy_mean(x1: torch.Tensor, pair_dim: int = 2, eps: float = EPS):
    x1 = torch.clamp(x1, min=eps, max=1.0)
    x2 = x1
    d1, d2 = x1.size(), x2.size()
    n1, n2 = d1[pair_dim], d2[pair_dim]
    inds = torch.triu_indices(n1, n2, offset=1)
    x1_batch = x1[..., inds[0], :]
    x2_batch = x2[..., inds[1], :]
    joint = -torch.sum(x1_batch * torch.log(x2_batch), dim=-1)  # Sum over action dimension
    return joint.mean(pair_dim)  # Mean over pairwise dimension (gives mean joint entropy for all combinations)
@torch.jit.script
def batch_pairwise_cross_entropy_mean_T_B(x1: torch.Tensor, pair_dim: int = 2, eps: float = EPS):
    x1 = torch.clamp(x1, min=eps, max=1.0)
    x2 = x1
    d1, d2 = x1.size(), x2.size()
    n1, n2 = d1[pair_dim], d2[pair_dim]
    inds = torch.triu_indices(n1, n2, offset=1)
    x1_batch = x1[:,:, inds[0], :]
    x2_batch = x2[:,:, inds[1], :]
    joint = -torch.sum(x1_batch * torch.log(x2_batch), dim=-1)  # Sum over action dimension
    return joint.mean(pair_dim)  # Mean over pairwise dimension (gives mean joint entropy for all combinations)