import torch
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class ScaleGrad(torch.autograd.Function):
    """Model component to scale gradients back from layer, without affecting
    the forward pass.  Used e.g. in dueling heads DQN models."""

    @staticmethod
    def forward(ctx, tensor, scale):
        """Stores the ``scale`` input to ``ctx`` for application in
        ``backward()``; simply returns the input ``tensor``."""
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Return the ``grad_output`` multiplied by ``ctx.scale``.  Also returns
        a ``None`` as placeholder corresponding to (non-existent) gradient of 
        the input ``scale`` of ``forward()``."""
        return grad_output * ctx.scale, None


# scale_grad = ScaleGrad.apply
# Supply a dummy for documentation to render.
scale_grad = getattr(ScaleGrad, "apply", None)


class EpislonGreedyLayer(torch.nn.Module):
    """Epsilon Greedy Layer. Assume last dim is action dim

    Args:
        epsilon (float): Probability of taking a random action
    Returns:
        a: Sampled action
        log_prob_a: Log probability of sampled option
        entropy: Entropy of distribution
        probs: All probabilities
    """

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def update_epsilon(self, epsilon):
        self.epsilon = epsilon

    def forward(self, x):
        baseprobs = torch.full_like(x, self.epsilon / x.size(-1))  # Base probability for any action
        max_a = x.argmax(-1)  # Best action for each
        max_a_prob = 1. - self.epsilon  # Probability of greedy action
        baseprobs[np.arange(x.size(0)), max_a] += max_a_prob  # Add to probability
        dist = torch.distributions.Categorical(probs=baseprobs)  # Use as probability input
        a = dist.sample()
        return dist.sample(), dist.log_prob(a), dist.entropy(), dist.probs


class View(torch.nn.Module):
    """View layer.

    Args:
        shape (tuple of ints): Shape of outputs
    Returns:
        x.view(-1, shape)  [i.e., flattened to shape]
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view((-1,) + self.shape)

class Dummy(torch.nn.Module):
    """ Dummy layer that outputs the same thing regardless of input

    Args:
        out_size (int): Output size
        out_value (float): Desired output value. Defaults to 1 (as in a mask)
    Returns:
        (B, out_size) tensor filled with out_value
    """
    def __init__(self, out_size, out_value=1.):
        super().__init__()
        self.out = torch.nn.Parameter(torch.full((out_size,), out_value), requires_grad=False)

    def forward(self, x):
        return self.out.expand(x.size(0), -1)


def update_state_dict(model, state_dict, tau=1, strip_ddp=True):
    """Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    if strip_ddp:
        state_dict = strip_ddp_state_dict(state_dict)
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {k: tau * state_dict[k] + (1 - tau) * v
                     for k, v in model.state_dict().items()}
        model.load_state_dict(update_sd)


def strip_ddp_state_dict(state_dict):
    """ Workaround the fact that DistributedDataParallel prepends 'module.' to
    every key, but the sampler models will not be wrapped in
    DistributedDataParallel. (Solution from PyTorch forums.)"""
    clean_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        key = k[7:] if k[:7] == "module." else k
        clean_state_dict[key] = v
    return clean_state_dict
