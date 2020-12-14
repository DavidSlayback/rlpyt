
import torch
import torch.distributed as dist
from rlpyt.utils.tensor import infer_leading_dims
from typing import Tuple


class RunningMeanStdModel(torch.nn.Module):

    """Adapted from OpenAI baselines.  Maintains a running estimate of mean
    and variance of data along each dimension, accessible in the `mean` and
    `var` attributes.  Supports multi-GPU training by all-reducing statistics
    across GPUs."""

    def __init__(self, shape, min_clip=-10, max_clip=10, min_var=1e-6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.zeros(()))
        self.shape = shape
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.min_var = min_var

    # Convenience call to update and compute normalization in one pass
    def forward(self, x):
        self.update(x)
        obs_var = torch.clamp(self.var, min=self.min_var)
        return torch.clamp((x - self.mean) / obs_var.sqrt(), min=self.min_clip, max=self.max_clip)

    def update(self, x):
        _, T, B, _ = infer_leading_dims(x, len(self.shape))
        x = x.view(T * B, *self.shape)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = T * B
        if dist.is_initialized():  # Assume need all-reduce.
            mean_var = torch.stack([batch_mean, batch_var])
            dist.all_reduce(mean_var)
            world_size = dist.get_world_size()
            mean_var /= world_size
            batch_count *= world_size
            batch_mean, batch_var = mean_var[0], mean_var[1]
        if self.count == 0:
            self.mean[:] = batch_mean
            self.var[:] = batch_var
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean[:] = self.mean + delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
            self.var[:] = M2 / total
        self.count += batch_count

    # Reset normalization for task switch
    def reset(self):
        self.mean = 0.
        self.var = 1.
        self.count = 0

class RunningReward(torch.nn.Module):

    """Adapted from OpenAI baselines.  Maintains a running estimate of mean
    and variance of data along each dimension, accessible in the `mean` and
    `var` attributes.  Supports multi-GPU training by all-reducing statistics
    across GPUs."""

    def __init__(self, shape=None, min_clip=-10, max_clip=10, min_var=1e-6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.zeros(()))
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.min_var = min_var

    # Convenience call to update and compute normalization in one pass
    def forward(self, x, dones=None):
        self.update(x)
        obs_var = torch.clamp(self.var, min=self.min_var)
        return torch.clamp((x - self.mean) / obs_var.sqrt(), min=self.min_clip, max=self.max_clip)

    def update(self, x, dones=None):
        batch_var, batch_mean = torch.var_mean(x, unbiased=False)
        batch_count = x.numel()
        if dist.is_initialized():  # Assume need all-reduce.
            mean_var = torch.stack([batch_mean, batch_var])
            dist.all_reduce(mean_var)
            world_size = dist.get_world_size()
            mean_var /= world_size
            batch_count *= world_size
            batch_mean, batch_var = mean_var[0], mean_var[1]
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
            self.var = M2 / total
        self.count += batch_count

    # Reset normalization for task switch
    def reset(self):
        self.mean = 0.
        self.var = 1.
        self.count = 0

"""
Assume rewards, done in shape (batch_T, batch_B), returns in shape (batch_B)
Calculate return for each timestep
"""
@torch.jit.script
def _calc_ret(original_returns: torch.Tensor,
              rewards: torch.Tensor,
              dones: torch.Tensor,
              gamma: float
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    new_returns = torch.empty_like(rewards)
    new_returns[0] = original_returns * gamma + rewards[0]  # First step
    for i in torch.arange(1, rewards.size(0)):
        new_returns[i] = new_returns[i-1] * (1-dones[i-1]) * gamma + rewards[i]  # ret * discount + rew
    final_returns = new_returns[-1] * (1-dones[-1])
    return new_returns, final_returns

class RunningReturn(torch.nn.Module):
    """
    "Incorrect" reward normalization [copied from OAI code]
    Incorrect in the sense that we
    1. update return
    2. divide reward by std(return) *without* subtracting and adding back mean

    Shape is batch_B (maintaining return for each environment so we can individually reset)
    """
    def __init__(self, shape, gamma, min_clip=-10., max_clip=10., min_var=1e-6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("ret", torch.zeros(shape))
        self.gamma = gamma
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.min_var = min_var

    # Convenience call to update and compute normalization in one pass
    def forward(self, x, dones):
        self.update(x, dones)
        obs_var = torch.clamp(self.var, min=self.min_var)
        return torch.clamp(x / obs_var.sqrt(), min=self.min_clip, max=self.max_clip)

    def update(self, x, dones):
        all_ret, self.ret = _calc_ret(self.ret, x, dones, self.gamma)
        batch_var, batch_mean = torch.var_mean(all_ret, unbiased=False)
        batch_count = x.numel()
        if dist.is_initialized():  # Assume need all-reduce.
            mean_var = torch.stack([batch_mean, batch_var])
            dist.all_reduce(mean_var)
            world_size = dist.get_world_size()
            mean_var /= world_size
            batch_count *= world_size
            batch_mean, batch_var = mean_var[0], mean_var[1]
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
            self.var = M2 / total
        self.count += batch_count

    # Reset normalization for task switch
    def reset(self):
        self.ret = 0.
        self.mean = 0.
        self.var = 1.
        self.count = 0
