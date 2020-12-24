import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.params import CONVNET_MINIGRID_TINY
from rlpyt.models.running_mean_std import ObsScaler

RnnState = namedarraytuple("RnnState", ["h"])  # For downstream namedarraytuples to work

class MinigridGRUModel(nn.Module):
    """
    Recurrent model for Minigrid agents: a convolutional network feeding a GRU, then
    MLP with outputs for action probabilities and state-value estimate.
    """
    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=128,
            lstm_size=128,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            scale_obs=True,  # Whether to scale observations
            obs_mean=4.,  # Mean to subtract
            obs_scale=8.  # Scale to apply
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        if channels is not None:  # Override defaults
            self.conv = Conv2dHeadModel(
                image_shape=image_shape,
                channels=channels or [16, 32],
                kernel_sizes=kernel_sizes or [8, 4],
                strides=strides or [4, 2],
                paddings=paddings or [0, 1],
                use_maxpool=use_maxpool,
                hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
            )
        else:
            self.conv = Conv2dHeadModel(image_shape=image_shape, hidden_sizes=fc_sizes, **CONVNET_MINIGRID_TINY)

        self.lstm = nn.GRU(self.conv.output_size, lstm_size)
        self.pi = torch.nn.Linear(lstm_size, output_size)
        self.value = torch.nn.Linear(lstm_size, 1)
        self.scaler = nn.Identity() if not scale_obs else ObsScaler(obs_mean, obs_scale)
        self.conv = nn.Sequential(self.scaler, self.conv)

    def forward(self, grid, prev_action, prev_reward, init_rnn_state):
        """
        Compute action probabilities and value estimate from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        *image_shape], with T=1,B=1 when not given.  Recurrent layers processed as [T,B,H]
        Expects uint8 grids, applies observation scaling if using.
        Used in both sampler and in algorithm (both via the agent). Also returns the
        next RNN state.
        """

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(grid, 3)
        fc_out = self.conv(grid.view(T * B, *img_shape))
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, hn = self.lstm(fc_out.view(T, B, -1), init_rnn_state)
        pi = F.softmax(self.pi(lstm_out.view(T * B, -1)), dim=-1)
        v = self.value(lstm_out.view(T *B , -1)).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        next_rnn_state = RnnState(h=hn)
        return pi, v, next_rnn_state