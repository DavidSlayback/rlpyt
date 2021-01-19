import numpy as np
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.oc import DiscreteIntraOptionPolicy
from rlpyt.models.params import CONVNET_DQN  # Used by IOC
from rlpyt.models.utils import Dummy

class MiniworldOcFfModel(torch.nn.Module):
    """
    Feedforward model for Miniworld agents: a convolutional network feeding an
    MLP with outputs for action probabilities and state-value estimate.
    """

    def __init__(
            self,
            image_shape,
            output_size,
            num_options,
            fc_sizes=512,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            use_interest=False
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        if channels is not None:
            self.conv = Conv2dHeadModel(
                image_shape=image_shape,
                channels=channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                use_maxpool=use_maxpool,
                hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
            )
        else:
            self.conv = Conv2dHeadModel(image_shape=image_shape, **CONVNET_DQN)
        self.pi = DiscreteIntraOptionPolicy(self.conv.output_size, num_options, output_size, True)
        self.q = torch.nn.Linear(self.conv.output_size, num_options)
        self.beta = torch.nn.Sequential(torch.nn.Linear(self.conv.output_size, num_options), torch.nn.Sigmoid())
        self.pi_omega = torch.nn.Sequential(torch.nn.Linear(self.conv.output_size, num_options), torch.nn.Softmax(dim=-1))
        self.I = torch.nn.Sequential(torch.nn.Linear(self.conv.output_size, num_options), torch.nn.Sigmoid()) if use_interest else Dummy(num_options)

    def forward(self, image, prev_action, prev_reward):
        """
        Compute action probabilities and value estimate from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        *image_shape], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Used in both sampler and in algorithm (both
        via the agent).
        """
        img = image.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = self.pi(fc_out)
        q = self.q(fc_out)
        beta = self.beta(fc_out)
        pi_o = self.pi_omega(fc_out)
        I = self.I(fc_out)

        # Multiply pi_o by interest, normalization occurs in multinomial
        pi_I = pi_o * I

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, q, beta, pi_I = restore_leading_dims((pi, q, beta, pi_I), lead_dim, T, B)

        return pi, q, beta, pi_I