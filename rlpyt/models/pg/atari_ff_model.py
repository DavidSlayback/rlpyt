
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel


class AtariFfModel(torch.nn.Module):
    """
    Feedforward model for Atari agents: a convolutional network feeding an
    MLP with outputs for action probabilities and state-value estimate.
    """

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [16, 32],
            kernel_sizes=kernel_sizes or [8, 4],
            strides=strides or [4, 2],
            paddings=paddings or [0, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )
        self.pi = torch.jit.script(torch.nn.Linear(self.conv.output_size, output_size))
        self.value = torch.jit.script(torch.nn.Linear(self.conv.output_size, 1))
        self.conv = torch.jit.script(self.conv)

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
        img = image.float()  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v

from rlpyt.models.oc import DiscreteIntraOptionPolicy
class AtariOcModel(torch.nn.Module):
    """
    Feedforward model for Atari agents: a convolutional network feeding an
    MLP with outputs for action probabilities and state-value estimate.
    """

    def __init__(
            self,
            image_shape,
            output_size,
            option_size,
            fc_sizes=512,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            baselines_init=True, # Orthogonal initialization of sqrt(2) until last layer, then 0.01 for policy, 1 for value
            use_interest=False,  # IOC sigmoid interest functions
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [16, 32],
            kernel_sizes=kernel_sizes or [8, 4],
            strides=strides or [4, 2],
            paddings=paddings or [0, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )
        self.use_interest = use_interest
        self.pi = DiscreteIntraOptionPolicy(self.conv.output_size, option_size, output_size, ortho_init=baselines_init)
        self.q = torch.nn.Linear(self.conv.output_size, option_size)
        self.beta = torch.nn.Sequential(torch.nn.Linear(self.conv.output_size, option_size), torch.nn.Sigmoid())
        self.pi_omega = torch.nn.Sequential(torch.nn.Linear(self.conv.output_size, option_size), torch.nn.Softmax(-1))
        self.pi_omega_I = torch.nn.Sequential(torch.nn.Linear(self.conv.output_size, option_size), torch.nn.Sigmoid()) if use_interest else torch.nn.Identity()

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
        pi_omega_I = self.pi_omega(fc_out)
        I = self.pi_omega_I(fc_out)
        if self.use_interest:
            pi_omega_I = pi_omega_I * I
            pi_omega_I.divide_(pi_omega_I.sum(-1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, q, beta, pi_omega_I = restore_leading_dims((pi, q, beta, pi_omega_I), lead_dim, T, B)

        return pi, q, beta, pi_omega_I

