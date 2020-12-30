
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.params import CONVNET_IMPALA_LARGE  # Used by IOC
from rlpyt.models.utils import layer_init, conv2d_output_shape, View


def conv_out_size(self, h, w, c=None):
    """Helper function ot return the output size for a given input shape,
    without actually performing a forward pass through the model."""
    for child in self.conv.children():
        try:
            h, w = conv2d_output_shape(h, w, child.kernel_size,
                                       child.stride, child.padding)
        except AttributeError:
            pass  # Not a conv or maxpool layer.
        try:
            c = child.out_channels
        except AttributeError:
            pass  # Not a conv layer.
    return h * w * c

# Padding is 'same'. k = (kernel_size-1 / 2)
# So for 3x3, k = 1
class IMPALAResidualBlock(torch.nn.Module):
    """
    Residual block from large IMPALA module. 2 conv2d layers, keep size same, k=3, stride=1
    """
    def __init__(self, in_channel):
        super().__init__()
        out_channel = in_channel
        self.conv = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x)

class IMPALABaseBlock(torch.nn.Module):
    """
    Conv2d+MaxPool(stride=2)+2xresidual block
    """
    def __init__(self, h, w, in_channels=3, out_channels=16, kernel=3, stride=1, maxstride=2):
        super().__init__()
        p = int((kernel - 1) // 2) # Same padding
        base_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=p),
            torch.nn.MaxPool2d(kernel, maxstride, p)
        )
        self.output_size = conv2d_output_shape(h, w, kernel_size=kernel, stride=maxstride, padding=p)  # Output after maxpool
        self.conv = torch.nn.Sequential(
            base_conv,
            IMPALAResidualBlock(out_channels),
            IMPALAResidualBlock(out_channels)
        )
    def forward(self, x):
        return self.conv(x)

class IMPALAConvModel(torch.nn.Module):
    def __init__(self,
                 image_shape,
                 channels=CONVNET_IMPALA_LARGE['channels'],
                 fc_sizes=256
                 ):
        super().__init__()
        c, h, w = image_shape
        blocks = []
        for oc in channels:
            blocks.append(IMPALABaseBlock(h, w, c, oc))
            h, w = blocks[-1].output_size  # Output size for next block
            c = oc  # In channel becomes last out channel
        out_size = c * h * w  # Size of flattened output
        blocks.extend([
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(out_size, fc_sizes),
            torch.nn.ReLU()
        ])
        self.conv = torch.nn.Sequential(*blocks)
        self.output_size = fc_sizes

    def forward(self, input):
        return self.conv(input)

class ProcgenFfModel(torch.nn.Module):
    """
    Feedforward model for MiniWorld agents: a convolutional network feeding an
    MLP with outputs for action probabilities and state-value estimate.
    """

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=256,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            baselines_init=True,
            init_v=1.,
            init_pi=.01
            ):
        """Instantiate neural net module according to inputs."""
        super().__init__()
        self.conv = IMPALAConvModel(image_shape, fc_sizes=fc_sizes)
        self.pi = torch.jit.script(layer_init(torch.nn.Linear(self.conv.output_size, output_size), init_pi))
        self.value = torch.jit.script(layer_init(torch.nn.Linear(self.conv.output_size, 1), init_v))
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
        img = image.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v
