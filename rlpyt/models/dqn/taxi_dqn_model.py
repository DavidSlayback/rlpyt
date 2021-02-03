
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel


class TaxiDqnModel(torch.nn.Module):
    """Standard convolutional network for DQN.  2-D convolution for multiple
    video frames per observation, feeding an MLP for Q-value outputs for
    the action set.
    """

    def __init__(
            self,
            input_classes,
            output_size,
            embedding_size=10,
            hidden_sizes=None,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.embedding = torch.nn.Embedding(input_classes, 10)
        hidden_sizes = [50, 50, 50] if hidden_sizes is None else hidden_sizes
        self.head = MlpModel(embedding_size, hidden_sizes, output_size)

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute action Q-value estimates from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        image_shape[0], image_shape[1],...,image_shape[-1]], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Used in both sampler and in algorithm (both
        via the agent).
        """
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        q = self.head(conv_out.view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        return q
