import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel

def initialize_parameters(m):
    """
    Convenient orthogonal initialization for a full module
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class BaseMLP(nn.Module):
    """
    Basic MLP used for actor and critic in BabyAI/AllenAct model (after state processing)
    """
    def __init__(self, input_size, output_size, hidden_size=64, nonlinearity=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size), nonlinearity(), nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.model(x)

class GridEmbedding(nn.Module):
    """
    Converts minigrid input through an embedding. Input is HxWxC
    """
    def __init__(self, embedding_dim=8):
        super().__init__()
        self.model = nn.Embedding(33, embedding_dim)
        self.offset = torch.LongTensor([0, 11, 22])

    def forward(self, x):
        h, w, c = x.shape[-3:]
        offset = torch.LongTensor([0, 11, 22]).view(1, 1, 1, 3).to(x.device)
        x = x + offset
        x = self.model(x.float()).view(*x.size()[:-1], self.model.embedding_dim).permute(0, 3, 1, 2)
class ImageBOWEmbedding(nn.Module):
   def __init__(self, max_value, embedding_dim=128):
       super().__init__()
       self.max_value = max_value
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(3 * max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()  # Assumes transposed image
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)

# Order
# Image -> ImageBOW -> Conv2d(128, 128, 3x3, 1, 1) -> BatchNorm2d(128) -> ReLU -> Conv2d(128,128,3,1,1) -> BatchNorm2d(128) -> ReLU -> MaxPool2d(7x7, stride=2)
# -> LSTMCell(in, 128) -> Actor/Critic (64 Linear, Tanh, 64 -> a/v)