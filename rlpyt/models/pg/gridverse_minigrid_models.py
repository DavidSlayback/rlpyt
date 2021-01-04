import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.envs.gym import GymSpaceWrapper as RLGS
from rlpyt.envs.gym_schema import GymSpaceWrapper as RLGSS

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.utils.collections import namedarraytuple
import gym_minigrid
import gym_gridverse

RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work


GRIDVERSE_OBSERVATION_KEYS = ['grid', 'agent']  # Keys into a gridverse observation (grid view and agent)
MINIGRID_OBSERVATION_KEYS = ['image', 'direction', 'mission']  # Keys into a minigrid observation (grid view, agent direction, mission string)
MINIGRID_DIMS = {'object':11, 'color':6, 'state':3}
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
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size), nonlinearity(),
                                   nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.model(x)


class GridEmbedding(nn.Module):
    """
    Converts minigrid input through an embedding. Input is BxHxWxC
    """

    def __init__(self, embedding_dim=8):
        super().__init__()
        self.model = nn.Embedding(33, embedding_dim)

    def forward(self, x):
        b, h, w, c = x.size()
        offset = torch.LongTensor([0, 11, 22]).view(1, 1, 1, 3).to(x.device)
        x = x + offset
        x = self.model(x.long()).view(b, h, w, self.model.embedding_dim*3).permute(0, 3, 1, 2)
        return x


class MiniGridConv(nn.Module):
    """
    BabyAI convolutional layers
    """

    def __init__(self, embedding_dim=8, view_size=7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 3, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(view_size, view_size), stride=2)
        )

    def forward(self, x):
        return self.model(x)


class BabyAILSTMModel(nn.Module):
    """
    Order
    Image -> ImageBOW -> Conv2d(128, 128, 3x3, 1, 1) -> BatchNorm2d(128) -> ReLU -> Conv2d(128,128,3,1,1) -> BatchNorm2d(128) -> ReLU -> MaxPool2d(7x7, stride=2)
    -> LSTMCell(in, 128) -> Actor/Critic (64 Linear, Tanh, 64 -> a/v)
    """

    def __init__(self, image_shape, output_size, embedding_dim=8, memory_dim=128):
        super().__init__()
        view_size = image_shape[0]
        self.pre_mempry_model = nn.Sequential(
            GridEmbedding(embedding_dim),
            MiniGridConv(embedding_dim, view_size)
        )
        self.lstm = nn.LSTM(128, memory_dim)
        self.actor = BaseMLP(memory_dim, output_size)
        self.critic = BaseMLP(memory_dim, 1)

    def forward(self, image, prev_action, prev_reward, init_rnn_state):
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        fc_out = self.pre_mempry_model(image.view(T * B, *img_shape))
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(fc_out.view(T, B, -1))
        pi = F.softmax(self.actor(lstm_out.view(T * B, -1)), dim=-1)
        v = self.critic(lstm_out.view(T * B, -1)).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)
        return pi, v, next_rnn_state


class ImageBOWEmbedding(nn.Module):
    """
    Image Bag-of-words embedding from BabyAI 1.1

    Parameters:
        max_value: Maximum value in any of the channels of grid
        embedding_dim: Dimension of embedding space (size of output)
    """

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

class AllenActFlatEmbedding_MiniGrid(nn.Module):
    """
    Flattened embedding for MiniGrid grid observation. Expects BxHxWxC. Returns BxHxWxEmb*3

    Parameters:
        embedding_dim: embedding dimension. Defaults to 8 as in AllenAct
    """
    def __init__(self, embedding_dim=8):
        super().__init__()
        self.o_emb = nn.Embedding(MINIGRID_DIMS['object'], embedding_dim)
        self.c_emb = nn.Embedding(MINIGRID_DIMS['color'], embedding_dim)
        self.s_emb = nn.Embedding(MINIGRID_DIMS['state'], embedding_dim)
        self.output_size = embedding_dim * 3

    def forward(self, x):
        x = x.long()  # Ensure it's usable integers
        o, c, s = self.o_emb(x[:, :, :, 0]), self.c_emb(x[:, :, :, 1]), self.s_emb(x[:, :, :, 2])
        return torch.cat((o, c, s), dim=-1)

class AllenActMiniGridModel(nn.Module):
    """
    AllenAct MiniGrid model with no convolution, using flattened embeddings for each channel

    Parameters:
        embedding_dim: Embedding dimension. Defaults to 8 as in AllenAct

    """
    def __init__(self,
                 image_shape,
                 output_size,
                 embedding_dim=8,
                 hidden_size=512):
        super().__init__()
        h, w, c = image_shape
        self.pre_memory_model = AllenActFlatEmbedding_MiniGrid(embedding_dim)
        out = self.pre_memory_model.output_size * h * w  # Input size to next layer
        self.rnn = nn.LSTM(out, hidden_size)  # RNN (LSTM)
        self.pi = nn.Linear(hidden_size, output_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, image, prev_action, prev_reward, init_rnn_state):
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(image, 3)
        fc_out = self.pre_memory_model(image.view(T * B, *img_shape))
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.rnn(fc_out.view(T, B, -1))
        pi = F.softmax(self.pi(lstm_out.view(T * B, -1)), dim=-1)
        v = self.v(lstm_out.view(T * B, -1)).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)
        return pi, v, next_rnn_state


class AllenActFlatEmbedding_GridVerse(nn.Module):
    """
    Flattened embedding for GridVerse observation.

    Parameters:
        grid_space: grid Box() (e.g., (7,7,6))
        agent_space: agent Box() (e.g., (3,)
    """
    def __init__(self, grid_space, agent_space):
        super().__init__()
        grid_dims = grid_space.high[0,0,:]


    def forward(self, x):
        x = x.long()
        pass


if __name__ == "__main__":
    import gym


    e = gym.make('MiniGrid-FourRooms-v0')
    img = e.reset()["image"]
    img_t = torch.tensor(img).unsqueeze(0)
    img2 = e.step(0)[0]["image"]
    img2_t = torch.tensor(img2).unsqueeze(0)
    test_B_tensor = torch.cat((img_t, img2_t), dim=0)
    emba = GridEmbedding()
    test_out = emba(test_B_tensor)
