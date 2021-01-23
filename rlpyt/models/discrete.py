import torch
import torch.nn.functional as F


class OneHotLayer(torch.nn.Module):

    def __init__(self,
                 input_classes: int
                 ):
        super().__init__()
        self.n = input_classes

    def forward(self, input):
        return F.one_hot(input, self.n).float()

    @property
    def output_size(self):
        return self.n
