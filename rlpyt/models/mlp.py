
import torch
from rlpyt.models.utils import layer_init, O_INIT_VALUES, apply_init
from functools import partial
import numpy as np


class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
        inits (list/tuple): List/tuple of 2 init values. First is for main layers, second is for last layer
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list or None for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            inits=None
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        if inits is None:
            hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
                zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        else:
            hidden_layers = [layer_init(torch.nn.Linear(n_in, n_out), inits[0]) for n_in, n_out in
                zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            if inits is None:
                sequence.append(torch.nn.Linear(last_size, output_size))
            else:
                sequence.append(layer_init(torch.nn.Linear(last_size, output_size), inits[1]))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size
