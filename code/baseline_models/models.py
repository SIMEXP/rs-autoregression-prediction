import torch
import torch.nn as nn
from src.tools import string_to_list


class SimpleMLP(nn.Module):
    """Simple MLP for connectome for binary classification.
    Default parameter from AH's paper.
    """

    def __init__(self, n_in=2028, n_hidden="256,64"):
        super().__init__()

        n_hidden = string_to_list(n_hidden)
        layers = []
        for i, n in enumerate(n_hidden):
            if i == 0:
                layers.append(nn.Linear(n_in, n))
            else:
                layers.append(nn.Linear(n_hidden[i - 1], n))

            if i < len(n_hidden):
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout())
                layers.append(nn.BatchNorm1d(n))
        layers.append(nn.Linear(n_hidden[-1], 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        x = torch.tensor(
            x, dtype=torch.float32, device=next(self.parameters()).device
        )
        self.eval()
        return self.forward(x).cpu().detach().numpy()
