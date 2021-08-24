import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LSTM(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Instantiating the hidden state
        self.hidden = self.init_hidden()

        # Instantiating the LSTM
        self.lstm = nn.LSTM(self.hparams.feature_dim, self.hparams.hidden_units, self.hparams.num_layers)
        self.actions = nn.Linear(self.hparams.hidden_units, self.hparams.action_space)

    def forward(self, x):
        self.lstm.flatten_parameters()

        if len(x.shape) == 2:
            x = x[None, ...]

        ## Hidden state
        actions, self.hidden = self.lstm(x, self.hidden)
        actions = self.actions(actions)
        actions = F.softmax(actions, dim=2)
        return actions[0]

    def init_hidden(self):
        hidden = torch.zeros(self.hparams.num_layers, 1, self.hparams.hidden_units)
        cell = torch.zeros(self.hparams.num_layers, 1, self.hparams.hidden_units)
        return hidden, cell
