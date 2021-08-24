import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SelfAttention(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.tokeys = nn.Linear(self.hparams.data_dim, self.hparams.query_dim)
        self.toqueries = nn.Linear(self.hparams.data_dim, self.hparams.query_dim)

    def forward(self, x):
        b, t, m = x.size()  # batch dimension, sequence length, input vector dimension

        # we obtain keys, queries
        keys = self.tokeys(x)
        queries = self.toqueries(x)

        # The dot product to obtain the weights should collapse the m dimension
        w_prime = torch.einsum('btm,bfm->btf', queries, keys) / math.sqrt(m)
        return F.softmax(w_prime, dim=-1).squeeze()
