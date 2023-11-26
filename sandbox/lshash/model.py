import itertools

import torch
import torch.nn as nn

from sandbox.lshash.dataset import HASH_SIZE, NUM_HASHES


class LSHashNGP(nn.Module):

    def __init__(
        self,
        hash_features=2,
        hash_num_heads=NUM_HASHES,
        hash_maximum=(2 ** HASH_SIZE),
        mlp_features=64,
        mlp_hidden_layers=2,
        out_features=3,
    ):
        super().__init__()

        self.embeds = nn.ModuleList([
            nn.Embedding(hash_maximum, hash_features)
            for _ in range(hash_num_heads)
        ])

        in_features = hash_features * hash_num_heads
        dims = [in_features] + ([mlp_features] * (mlp_hidden_layers + 1)) + [out_features]

        mlp = []
        for d1, d2 in itertools.pairwise(dims):
            mlp.extend([nn.Linear(d1, d2, bias=True), nn.ReLU()])
        mlp = mlp[:-1]  # delete last RELU
        self.mlp = nn.Sequential(*mlp)

    def forward(self, input):
        embs = [h(input[..., i]) for i, h in enumerate(self.embeds)]
        return self.mlp(torch.cat(embs, dim=-1))
