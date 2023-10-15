import itertools

import scipy
import torch
import torch.nn as nn


# Reference: https://codeplea.com/triangular-interpolation
def barycentric_interpolation(vertices, values, queries):
    v1, v2, v3 = vertices.unbind(-1)  # (B D 3) -> (B D)
    v1x, v1y = torch.split(v1, 1, dim=-1)  # (B D) -> 2 x (B 1)
    v2x, v2y = torch.split(v2, 1, dim=-1)
    v3x, v3y = torch.split(v3, 1, dim=-1)
    x, y = torch.split(queries, 1, dim=-1)

    denom = (v2y - v3y) * (v1x - v3x) + (v3x - v2x) * (v1y - v3y)
    w1 = ((v2y - v3y) * (x - v3x) + (v3x - v2x) * (y - v3y)) / denom
    w2 = ((v3y - v1y) * (x - v3x) + (v1x - v3x) * (y - v3y)) / denom
    w3 = 1 - w1 - w2

    o1, o2, o3 = values.unbind(-1)  # (B D 3) -> (B D)
    return (w1 * o1) + (w2 * o2) + (w3 * o3)


class DelaunayHashEmbedder(nn.Module):

    def __init__(self, features=2, num_points=100):
        super().__init__()

        self.anchors = nn.Parameter(torch.randn(num_points, 2))
        self.embs = nn.Parameter(torch.randn(num_points + 4, features))

    def forward(self, input):
        anchors = torch.tanh(self.anchors)
        corners = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]]).to(input)
        anchors = torch.cat([anchors, corners], dim=0)

        # Traingulation
        tri = scipy.spatial.Delaunay(anchors.detach().numpy())
        simplices = tri.simplices[tri.find_simplex(input.detach().numpy())]
        simplices = torch.tensor(simplices, dtype=torch.int, device=input.device)
        assert torch.all(simplices >= 0)

        # Barycentric interpolation
        vertices = anchors[simplices].mT  # (B D 3), D = 2
        values = self.embs[simplices].mT  # (B D 3)
        return barycentric_interpolation(vertices, values, queries=input)


class DelaunayNGP(nn.Module):

    def __init__(
        self,
        hash_features=2,
        hash_num_points=1000,
        hash_num_heads=3,
        mlp_features=32,
        mlp_hidden_layers=2,
        out_features=3,
    ):
        super().__init__()

        # Multiple independent copies of hash
        self.hashers = nn.ModuleList([
            DelaunayHashEmbedder(
                features=hash_features,
                num_points=hash_num_points,
            )
            for _ in range(hash_num_heads)
        ])

        in_features = hash_features * hash_num_heads
        dims = [in_features] + ([mlp_features] * (mlp_hidden_layers + 1)) + [out_features]

        mlp = []
        for d1, d2 in itertools.pairwise(dims):
            mlp.extend([nn.Linear(d1, d2, bias=True), nn.ReLU()])
        mlp = mlp[:-1]  # delete last GELU
        self.mlp = nn.Sequential(*mlp)

    def forward(self, input):
        input = torch.cat([h(input) for h in self.hashers], dim=-1)
        return self.mlp(input)
