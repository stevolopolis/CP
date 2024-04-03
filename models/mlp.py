import torch
import numpy as np
import torch.nn as nn


class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies=13):
        super().__init__()

        self.in_features = in_features
        self.num_frequencies = num_frequencies

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def forward(self, coords):
        #coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc       #.reshape(1, coords.shape[0], self.out_dim)


class ReLULayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_bias=True
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.use_bias = use_bias
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.activation(self.linear(x))

class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        mlp_configs
    ):
        
        super().__init__()
        self.mlp_configs = mlp_configs.NET
        num_layers = self.mlp_configs.num_layers
        dim_hidden = self.mlp_configs.dim_hidden
        use_bias = self.mlp_configs.use_bias

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.use_bias = use_bias

        layers = []
        for i in range(num_layers - 1):
            if i==0:
                layers.append(ReLULayer(dim_in, dim_hidden, use_bias=use_bias))
            else:
                layers.append(ReLULayer(dim_hidden, dim_hidden, use_bias=use_bias))

        self.net = nn.Sequential(*layers)
        self.last_layer = nn.Linear(dim_hidden, dim_out, bias=use_bias)

    def forward(self, x, labels=None):
        x = self.net(x)
        return self.last_layer(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)