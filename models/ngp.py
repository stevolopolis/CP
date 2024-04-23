import math

import torch
from torch import nn


class ReluLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        #change weights to glorot and bengio as state in the paper
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
    
    def forward(self, input):
        return torch.relu(self.linear(input))

class HashEmbedder1D(nn.Module):
    """
    Reimplementation of the hash encoder from:
        - HashNerf: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py
    
    to suit the 1D image fitting scenario.
    
    """
    def __init__(self, img_size, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder1D, self).__init__()
        self.img_size = img_size[0]
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_sizes = 2**log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.grid_offset = torch.tensor([[0], [1]])

        if n_levels == 1:
            self.b = 1.0
        else:
            self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        hash_list = []
        for i in range(n_levels):
            resolution = math.floor(self.base_resolution * self.b**i)
            if resolution**2 < self.hashmap_sizes:
                embeddings = nn.Embedding((resolution+1)**2, self.n_features_per_level)
            else:
                embeddings = nn.Embedding(self.hashmap_sizes, self.n_features_per_level)
            hash_list.append(embeddings)

        self.embeddings = nn.ModuleList(hash_list)
        #self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
        #                                self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        self.reset_parameters()

    def reset_parameters(self, ordered=False):
        if ordered:
            for i in range(self.n_levels):
                n_weights = self.embeddings[i].weight.shape[0]
                self.embeddings[i].weight.data = torch.linspace(-0.0001, 0.0001, n_weights).unsqueeze(1).float().to(self.embeddings[i].weight.device)
        else:
            for i in range(self.n_levels):
                nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def linear_interp(self, x, grid_min_vertex, grid_max_vertex, grid_embedds):
        '''
        x: B x 1
        grid_min_vertex: B x 1
        grid_max_vertex: B x 1
        grid_embedds: B x 2 x 2
        '''
        weights = (x - grid_min_vertex)/(grid_max_vertex-grid_min_vertex) # B x 1

        c = grid_embedds[:,0]*(1-weights) + grid_embedds[:,1]*weights

        return c

    def forward(self, x, interp=True):
        # x is 1D point position: B x 1
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            grid_min_vertex, grid_max_vertex, hashed_grid_indices = self.get_grid_vertices(x, resolution)
            #print(torch.min(hashed_grid_indices), torch.max(hashed_grid_indices))
            grid_embedds = self.embeddings[i](hashed_grid_indices)
            
            if interp:
                x_embedded = self.linear_interp(x, grid_min_vertex, grid_max_vertex, grid_embedds)
            else:
                x_embedded = grid_embedds
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)

    def get_grid_vertices(self, x, resolution):
        '''
        x: 1D coordinates of samples. B x 1
        resolution: number of grids per axis
        '''
        grid_size = self.img_size/resolution
        left_idx = torch.floor(torch.div(x, grid_size)).int()
        grid_min_vertex = torch.mul(left_idx, grid_size)
        grid_max_vertex = grid_min_vertex + grid_size
        
        grid_indices = left_idx.unsqueeze(1) + self.grid_offset.to(x.device)
        if resolution**2 < self.hashmap_sizes:
            hash_grid_indices = self.one2one_hash(grid_indices, resolution)
        # If hash table is not injective (i.e. number of grid vertices > hash table size)
        else:
            hash_grid_indices = self.hash(grid_indices)

        return grid_min_vertex, grid_max_vertex, hash_grid_indices
    
    def hash(self, coords):
        '''
        coords: this function can process upto 7 dim coordinates
        log2T:  logarithm of T w.r.t 2
        '''
        primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

        xor_result = torch.zeros_like(coords)[..., 0]
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i]*primes[i]

        return torch.tensor((1<<self.log2_hashmap_size)-1).to(xor_result.device) & xor_result

    def one2one_hash(self, coords, resolution):
        new_coords = coords[..., 0]
        return new_coords.type(torch.int)


class HashEmbedder(nn.Module):
    """
    Reimplementation of the hash encoder from:
        - HashNerf: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py
    
    to suit the 2D image fitting scenario.
    
    """
    def __init__(self, img_size, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder, self).__init__()
        self.img_size = img_size
        self.img_h = img_size[0]
        self.img_w = img_size[1]
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_sizes = 2**log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.grid_offset = torch.tensor([[i,j] for i in [0, 1] for j in [0, 1]])

        if n_levels == 1:
            self.b = 1.0
        else:
            self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        hash_list = []
        for i in range(n_levels):
            resolution = math.floor(self.base_resolution * self.b**i)
            if resolution**2 < self.hashmap_sizes:
                embeddings = nn.Embedding((resolution+1)**2, self.n_features_per_level)
            else:
                embeddings = nn.Embedding(self.hashmap_sizes, self.n_features_per_level)
            hash_list.append(embeddings)

        self.embeddings = nn.ModuleList(hash_list)
        # custom uniform initialization
        self.reset_parameters()

    def reset_parameters(self, ordered=False):
        if ordered:
            for i in range(self.n_levels):
                resolution = math.floor(self.base_resolution * self.b**i)
                if resolution**2 < self.hashmap_sizes:
                    n_weights_per_axis = resolution+1
                else:
                    n_weights_per_axis = int(math.sqrt(self.hashmap_sizes))
                xw = torch.linspace(-0.0001, 0.0001, n_weights_per_axis)
                yw = torch.linspace(-0.0001, 0.0001, n_weights_per_axis)
                weight_grid = torch.stack(torch.meshgrid(xw, yw), dim=1).view(-1, 2).to(self.embeddings[i].weight.device)
                self.embeddings[i].weight.data = weight_grid
        else:
            for i in range(self.n_levels):
                nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def bilinear_interp(self, x, grid_min_vertex, grid_max_vertex, grid_embedds):
        '''
        x: B x 2
        grid_min_vertex: B x 2
        grid_max_vertex: B x 2
        grid_embedds: B x 4 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Bilinear_interpolation
        weights = (x - grid_min_vertex)/(grid_max_vertex-grid_min_vertex) # B x 2

        # step 1
        # 0->00, 1->01, 2->10, 3->11
        c0 = grid_embedds[:,0]*(1-weights[:,1][:,None]) + grid_embedds[:,1]*weights[:,1][:,None]
        c1 = grid_embedds[:,2]*(1-weights[:,1][:,None]) + grid_embedds[:,3]*weights[:,1][:,None]

        # step 2
        c = c0*(1-weights[:,0][:,None]) + c1*weights[:,0][:,None]

        return c

    def forward(self, x, interp=True):
        # x is 2D point position: B x 2
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            grid_min_vertex, grid_max_vertex, hashed_grid_indices = self.get_grid_vertices(x, resolution)
            #print(torch.min(hashed_grid_indices), torch.max(hashed_grid_indices))
            grid_embedds = self.embeddings[i](hashed_grid_indices)

            if interp:
                x_embedded = self.bilinear_interp(x, grid_min_vertex, grid_max_vertex, grid_embedds)
            else: 
                x_embedded = hashed_grid_indices
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1)

    def get_grid_vertices(self, xy, resolution):
        '''
        xy: 2D coordinates of samples. B x 2
        resolution: number of grids per axis
        '''
        x_grid_size = self.img_w/resolution
        y_grid_size = self.img_h/resolution
        grid_size = torch.tensor([y_grid_size, x_grid_size]).to(xy.device)
        bottom_left_idx = torch.floor(torch.div(xy, grid_size)).int()
        grid_min_vertex = torch.mul(bottom_left_idx, grid_size)
        grid_max_vertex = grid_min_vertex + grid_size
        
        grid_indices = bottom_left_idx.unsqueeze(1) + self.grid_offset.to(xy.device)
        if resolution**2 < self.hashmap_sizes:
            hash_grid_indices = self.one2one_hash(grid_indices, resolution)
        # If hash table is not injective (i.e. number of grid vertices > hash table size)
        else:
            hash_grid_indices = self.hash(grid_indices)

        return grid_min_vertex, grid_max_vertex, hash_grid_indices
    
    def hash(self, coords):
        '''
        coords: this function can process upto 7 dim coordinates
        log2T:  logarithm of T w.r.t 2
        '''
        primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

        xor_result = torch.zeros_like(coords)[..., 0]
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i]*primes[i]

        return torch.tensor((1<<self.log2_hashmap_size)-1).to(xor_result.device) & xor_result

    def one2one_hash(self, coords, resolution):
        new_coords = coords[..., 0]*resolution + coords[..., 1]
        return new_coords.type(torch.int)

class NGP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        data_size,
        ngp_configs
    ):
        super().__init__()
        self.ngp_configs = ngp_configs.NET
        self.dim_in = dim_in
        self.dim_out = dim_out
        dim_hidden = self.ngp_configs.dim_hidden
        if dim_in == 1:
            self.hash_table = HashEmbedder1D(data_size,
                                        n_levels=self.ngp_configs.n_levels,
                                        n_features_per_level=self.ngp_configs.feature_dim,
                                        log2_hashmap_size=self.ngp_configs.log2_n_features,
                                        base_resolution=self.ngp_configs.base_resolution,
                                        finest_resolution=self.ngp_configs.finest_resolution)
        elif dim_in == 2:
            self.hash_table = HashEmbedder(data_size,
                                        n_levels=self.ngp_configs.n_levels,
                                        n_features_per_level=self.ngp_configs.feature_dim,
                                        log2_hashmap_size=self.ngp_configs.log2_n_features,
                                        base_resolution=self.ngp_configs.base_resolution,
                                        finest_resolution=self.ngp_configs.finest_resolution)
        # Hash Table Parameters
        in_features = self.ngp_configs.n_levels * self.ngp_configs.feature_dim

        self.net = []

        self.net.append(ReluLayer(in_features, dim_hidden))

        for i in range(self.ngp_configs.num_layers - 2):
            self.net.append(ReluLayer(dim_hidden, dim_hidden))

        self.net.append(nn.Linear(dim_hidden, dim_out))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, hash=True, interp=True):
        x = self.hash_table(x, interp=interp) if hash else x
        output = self.net(x)

        return output

    def init_weights(self, ordered=False):
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            
        self.hash_table.reset_parameters(ordered=ordered)
