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
        for i in range(n_levels):
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

    def forward(self, x):
        # x is 2D point position: B x 2
        x_embedded_all = []
        hashed_grid_indices_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            grid_min_vertex, grid_max_vertex, hashed_grid_indices = self.get_grid_vertices(x, resolution)
            hashed_grid_indices_all.append(hashed_grid_indices)
            #print(torch.min(hashed_grid_indices), torch.max(hashed_grid_indices))
            grid_embedds = self.embeddings[i](hashed_grid_indices)

            x_embedded = self.bilinear_interp(x, grid_min_vertex, grid_max_vertex, grid_embedds)
            x_embedded_all.append(x_embedded)

        return torch.cat(x_embedded_all, dim=-1), hashed_grid_indices_all

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
        if (resolution + 1) ** 2 <= self.hashmap_sizes:
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


class ConsistentHashEmbedder(HashEmbedder):
    """Notes:
        - Consistent hashing: https://en.wikipedia.org/wiki/Consistent_hashing
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # for verification purposes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # track gradient accumulation
        start_level = 0
        for level in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b ** level).item()
            if (resolution + 1) ** 2 > self.hashmap_sizes:
                start_level = level
                break
        self.start_level = start_level
        self.first_time = [True for _ in range(self.start_level, self.n_levels)]
        # need a list for this, since each level's embedding shape may change
        self.table_grad_accum = [
            torch.zeros(
                2 ** self.log2_hashmap_size,
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(self.start_level, self.n_levels)
        ]
        self.denom = [torch.zeros_like(self.table_grad_accum[0]) for _ in range(self.start_level, self.n_levels)]

        # initialize circle
        # multiply coordinates in {0, ..., resolution}^2 by this to convert to position on circle
        self.index_to_radians = 2 * torch.pi / 2 ** self.log2_hashmap_size
        resolutions = []
        self.resolution_map = {}
        for level in range(self.start_level, self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b ** level).item()
            resolutions.append(resolution)
            self.resolution_map[resolution] = level

        target_key = []  # location on unit circle
        target_value = []  # index in embedding

        target_average = []

        for i, resolution in enumerate(resolutions):
            x, y = torch.meshgrid(torch.arange(resolution + 1), torch.arange(resolution + 1), indexing="xy")
            xy = torch.stack([x, y], dim=-1).view(-1, 2).to(self.device)
            # integer hash
            xy_hashed = self.hash_integer(xy.to(int))
            xy_hashed_radians = xy_hashed.to(torch.float64) * self.index_to_radians
            # sort xy_hashed_radians and bring xy_hashed with it
            xy_hashed_radians, from_index = torch.sort(xy_hashed_radians)
            xy_hashed = xy_hashed[from_index]
            target = xy_hashed_radians + self.index_to_radians
            assert torch.all(0 < target) and torch.all(target <= 2 * torch.pi)
            target_key.append(target)
            target_value.append(xy_hashed)

            target_average.append(
                torch.zeros((len(target_key), 2), dtype=torch.float, device=self.device)
            )

        self.target_key = target_key
        self.target_value = target_value

        self.target_average = target_average

    def get_grid_vertices(self, xy, resolution):
        '''
        xy: 2D coordinates of samples. B x 2
        resolution: number of grids per axis
        '''
        x_grid_size = self.img_w / resolution
        y_grid_size = self.img_h / resolution
        grid_size = torch.tensor([y_grid_size, x_grid_size]).to(xy.device)
        bottom_left_idx = torch.floor(torch.div(xy, grid_size)).int()
        grid_min_vertex = torch.mul(bottom_left_idx, grid_size)
        grid_max_vertex = grid_min_vertex + grid_size

        grid_indices = bottom_left_idx.unsqueeze(1) + self.grid_offset.to(xy.device)
        if (resolution + 1) ** 2 <= self.hashmap_sizes:
            hash_grid_indices = self.one2one_hash(grid_indices, resolution)
        # If hash table is not injective (i.e. number of grid vertices > hash table size)
        else:
            hash_grid_indices = self.consistent_hash(grid_indices, resolution)

        return grid_min_vertex, grid_max_vertex, hash_grid_indices

    def hash(self, coords):
        '''
        coords: this function can process upto 7 dim coordinates
        log2T:  logarithm of T w.r.t 2
        '''
        hash = self.hash_integer(coords) + self.hash_decimal(coords)
        assert torch.all(hash < (1 << self.log2_hashmap_size))

        return hash

    def hash_integer(self, coords):
        '''
        coords: this function can process upto 7 dim coordinates
        log2T:  logarithm of T w.r.t 2
        '''
        primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

        xor_result = torch.zeros_like(coords)[..., 0]
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i]*primes[i]

        return torch.tensor((1 << self.log2_hashmap_size)-1).to(xor_result.device) & xor_result

    def hash_decimal(self, coords):
        '''
        coords: this function can process upto 6 dim coordinates
        log2T:  logarithm of T w.r.t 2
        '''
        log2_decimal_size = 12
        pad = 1.0 / (1 << (log2_decimal_size + 1))

        primes = [2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

        xor_result = torch.zeros_like(coords)[..., 0]
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i]*primes[i]

        decimal = (torch.tensor((1 << log2_decimal_size)-1).to(xor_result.device) & xor_result).to(torch.float64) / (1 << log2_decimal_size)
        decimal = decimal + pad
        assert torch.all(decimal > 0) and torch.all(decimal < 1)

        return decimal

    def consistent_hash(self, coords, resolution):
        level = self.resolution_map[resolution.item()] - self.start_level
        spatial_hash = self.hash(coords)
        target_query = spatial_hash * self.index_to_radians
        target_key = self.target_key[level]
        key = torch.searchsorted(target_key, target_query)
        # wrap around
        # key[key == target_key.nelement()] = 0
        value = self.target_value[level][key]
        if self.first_time[level]:
            self.first_time[level] = False
            assert torch.all(value == self.hash_integer(coords))
        return value

    def densify(self):
        """If densify_grad_threshold is not set, remove anything above two standard deviations past mean"""
        for i in range(len(self.table_grad_accum)):
            # setup
            level = i + self.start_level
            accum = self.table_grad_accum[i] / self.denom[i]

            print(f"nan={accum.isnan().nonzero().sum()}")

            accum[accum.isnan()] = 0
            # set threshold to anything above mu + 2 * sigma
            mu = accum.mean()
            sigma = accum.var() ** 0.5
            densify_grad_threshold = mu + sigma

            # do top 5 percent
            densify_grad_threshold = torch.sort(accum)[0][int(len(accum) * 0.90)]

            target_key = self.target_key[i]
            target_value = self.target_value[i]
            sorted_target_value, from_index = torch.sort(target_value)

            # densify
            large_accum_index = (accum > densify_grad_threshold).nonzero().flatten()

            if len(large_accum_index) == 0:
                print(f"RETURNED; {i}")
                return

            target_index = from_index[torch.searchsorted(sorted_target_value, large_accum_index)]
            assert target_index.max() < len(sorted_target_value)
            # TODO: treat edge case
            shifted_target_index = (target_index - 1).clamp(0, len(sorted_target_value) - 1)
            midpoints = (target_key[target_index] + target_key[shifted_target_index]) / 2
            num_new_entries = len(midpoints)
            start_new_entries = sorted_target_value[-1] + 1
            end_new_entries = start_new_entries + num_new_entries  # not including
            assert start_new_entries == self.embeddings[level].weight.shape[0]
            # NOTE: target_key must be in sorted order
            target_key = torch.cat([target_key, midpoints])
            target_value = torch.cat([target_value, torch.arange(
                start_new_entries, end_new_entries, device=target_value.device, dtype=target_value.dtype)])
            target_key, from_index = torch.sort(target_key)
            target_value = target_value[from_index]
            self.embeddings[level].weight = nn.Parameter(torch.cat([
                self.embeddings[level].weight,
                torch.zeros((num_new_entries, self.n_features_per_level),
                            device=self.embeddings[level].weight.device,
                            dtype=self.embeddings[level].weight.dtype),
            ]))
            self.table_grad_accum[i] = torch.cat([self.table_grad_accum[i],
                                                  torch.arange(start_new_entries, end_new_entries,
                                                               device=self.table_grad_accum[i].device,
                                                               dtype=self.table_grad_accum[i].dtype),])
            self.denom[i] = torch.cat([self.denom[i], torch.arange(start_new_entries, end_new_entries,
                                                                   device=self.denom[i].device,
                                                                   dtype=self.denom[i].dtype), ])

            print(f"more={len(self.table_grad_accum[i]) - 1024}")

            # update values
            self.target_key[i] = target_key
            self.target_value[i] = target_value
            self.reset_table_grad_accum(i)

    def update_table_grad_accum(self, i, mask: torch.Tensor):
        norm_grad = self.embeddings[i + self.start_level].weight.grad.norm(dim=-1)
        self.table_grad_accum[i][mask] += norm_grad[mask]
        self.denom[i][mask] += 1

    def reset_table_grad_accum(self, i):
        self.table_grad_accum[i] *= 0
        self.denom[i] *= 0


class NGP(nn.Module):
    def __init__(self,
                hash_table,
                hidden_features,
                hidden_layers,
                out_features):
                
        super().__init__()

        self.hash_mod = True
        self.hash_table = hash_table

        # Hash Table Parameters
        
        hash_levels = hash_table.n_levels
        hash_feature_dim = hash_table.n_features_per_level
        in_features = hash_levels * hash_feature_dim

        self.net = []

        self.net.append(ReluLayer(in_features, hidden_features))

        for i in range(hidden_layers):
            self.net.append(ReluLayer(hidden_features, hidden_features))

        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, hash=True):
        if hash:
            x, hashed_grid_indices = self.hash_table(x)
            
        output = self.net(x)

        return output, hashed_grid_indices

"""
class TinyHashEmbedder(nn.Module):
    def __init__(self, img_size, n_levels=16, n_features_per_level=2,\
            log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(TinyHashEmbedder, self).__init__()
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
        
        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        self.embedder =  tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,   
                "log2_hashmap_size": log2_hashmap_size, 
                "base_resolution": base_resolution,
                "per_level_scale": self.b.item(),
            },
        )

    def forward(self, x):
        return self.embedder(x)
    

class TinyNGP(nn.Module):
    def __init__(self,
                hash_table,
                hidden_features,
                hidden_layers,
                out_features):
                
        super().__init__()

        # Hash Table Parameters
        
        hash_levels = hash_table.n_levels
        hash_feature_dim = hash_table.n_features_per_level
        in_features = hash_levels * hash_feature_dim

        self.net = tcnn.Network(
            n_input_dims=in_features,
            n_output_dims=out_features,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_features,
                "n_hidden_layers": hidden_layers,
            },
        )

    def forward(self, x):
        return self.net(x)
"""
