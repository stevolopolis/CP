import math

import torch
from torch import nn

#import tinycudann as tcnn

from lib.models.BaseLOD import BaseLOD


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


class HashEmbedder2D(nn.Module):
    """
    Reimplementation of the hash encoder from:
        - HashNerf: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py
    
    to suit the 2D image fitting scenario.
    
    """
    def __init__(self, img_size, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder2D, self).__init__()
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
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            grid_min_vertex, grid_max_vertex, hashed_grid_indices = self.get_grid_vertices(x, resolution)
            #print(torch.min(hashed_grid_indices), torch.max(hashed_grid_indices))
            grid_embedds = self.embeddings[i](hashed_grid_indices)

            x_embedded = self.bilinear_interp(x, grid_min_vertex, grid_max_vertex, grid_embedds)
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


class HashEmbedder3D(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder3D, self).__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_sizes = 2**log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.grid_offset = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        hash_list = []
        for i in range(n_levels):
            resolution = math.floor(self.base_resolution * self.b**i)
            if resolution**3 < self.hashmap_sizes:
                embeddings = nn.Embedding((resolution+1)**3, self.n_features_per_level)
            else:
                embeddings = nn.Embedding(self.hashmap_sizes, self.n_features_per_level)
            hash_list.append(embeddings)

        self.embeddings = nn.ModuleList(hash_list)
        #self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
        #                                self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:,0]*(1-weights[:,0][:,None]) + voxel_embedds[:,4]*weights[:,0][:,None]
        c01 = voxel_embedds[:,1]*(1-weights[:,0][:,None]) + voxel_embedds[:,5]*weights[:,0][:,None]
        c10 = voxel_embedds[:,2]*(1-weights[:,0][:,None]) + voxel_embedds[:,6]*weights[:,0][:,None]
        c11 = voxel_embedds[:,3]*(1-weights[:,0][:,None]) + voxel_embedds[:,7]*weights[:,0][:,None]

        # step 2
        c0 = c00*(1-weights[:,1][:,None]) + c10*weights[:,1][:,None]
        c1 = c01*(1-weights[:,1][:,None]) + c11*weights[:,1][:,None]

        # step 3
        c = c0*(1-weights[:,2][:,None]) + c1*weights[:,2][:,None]

        return c
    
    def get_voxel_vertices(self, xyz, resolution):
        '''
        xyz: 3D coordinates of samples. B x 3
        bounding_box: min and max x,y,z coordinates of object bbox
        resolution: number of voxels per axis
        '''
        box_min, box_max = torch.tensor([-1.0, -1.0, -1.0]), torch.tensor([1.0, 1.0, 1.0])
        box_min = box_min.to(xyz.device)
        box_max = box_max.to(xyz.device)

        keep_mask = xyz==torch.max(torch.min(xyz, box_max), box_min)
        if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
            # print("ALERT: some points are outside bounding box. Clipping them!")
            xyz = torch.clamp(xyz, min=box_min, max=box_max)

        grid_size = (box_max-box_min)/resolution
        
        bottom_left_idx = torch.floor((xyz-box_min)/grid_size).int()
        voxel_min_vertex = bottom_left_idx*grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + grid_size

        voxel_indices = bottom_left_idx.unsqueeze(1) + self.grid_offset.to(xyz.device)
    
        if resolution**3 < self.hashmap_sizes:
            hashed_voxel_indices = self.one2one_hash(voxel_indices, resolution)
        # If hash table is not injective (i.e. number of grid vertices > hash table size)
        else:
            hashed_voxel_indices = self.hash(voxel_indices)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask
    
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
        new_coords = coords[..., 0]*(resolution**2) + coords[..., 1]*resolution + coords[..., 2]
        return new_coords.type(torch.int)

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = self.get_voxel_vertices(\
                                                x, resolution)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1)    

        
class NGP(BaseLOD):
    def __init__(self, args, init=None):
        super().__init__(args)
        
        #self.args = args
        self.hash_mod = True
        #self.hash_table = hash_table
        self.hash_table =  HashEmbedder3D(n_levels=self.args.n_levels,
                              n_features_per_level=self.args.feature_dim,
                              log2_hashmap_size=self.args.log2_n_features,
                              base_resolution=self.args.base_resolution,
                              finest_resolution=self.args.finest_resolution)

        # Hash Table Parameters        
        hash_levels = self.hash_table.n_levels
        hash_feature_dim = self.hash_table.n_features_per_level
        in_features = hash_levels * hash_feature_dim

        self.net = []

        self.net.append(ReluLayer(in_features, self.args.hidden_features))

        for i in range(self.args.hidden_layers):
            self.net.append(ReluLayer(self.args.hidden_features, self.args.hidden_features))

        self.net.append(nn.Linear(self.args.hidden_features, self.args.out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, lod=None):
        return self.sdf(x)

    def sdf(self, x, hash=True, lod=None):
        if hash:
            x = self.hash_table(x)
            
        output = self.net(x)

        return output
