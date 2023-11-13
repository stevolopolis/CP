import itertools
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from lib.models.siren import Siren
from tqdm import tqdm
import logging

from lib.models.BaseLOD import BaseLOD

log = logging.getLogger(__name__)

def general_rearrange(A, dimensions, n_levels, channel=3):
    import string
    all_chars = string.ascii_lowercase

    ndim = A.ndim
    from_str = ""
    for i in range(ndim-1):
        tmp_str = "("
        for j in range(n_levels):
            tmp_str += f"{all_chars[i]}{j+1} "
        from_str += tmp_str[:-1] + ") "

    to_str = ""
    for j in range(n_levels):
        tmp_str = "("
        for i in range(ndim-1):
            tmp_str += f"{all_chars[i]}{j+1} "
        to_str += tmp_str[:-1] + ") "
        
    input_str = from_str + " k -> " + to_str + " k"
    
    kwargs = {}
    for i, dim in enumerate(dimensions):
        for j in range(n_levels):
            kwargs[f"{all_chars[i]}{j+1}"] = dim[j]
    
    kwargs["k"] = channel
    return rearrange(A, input_str, **kwargs)


class Modulator(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30.):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        self.w0 = w0

    def forward(self, x):
        return self.w0*self.linear(x)


class Modulator_nonlinear(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30.): 
        from easydict import EasyDict
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        siren_config = EasyDict({"NET": {"num_layers":3, "dim_hidden":32, "w0":w0, "w0_initial": w0, "use_bias": True}})
        self.model = Siren(dim_in, dim_out, siren_config)

    def forward(self, x):
        return self.model(x)


class Embedder(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30.):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.embedding = nn.Embedding(dim_in**2, dim_out)

    def forward(self, x):
        """
        x is the tensorized and normalized xy coordinates.
        We have to calculate the index for each coordinate combination.
        """
        idx = x[..., 1] * self.dim_in + x[..., 0]
        return self.embedding(idx)
    

class EmbeddedModulator(nn.Module):
    def __init__(self, dim_in, dim_out, tile_dim=2, w0=30.):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.tile_dim = tile_dim
        self.embedding = nn.Embedding(tile_dim**2, tile_dim**2)
        self.linear = nn.Linear(tile_dim**2, dim_out, bias=False)
        self.w0 = w0

    def forward(self, x):
        """
        x is the tensorized and normalized xy coordinates.
        We have to calculate the index for each coordinate combination.
        """
        idx = x[..., 1] * self.tile_dim + x[..., 0]
        return self.w0 * self.linear(self.embedding(idx))


class TACO_hashed(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        data_size,
        taco_configs
    ):
        super().__init__()
        self.taco_configs = taco_configs.NET
        self.dim_in = dim_in
        self.dim_out = dim_out
        num_layers = self.taco_configs.num_layers
        dim_hidden = self.taco_configs.dim_hidden
        w0 = self.taco_configs.w0

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.data_size = data_size
        self.dimensions = self.taco_configs.dimensions # list of lists
        self.use_proxy = self.taco_configs.use_proxy
        self.eps = self.taco_configs.eps
        self.partition_fixed = True
        self.random_partition = False

        assert all(self.data_size[i] == math.prod(self.dimensions[i]) for i in range(len(self.data_size))), "The product of dimensions along each axis should equal the size of data along that axis"
        assert all(len(dim) == len(self.dimensions[0]) for dim in self.dimensions), "The no. of resolution levels along each axis should match"

        self.n_levels = len(self.dimensions[0])
        self.modulators = nn.ModuleList([EmbeddedModulator(dim_in, dim_hidden, tile_dim=self.dimensions[0][0], w0=w0) for _ in range(self.n_levels-1)])
        
        siren = Siren(
            dim_in=dim_in,
            dim_out=dim_out,
            siren_configs=taco_configs
        )        

        self.net = siren.net
        self.last_layer = siren.last_layer


    def get_all_permutations(self):
        ndim = len(self.dimensions)
        all_permutations = []
        all_dimension_permutations = [list(set(itertools.permutations(self.dimensions[i]))) for i in range(ndim)] # unique permutations along each axis
        for idx in itertools.product(*[list(range(len(dimension_permutations))) for dimension_permutations in all_dimension_permutations]):
            all_permutations.append([all_dimension_permutations[i][idx[i]] for i in range(ndim)])
        return sorted(list(all_permutations))

    def ttsvd(self, A, eps=0.1):
        d = A.ndim
        B = [None] * d
        C = A.clone()
        delta = (eps / math.sqrt(d-1))*torch.norm(A)
        ranks = [1] + [None] * (d-1)
        for k in range(1, d):
            C = C.reshape(ranks[k-1] * A.shape[k-1], -1)
            U, S, V = torch.linalg.svd(C, full_matrices=False)
            # truncate SVD with reconstruction error < delta 
            for i in range(1, len(S)+1):
                C_temp = U[:, :i] @ torch.diag_embed(S[:i]) @ V[:i, :]
                E = torch.norm(C - C_temp)
                if E <= delta:
                    break

            r = i
            ranks[k] = r
            U = U[:, :r].reshape(-1, A.shape[k-1], r)
            S = S[:r]
            V = V[:r, :].reshape(r, -1)
            B[k-1] = U
            C = S.reshape(-1, 1) * V

        B[d-1] = C.reshape(ranks[d-1], A.shape[d-1], -1)
        n_params = sum([B[i].numel() for i in range(d)])
        return B, ranks, n_params

    def ttsvd_proxy(self, A, eps=0.1):
        all_permutations = self.get_all_permutations()
        least_n_params = float('inf')
        best_permutation = []
        print("Using proxy to find best partition...")
        for permutation in tqdm(all_permutations):
            reshaped_A = general_rearrange(A, permutation, self.n_levels, channel=self.dim_out)
            _, _, n_params = self.ttsvd(reshaped_A, eps=eps)
            if n_params < least_n_params:
                least_n_params = n_params
                best_permutation =  permutation
        return best_permutation
        
    def decompose_coords(self, coords):
        d = len(self.data_size)
        cum_bases = [self.data_size[i] // np.cumprod(self.dimensions[i]) for i in range(d)]
        ori_coords = [coords[..., i].unsqueeze(1) for i in range(d)]
        level_coords = []
        for l in range(self.n_levels):
            coords_l = [torch.floor_divide(ori_coords[i], cum_bases[i][l]) % self.dimensions[i][l] for i in range(d)]
            level_coords.append(torch.cat(coords_l, dim=-1))

        return torch.stack(level_coords, dim=1).type(torch.int)

    def normalize_coords(self, coords):
        coords_max = (torch.tensor(self.dimensions).T - 1).unsqueeze(0).to(coords.device)
        coords /= coords_max # [0, 1]
        return coords

    def forward(self, x, labels=None):
        if not self.partition_fixed: # do once only
            if self.use_proxy:
                self.dimensions = self.ttsvd_proxy(labels.reshape(*self.data_size, -1), eps=self.eps)
                print(f"Epsilon: {self.eps}")
            else: # random select
                all_permutations = self.get_all_permutations()
                idx = np.random.randint(len(all_permutations))
                self.dimensions = all_permutations[idx]
                
            self.partition_fixed = True
            print("Partition fixed: ", self.dimensions)
            log.info("Partition fixed: ", self.dimensions)

        x_levels = self.decompose_coords(x) # [b, hw, n_levels, c]
        #x_levels = self.normalize_coords(x_levels)
        #modulations = [self.modulators[i-1](x_levels[..., i, :]) for i in range(1, self.n_levels)]
        modulations = [i for i in range(1, self.n_levels)]      # To reducec GPU memory load without losing much inference speed
        x = x_levels[..., 0, :].type(torch.float32) # first level coords
        for i, module in enumerate(self.net):
            x = module.linear(x)
            if i < len(modulations):
                x = x + self.modulators[i](x_levels[..., i+1, :]) #modulations[i]
            else:
                x = x + self.modulators[-1](x_levels[..., -1, :]) #modulations[-1] # if number layers > number of modulations, reuse the last modulation
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

        out = self.last_layer(x)
        return out


class TACO_nonlinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        data_size,
        taco_configs
    ):
        super().__init__()
        self.taco_configs = taco_configs.NET
        self.dim_in = dim_in
        self.dim_out = dim_out
        num_layers = self.taco_configs.num_layers
        dim_hidden = self.taco_configs.dim_hidden
        w0 = self.taco_configs.w0

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.data_size = data_size
        self.dimensions = self.taco_configs.dimensions # list of lists
        self.use_proxy = self.taco_configs.use_proxy
        self.eps = self.taco_configs.eps
        self.partition_fixed = True

        assert all(self.data_size[i] == math.prod(self.dimensions[i]) for i in range(len(self.data_size))), "The product of dimensions along each axis should equal the size of data along that axis"
        assert all(len(dim) == len(self.dimensions[0]) for dim in self.dimensions), "The no. of resolution levels along each axis should match"

        self.n_levels = len(self.dimensions[0])
        #self.modulators = nn.ModuleList([Modulator(dim_in, dim_hidden, w0=w0) for _ in range(self.n_levels-1)])
        
        self.modulators = nn.ModuleList([Modulator_nonlinear(dim_in, dim_hidden, w0=w0) for _ in range(self.n_levels-1)])
        
        siren = Siren(
            dim_in=dim_in,
            dim_out=dim_out,
            siren_configs=taco_configs
        )        

        self.net = siren.net
        self.last_layer = siren.last_layer


    def get_all_permutations(self):
        ndim = len(self.dimensions)
        all_permutations = []
        all_dimension_permutations = [list(set(itertools.permutations(self.dimensions[i]))) for i in range(ndim)] # unique permutations along each axis
        for idx in itertools.product(*[list(range(len(dimension_permutations))) for dimension_permutations in all_dimension_permutations]):
            all_permutations.append([all_dimension_permutations[i][idx[i]] for i in range(ndim)])
        return sorted(list(all_permutations))

    def ttsvd(self, A, eps=0.1):
        d = A.ndim
        B = [None] * d
        C = A.clone()
        delta = (eps / math.sqrt(d-1))*torch.norm(A)
        ranks = [1] + [None] * (d-1)
        for k in range(1, d):
            C = C.reshape(ranks[k-1] * A.shape[k-1], -1)
            U, S, V = torch.linalg.svd(C, full_matrices=False)
            # truncate SVD with reconstruction error < delta 
            for i in range(1, len(S)+1):
                C_temp = U[:, :i] @ torch.diag_embed(S[:i]) @ V[:i, :]
                E = torch.norm(C - C_temp)
                if E <= delta:
                    break

            r = i
            ranks[k] = r
            U = U[:, :r].reshape(-1, A.shape[k-1], r)
            S = S[:r]
            V = V[:r, :].reshape(r, -1)
            B[k-1] = U
            C = S.reshape(-1, 1) * V

        B[d-1] = C.reshape(ranks[d-1], A.shape[d-1], -1)
        n_params = sum([B[i].numel() for i in range(d)])
        return B, ranks, n_params

    def ttsvd_proxy(self, A, eps=0.1):
        all_permutations = self.get_all_permutations()
        least_n_params = float('inf')
        best_permutation = []
        print("Using proxy to find best partition...")
        for permutation in tqdm(all_permutations):
            reshaped_A = general_rearrange(A, permutation, self.n_levels, channel=self.dim_out)
            _, _, n_params = self.ttsvd(reshaped_A, eps=eps)
            if n_params < least_n_params:
                least_n_params = n_params
                best_permutation =  permutation
        return best_permutation
        
    def decompose_coords(self, coords):
        d = len(self.data_size)
        cum_bases = [self.data_size[i] // np.cumprod(self.dimensions[i]) for i in range(d)]
        ori_coords = [coords[..., i].unsqueeze(1) for i in range(d)]
        level_coords = []
        for l in range(self.n_levels):
            #coords_l = [torch.floor_divide(ori_coords[i], cum_bases[i][l]) % self.dimensions[i][l] for i in range(d)]
            coords_l = [torch.divide(ori_coords[i], cum_bases[i][l]) % self.dimensions[i][l] for i in range(d)]
            level_coords.append(torch.cat(coords_l, dim=-1))

        return torch.stack(level_coords, dim=1)

    def normalize_coords(self, coords):
        coords_max = (torch.tensor(self.dimensions).T - 1).unsqueeze(0).to(coords.device)
        coords /= coords_max # [0, 1]
        return coords

    def forward(self, x, labels=None):
        if not self.partition_fixed: # do once only
            if self.use_proxy:
                self.dimensions = self.ttsvd_proxy(labels.reshape(*self.data_size, -1), eps=self.eps)
                print(f"Epsilon: {self.eps}")
            else: # random select
                all_permutations = self.get_all_permutations()
                idx = np.random.randint(len(all_permutations))
                self.dimensions = all_permutations[idx]
                
            self.partition_fixed = True
            print("Partition fixed: ", self.dimensions)
            log.info("Partition fixed: ", self.dimensions)

        x_levels = self.decompose_coords(x) # [b, hw, n_levels, c]
        x_levels = self.normalize_coords(x_levels)
        #modulations = [self.modulators[i-1](x_levels[..., i, :]) for i in range(1, self.n_levels)]
        modulations = [i for i in range(1, self.n_levels)]      # To reducec GPU memory load without losing much inference speed
        x = x_levels[..., 0, :] # first level coords
        for i, module in enumerate(self.net):
            x = module.linear(x)
            if i < len(modulations):
                x = x + self.modulators[i](x_levels[..., i+1, :]) #modulations[i]
            else:
                x = x + self.modulators[-1](x_levels[..., -1, :]) #modulations[-1] # if number layers > number of modulations, reuse the last modulation
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

        out = self.last_layer(x)
        return out


class TACO(BaseLOD):
    def __init__(
        self, args
    ):
        super().__init__(args)
        self.taco_configs = args
        self.dim_in = self.taco_configs.dim_in
        self.dim_out = self.taco_configs.dim_out
        self.dim_hidden = self.taco_configs.dim_hidden
        w0 = self.taco_configs.w0

        self.data_resolution = self.taco_configs.data_resolution
        #self.dimensions = self.taco_configs.dimensions # list of lists
        self.dimensions = [[4,4,4,2,2,2,2,2,2],[4,4,4,2,2,2,2,2,2],[4,4,4,2,2,2,2,2,2]]
        #self.dimensions = [[4,4,4,4,4,2,2],[4,4,4,4,4,2,2],[4,4,4,4,4,2,2]]
        #self.dimensions = [[16,8,8],[16,8,8],[16,8,8]]
        self.random_partition = False

        #dim_hidden = [512, 384, 256, 256, 256, 256, 256]

        #assert all(self.data_size[i] == math.prod(self.dimensions[i]) for i in range(len(self.data_size))), "The product of dimensions along each axis should equal the size of data along that axis"
        assert all(len(dim) == len(self.dimensions[0]) for dim in self.dimensions), "The no. of resolution levels along each axis should match"

        self.n_levels = len(self.dimensions[0])
        self.modulators = nn.ModuleList([Modulator(self.dim_in, self.dim_hidden, w0=w0) for i in range(self.n_levels-1)])

        siren = Siren(
            dim_in=self.dim_in,
            dim_out=self.dim_out,
            siren_configs=self.taco_configs
        )        

        self.net = siren.net
        self.last_layer = siren.last_layer

        if self.random_partition:
            print("Use random partition...")
            all_permutations = self.get_all_permutations()
            idx = np.random.randint(len(all_permutations))
            self.dimensions = all_permutations[idx]
        else:
            print("Use fixed partition...")
            
        log.info("Partition: " + str(self.dimensions))
        self.partition_coords = []
        for i in range(self.n_levels):
            grid = [torch.linspace(0., 1., self.dimensions[j][i]) for j in range(self.dim_in)]
            self.partition_coords.append(torch.stack(torch.meshgrid(grid), dim=-1).view(-1, self.dim_in))

    def get_all_permutations(self):
        ndim = len(self.dimensions)
        all_permutations = []
        all_dimension_permutations = [list(set(itertools.permutations(self.dimensions[i]))) for i in range(ndim)] # unique permutations along each axis
        for idx in itertools.product(*[list(range(len(dimension_permutations))) for dimension_permutations in all_dimension_permutations]):
            all_permutations.append([all_dimension_permutations[i][idx[i]] for i in range(ndim)])
        return sorted(list(all_permutations))

    def decompose_coords(self, coords):
        d = coords.shape[-1]
        # Normalize coords from [-1, 1] to [0, 1]
        coords = coords * .5 + .5
        # Scale coords to specified resolution
        coords = (coords * self.data_resolution).int().float()
        # Start decomposing coords
        cum_bases = [self.data_resolution // np.cumprod(self.dimensions[i]) for i in range(d)]
        ori_coords = [coords[..., i].unsqueeze(1) for i in range(d)]
        level_coords = []
        for l in range(self.n_levels-1):
            coords_l = [torch.floor_divide(ori_coords[i], cum_bases[i][l]) % self.dimensions[i][l] for i in range(d)]
            level_coords.append(torch.cat(coords_l, dim=-1))

        coords_l = [ori_coords[i] for i in range(d)]
        level_coords.append(torch.cat(coords_l, dim=-1))
        return torch.stack(level_coords, dim=1)                 

    def normalize_coords(self, coords):
        coords_max = (torch.tensor(self.dimensions).T - 1).unsqueeze(0).to(coords.device)
        coords /= coords_max # [0, 1]
        coords[coords != coords] = 0       # Remove Nan when certain coords at certain partition is 1 (i.e. max=0)
        return coords

    def forwardz(self, x, labels=None):
        # TACO with hierarchical forward to save MACs
        d = x.shape[-1]
        cumprods = [np.cumprod(self.dimensions[i]) for i in range(d)]
        modulations = [self.modulators[i-1](self.partition_coords[i].to(x.device)) for i in range(1, self.n_levels)]
        x = self.partition_coords[0].to(x.device)
        for i, module in enumerate(self.net):
            x = module.linear(x)
            # x = rearrange(x, '(h w) c -> h w c', h=cumprods[0][i], w=cumprods[1][i])
            # x = F.conv2d(x, module.linear.weight.unsqueeze(-1).unsqueeze(-1), module.linear.bias)
            # x = repeat(x, 'h w c -> (h h2) (w w2) c', h2=cumprods[0][i+1], w2=cumprods[1][i+1]) + repeat(shift, 'h w c -> (h h2) (w w2) c', h2=cumprods[0][i+1], w2=cumprods[1][i+1])
            x = rearrange(x, '(h w) c -> h w c', h=cumprods[0][i], w=cumprods[1][i])
            x = repeat(x, 'h w c -> (h h2) (w w2) c', h2=self.dimensions[0][i+1], w2=self.dimensions[1][i+1])  # upsample h2xw2

            shift = rearrange(modulations[i], '(h w) c -> h w c', h=self.dimensions[0][i+1], w=self.dimensions[1][i+1])
            x = x + repeat(shift, 'h w c -> (h2 h) (w2 w) c', h2=cumprods[0][i], w2=cumprods[1][i]) # repeat h2 times along h axis, w2 times along w axis
            x = rearrange(x, 'h w c -> (h w) c')
            x = module.activation(x)
         
        out = self.last_layer(x)
        return out

    def forward(self, x, labels=None):
        return self.sdf(x)

    def sdf(self, x, lod=None):
        x_levels = self.decompose_coords(x) # [b, hw, n_levels, c]
        x_levels = self.normalize_coords(x_levels)
        x = x_levels[..., 0, :] # first level coords
        for i, module in enumerate(self.net):
            x = module.linear(x)
            if i < self.n_levels - 1:
                x = x + self.modulators[i](x_levels[..., i+1, :])
            else:
                x = x + self.modulators[self.n_levels-2](x_levels[..., self.n_levels-1, :]) # if number layers > number of modulations, reuse the last modulation
            x = module.activation(x)  # (batch_size, num_points, dim_hidden)

        out = self.last_layer(x)
        return out
