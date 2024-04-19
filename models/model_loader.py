import torch

from collections import namedtuple

from models.mlp import MLP
from models.linear import LinearModel
from models.ngp import NGP


hidden_dim = 64
n_layers = 5
dim_in = 1
dim_out = 1


# Load default models
def load_default_models(model_type, epoch=5000, configs=None, device="cuda"):
    if model_type == 'relu':
        if configs is None:
            Config = namedtuple("config", ["NET"])
            NetworkConfig = namedtuple("NET", ["num_layers", "dim_hidden", "use_bias"])
            c_net = NetworkConfig(num_layers=n_layers, dim_hidden=hidden_dim, use_bias=True)
            c = Config(NET=c_net)
        else:
            c = configs
        model = MLP(dim_in, dim_out, c).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-4)
    elif model_type == 'linear':
        model = LinearModel().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-2)
    elif model_type == "ngp":
        if configs is None:
            Config = namedtuple("config", ["NET"])
            NetworkConfig = namedtuple("NET", ["dim_hidden", "n_levels", "feature_dim", "log2_n_features", "base_resolution", "finest_resolution", "num_layers"])
            c_net = NetworkConfig(dim_hidden=hidden_dim, n_levels=1, feature_dim=1, log2_n_features=5, base_resolution=5, finest_resolution=5, num_layers=n_layers)
            c = Config(NET=c_net)
        else:
            c = configs
        model = NGP(dim_in, dim_out, 1, c).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-4)
    else:
        raise ValueError("Model type not recognized")
    
    return model, optim, scheduler, c