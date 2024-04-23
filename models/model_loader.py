import torch

from collections import namedtuple

from models.mlp import MLP
from models.linear import LinearModel
from models.ngp import NGP


hidden_dim = 64
n_layers = 5
dim_in = 1
dim_out = 1
img_size = ()


def get_default_model_configs(model_type):
    if model_type == 'relu':
        Config = namedtuple("config", ["NET"])
        NetworkConfig = namedtuple("NET", ["num_layers", "dim_hidden", "use_bias"])
        c_net = NetworkConfig(num_layers=n_layers, dim_hidden=hidden_dim, use_bias=True)
        c = Config(NET=c_net)
    elif model_type == 'linear':
        c = None
    elif model_type == "ngp":
        Config = namedtuple("config", ["NET"])
        NetworkConfig = namedtuple("NET", ["dim_hidden", "n_levels", "feature_dim", "log2_n_features", "base_resolution", "finest_resolution", "num_layers"])
        c_net = NetworkConfig(dim_hidden=hidden_dim, n_levels=1, feature_dim=1, log2_n_features=5, base_resolution=25, finest_resolution=25, num_layers=n_layers)
        c = Config(NET=c_net)
    elif model_type == "ngp_2d":
        Config = namedtuple("config", ["NET"])
        NetworkConfig = namedtuple("NET", ["dim_hidden", "n_levels", "feature_dim", "log2_n_features", "base_resolution", "finest_resolution", "num_layers"])
        c_net = NetworkConfig(dim_hidden=256, n_levels=1, feature_dim=2, log2_n_features=14, base_resolution=50, finest_resolution=50, num_layers=n_layers)
        c = Config(NET=c_net)
    elif model_type == "ngp_2d_one2one":
        Config = namedtuple("config", ["NET"])
        NetworkConfig = namedtuple("NET", ["dim_hidden", "n_levels", "feature_dim", "log2_n_features", "base_resolution", "finest_resolution", "num_layers"])
        c_net = NetworkConfig(dim_hidden=64, n_levels=1, feature_dim=2, log2_n_features=18, base_resolution=50, finest_resolution=50, num_layers=n_layers)
        c = Config(NET=c_net)
    elif model_type == "ngp_multilevel_2d":
        Config = namedtuple("config", ["NET"])
        NetworkConfig = namedtuple("NET", ["dim_hidden", "n_levels", "feature_dim", "log2_n_features", "base_resolution", "finest_resolution", "num_layers"])
        c_net = NetworkConfig(dim_hidden=64, n_levels=8, feature_dim=2, log2_n_features=12, base_resolution=16, finest_resolution=256, num_layers=3)
        c = Config(NET=c_net)
    else:
        raise ValueError("Model type not recognized")

    return c


# Load default models
def get_default_model_opts(model_type, model, epoch=5000):
    if model_type == 'relu':
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-4)
    elif model_type == 'linear':
        optim = torch.optim.Adam(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-2)
    elif model_type == "ngp":
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-4)
    elif model_type == "ngp_2d" or model_type == "ngp_2d_one2one":
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-4)
    elif model_type == "ngp_multilevel_2d":
        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-4)
    else:
        raise ValueError("Model type not recognized")
    
    return optim, scheduler


def get_model(model_type, dim_in, dim_out, data_size, configs, device="cuda"):
    if model_type == 'relu':
        model = MLP(dim_in, dim_out, hidden_dim, configs).to(device)
    elif model_type == 'linear':
        model = LinearModel().to(device)
    elif model_type in ["ngp", "ngp_2d", "ngp_multilevel_2d", "ngp_2d_one2one"]:
        model = NGP(dim_in, dim_out, data_size, configs).to(device)
    else:
        raise ValueError("Model type not recognized")

    return model