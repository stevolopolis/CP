import numpy as np
import torch
import random
from tqdm import tqdm
import wandb
import json

from utils import *
from scorers import *
from models import *
from hash_visualizer import *


# Set random seed
random.seed(21)
torch.manual_seed(21)
np.random.seed(21)

# Path parameters 
experiment_name = "collision_error"

# Model parameters
ordered=False
MODEL = 'ngp_2d'
if ordered:
    MODEL_NAME = f"{MODEL}_ordered"
else:
    MODEL_NAME = f"{MODEL}"

# Training parameters
n_imgs = 24
n_seeds = 3
n_samples = 50000
n = 1000
epoch = 3000
max_bandwidth = 500

# Animation parameters
nframes = 30


def train(base_path, data_path, img_idx, n_seeds, device="cuda"):
    print("generating samples...")
    sample = torch.tensor(np.linspace(0, 1, n_samples)).to(torch.float32).to(device)

    dataLoader = ImageFile(data_path, coord_mode=0)
    H, W, C = dataLoader.get_data_shape()
    sample, signal = next(iter(dataLoader))
    sample = sample.to(device)
    signal = signal.to(device)

    # Generate specific hash vals
    for resolution in range(50, 500, 50):
        # Load default model configs
        configs = get_default_model_configs(MODEL)
        if MODEL == "ngp_2d" or MODEL == "ngp_2d_one2one":
            temp_c = configs.NET._replace(base_resolution=resolution)
            temp_c = temp_c._replace(finest_resolution=resolution)
        elif MODEL == "ngp_multilevel_2d":
            temp_c = configs.NET._replace(finest_resolution=resolution)
        else:
            temp_c = configs.NET

        configs = configs._replace(NET = temp_c)
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            # Get model
            model = get_model(MODEL, 2, 3, (H, W), configs, device=device)
            # Initialize model weights
            model.init_weights(ordered=ordered)
            # Load default model optimizers and schedulers
            optim, scheduler = get_default_model_opts(MODEL, model, epoch)

            n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
            print(f"No. of parameters: {n_params}")

            wandb.init(
                project="collision-error-v2",
                entity="utmist-parsimony",
                config=configs.NET._asdict(),
                group=f"{MODEL_NAME}",
                name=f"{img_idx}_{resolution}_{seed}",
            )
            wandb.run.summary['n_params'] = n_params

            model_loss, model_preds = trainer(sample, signal, model, optim, scheduler, epoch, nframes, use_wandb=True)
            
            # Animate model predictions
            animate_model_preds_2d([signal.cpu().numpy()]+model_preds, (H, W, C), nframes, f"{base_path}/preds_{img_idx}_{resolution}_{seed}.mp4")
            # Save model configs
            save_configs(configs, f"{base_path}/configs_{resolution}.json")
            # Save model loss
            save_vals([model_loss], f"{base_path}/loss_{img_idx}_{resolution}_{seed}.txt")
            # Save model weights
            torch.save(model.state_dict(), f"{base_path}/weights_{img_idx}_{resolution}_{seed}.pth")
            print(f"model weights saved at {base_path}/weights_{img_idx}_{resolution}_{seed}.pth")


if __name__ == "__main__":
    DEVICE = "cuda:0"
    DATA_PATH = f"../data/kodak"
    BASE_PATH = f"vis/{experiment_name}/{MODEL_NAME}"
    EMPIRICAL_PATH = f"{BASE_PATH}/empirical"
    FIGURE_PATH = f"{BASE_PATH}/figures"
    create_subdirectories(EMPIRICAL_PATH)
    create_subdirectories(FIGURE_PATH)

    for img_idx in range(1, 4):
        empirical_save_path = f"{BASE_PATH}/{img_idx}"
        create_subdirectories(empirical_save_path)
        
        train(empirical_save_path, f"{DATA_PATH}/kodim{str(img_idx).zfill(2)}.png", img_idx, n_seeds, device=DEVICE)
