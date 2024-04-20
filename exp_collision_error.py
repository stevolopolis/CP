import numpy as np
import torch
import random
from tqdm import tqdm

from utils import *
from scorers import *
from models import *
from hash_visualizer import *


# Set random seed
random.seed(21)
torch.manual_seed(21)
np.random.seed(21)

# Path parameters 
experiment_name = "collion_error"

# Model parameters
MODEL = 'ngp_2d'

# Training parameters
n_imgs = 24
n_seeds = 3
n_samples = 50000
n = 1000
epoch = 25000
max_bandwidth = 500

# Animation parameters
nframes = 30


def train(base_path, data_path, img_idx, n_seeds, device="cuda"):
    print("generating samples...")
    sample = torch.tensor(np.linspace(0, 1, n_samples)).to(torch.float32).to(device)

    empirical_save_path = f"{base_path}/{img_idx}"
    create_subdirectories(empirical_save_path)

    dataLoader = ImageFile(data_path)
    H, W, C = dataLoader.get_data_shape()
    sample, signal = next(iter(dataLoader))
    sample = sample.to(device)
    signal = signal.to(device)

    # Generate specific hash vals
    for resolution in range(50, 500, 50):
        # Load default model configs
        configs = get_default_model_configs(MODEL)
        temp_c = configs.NET._replace(base_resolution=resolution)
        temp_c = configs.NET._replace(finest_resolution=resolution)
        configs = configs._replace(NET = temp_c)
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            # Get model
            model = get_model(MODEL, 2, 3, (H, W), configs, device=device)
            # Initialize model weights
            model.init_weights(ordered=False)
            # Load default model optimizers and schedulers
            optim, scheduler = get_default_model_opts(MODEL, model, epoch)

            model_loss, model_preds = trainer(sample, signal, model, optim, scheduler, epoch, nframes, device=device)
            
            # Animate model predictions
            animate_model_preds_2d([signal.cpu().numpy()]+model_preds, (H, W, C), nframes, f"{empirical_save_path}/preds_{img_idx}_{resolution}_{seed}.mp4")
            # Save model configs
            save_configs(configs, f"{empirical_save_path}/configs_{resolution}.json")
            # Save model loss
            save_vals([model_loss], f"{empirical_save_path}/loss_{img_idx}_{resolution}_{seed}.txt")
            # Save model weights
            torch.save(model.state_dict(), f"{empirical_save_path}/weights_{img_idx}_{resolution}_{seed}.pth")
            print(f"model weights saved at {empirical_save_path}/weights_{img_idx}_{resolution}_{seed}.pth")


if __name__ == "__main__":
    DATA_PATH = f"../data/kodak"
    BASE_PATH = f"vis/{experiment_name}/{MODEL}_piecewise"
    EMPIRICAL_PATH = f"{BASE_PATH}/empirical"
    FIGURE_PATH = f"{BASE_PATH}/figures"
    create_subdirectories(EMPIRICAL_PATH)
    create_subdirectories(FIGURE_PATH)

    for img_idx in range(1, n_imgs+1):
        train(EMPIRICAL_PATH, f"{DATA_PATH}/kodim{str(img_idx).zfill(2)}.png", img_idx, n_seeds)
