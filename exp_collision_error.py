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
experiment_name = "bandwidth"

# Model parameters
MODEL = 'ngp_2d'

# Training parameters
n_trials = 10
n_seeds = 3
n_samples = 50000
n = 1000
epoch = 2500
max_bandwidth = 500

# Animation parameters
nframes = 30


def train(base_path, data_path, trial, n_seeds):
    print("generating samples...")
    sample = torch.tensor(np.linspace(0, 1, n_samples)).to(torch.float32).to("cuda")

    empirical_save_path = f"{base_path}/{trial}"
    create_subdirectories(empirical_save_path)

    for img_idx in range(0, 25):
        dataLoader = ImageFile(f"{data_path}/img_{str.zfill(img_idx, 2)}.png")
        sample, signal = iter(dataLoader)[0]

        # Generate specific hash vals
        for seed in range(n_seeds):
            # Model training
            model, configs, model_loss, model_preds = trainer(MODEL, sample, signal, epoch, nframes, device="cuda")
            
            # Animate model predictions
            animate_model_preds_2d(sample, signal, model_preds, nframes, f"{empirical_save_path}/preds_{img_idx}_{seed}.mp4")
            # Save model configs
            save_configs(configs, f"{empirical_save_path}/configs.json")
            # Save model loss
            save_vals([model_loss], f"{empirical_save_path}/loss_{img_idx}_{seed}.txt")
            # Save model weights
            torch.save(model.state_dict(), f"{empirical_save_path}/weights_{img_idx}_{seed}.pth")
            print(f"model weights saved at {empirical_save_path}/weights_{img_idx}_{seed}.pth")
            input()
