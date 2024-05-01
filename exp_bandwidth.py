import numpy as np
import torch
import random
import wandb
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
signal_type = "fourier"
MODEL = 'ngp_feature2d'
# special model config options: 
# - "*ordered*" for ordered hash table
# - "*frozen*" for frozen hash table
# - "*flipped*" for flipped hash table
MODEL_NAME = f"{MODEL}_trainable"

# Training parameters
n_trials = 10
n_seeds = 3
n_samples = 50000
n = 1000
epoch = 10000
max_bandwidth = 100 if signal_type == "fourier" else 500
bandwidth_decrement = 10 if signal_type == "fourier" else 50

# Animation parameters
nframes = 30


def train(base_path, trial, n_seeds, signal_type="fourier", device="cuda", use_wandb=False):
    torch.manual_seed(trial)
    print("generating samples...")
    sample = torch.tensor(np.linspace(0, 1, n_samples)).to(torch.float32).to(device)

    # Generate full bandwidth signal
    if signal_type == "fourier":
        full_band_signal, coeffs, freqs, phases = generate_fourier_signal(sample, max_bandwidth, device=device)
    elif signal_type == "piecewise":
        full_band_signal, knot_idx, _, _ = generate_piecewise_signal(sample, max_bandwidth, seed=trial, device=device)
    else:
        raise ValueError("Signal type not recognized")
    
    for bandwidth in range(max_bandwidth, 1, -bandwidth_decrement):
        if signal_type == "fourier":
            signal = decrement_fourier_signal(sample, coeffs, freqs, phases, bandwidth, device=device)
        elif signal_type == "piecewise":
            signal = decrement_piecewise_signal(sample, full_band_signal, knot_idx, bandwidth)
            # signal, _, _, _ = generate_piecewise_signal(sample, bandwidth, seed=trial)

        # Save data & configs
        save_data(sample.cpu().numpy(), signal.cpu().numpy(), f"{base_path}/data_{bandwidth}.npy")

        # Generate specific hash vals
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            # Load default model configs
            configs = get_default_model_configs(MODEL)
            # Get model
            model = get_model(MODEL, 1, 1, [1], configs, device=device)
            # Initialize model weights
            if "ordered" in MODEL_NAME:
                model.init_weights(ordered=True)
            elif "flipped" in MODEL_NAME:
                model.init_weights(flipped=True)
            else:
                model.init_weights()

            if "frozen" in MODEL_NAME:
                model.freeze_hash_table()
                print("hash table weights frozen")

            # Load default model optimizers and schedulers
            optim, scheduler = get_default_model_opts(MODEL, model, epoch)

            if use_wandb:
                wandb.init(
                    project="1d-input-2d-feature",
                    entity="utmist-parsimony",
                    config=configs.NET._asdict(),
                    group=f"{MODEL_NAME}",
                    name=f"{trial}_{bandwidth}_{seed}",
                )

            # Model training
            model_loss, model_preds = trainer(sample.unsqueeze(1), signal.unsqueeze(1), model, optim, scheduler, epoch, nframes, use_wandb=use_wandb)
            
            # Animate model predictions
            animate_model_preds(sample, signal, model_preds, nframes, f"{base_path}/preds_{bandwidth}_{seed}.mp4")
            # Save model configs
            save_configs(configs, f"{base_path}/configs.json")
            # Save model loss
            save_vals([model_loss], f"{base_path}/loss_{bandwidth}_{seed}.txt")
            # Save model weights
            torch.save(model.state_dict(), f"{base_path}/weights_{bandwidth}_{seed}.pth")
            print(f"model weights saved at {base_path}/weights_{bandwidth}_{seed}.pth")


if __name__ == "__main__":
    DEVICE = "cuda:1"
    BASE_PATH = f"vis/{experiment_name}/{MODEL_NAME}"
    EMPIRICAL_PATH = f"{BASE_PATH}/empirical"
    FIGURE_PATH = f"{BASE_PATH}/figures"
    create_subdirectories(EMPIRICAL_PATH)
    create_subdirectories(FIGURE_PATH)

    # train
    for trial in range(n_trials):
        empirical_save_path = f"{EMPIRICAL_PATH}/{trial}"
        create_subdirectories(empirical_save_path)
        train(empirical_save_path, trial, n_seeds=n_seeds, signal_type=signal_type, device=DEVICE, use_wandb=True)

    # Plot
    plot_segment_summary(EMPIRICAL_PATH, FIGURE_PATH, hashing=MODEL=="ngp", device=DEVICE)

    