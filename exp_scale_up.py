import numpy as np
import torch
import random
import math
from tqdm import tqdm

from utils import *
from scorers import *
from models import *


# Set random seed
random.seed(21)
torch.manual_seed(21)
np.random.seed(21)
 
# Path parameters 
experiment_name = "scaling"

# Model parameters
MODEL = 'ngp'

# Training parameters
n_trials = 5
n_samples = 50000
n = 1000
epoch = 25000
device = "cuda:2"

# Animation parameters
nframes = 30

def train(trial, n_seeds):
    # Data setup
    sample = torch.tensor(np.linspace(0, 1, n_samples)).to(torch.float32).to(device)

    n_max_pieces = 1000
    n_segs = 2

    analytical_save_path = f"vis/{experiment_name}/analytical/{trial}"
    empirical_save_path = f"vis/{experiment_name}/empirical/{trial}"
    create_subdirectories(analytical_save_path)
    create_subdirectories(empirical_save_path)

    # Generate signal
    segmented_n_pieces = [random.randint(1, n_max_pieces) for _ in range(n_segs)]
    signal, _, _, _ = generate_piecewise_signal(sample, segmented_n_pieces, seed=trial, device=device)

    # Save data & configs
    save_data(sample.cpu().numpy(), signal.cpu().numpy(), f"{empirical_save_path}/data.npy")
    save_vals(segmented_n_pieces, f"{empirical_save_path}/n_pieces.txt")
    
    for seed in range(n_seeds):
        # Generate specific hash vals
        mid_hashes = [round(0.1 * i, 1) for i in range(1, 10)]
        min_hash, max_hash = 0.0, 1.0
        for mid_hash in mid_hashes:
            new_hash_vals1 = torch.linspace(min_hash, mid_hash, n_samples // 2).unsqueeze(1).to(device)
            new_hash_vals2 = torch.linspace(mid_hash, max_hash, n_samples // 2).unsqueeze(1).to(device)
            new_hash_vals = torch.cat((new_hash_vals1, new_hash_vals2), dim=0)

            # Model training
            model, configs, model_loss, model_preds = trainer(MODEL, sample, signal, epoch, nframes, hash_vals=new_hash_vals, device=device)
            # Save model loss
            save_vals([model_loss], f"{empirical_save_path}/loss_{mid_hash}_{seed}.txt")
            # Animate model predictions
            animate_model_preds(sample, signal, model_preds, nframes, f"{empirical_save_path}/preds_{mid_hash}_{seed}.mp4")
            # Save model weights
            torch.save(model.state_dict(), f"{empirical_save_path}/weights_{mid_hash}_{seed}.pth")
            print(f"model weights saved at {empirical_save_path}/weights_{mid_hash}_{seed}.pth")

    # Save model configs
    save_configs(configs, f"{empirical_save_path}/configs.json")


def plot(trial, n_seeds):
    """Plot loss against segment ratio.
    Loss should be lowest at the segment ratio equal to n_pieces ratio"""
    figure_save_path = f"vis/{experiment_name}/figures/{trial}"
    create_subdirectories(figure_save_path)

    # Load n_pieces
    n_pieces = load_vals(f"vis/{experiment_name}/empirical/{trial}/n_pieces.txt")
    # Initialize hash_vals
    mid_hashes = [round(0.1 * i, 1) for i in range(1, 10)]
    # Load losses
    losses = []
    losses_err = []
    for mid_hash in mid_hashes:
        temp_loss = []
        for seed in range(n_seeds):
            temp_loss.append(load_vals(f"vis/{experiment_name}/empirical/{trial}/loss_{mid_hash}_{seed}.txt")[0])
        loss = sum(temp_loss) / n_seeds
        err = np.std(temp_loss)
        losses.append(loss)
        losses_err.append(err)

    # Plot
    n_pieces_ratio = n_pieces[0] / sum(n_pieces)
    plt.plot(mid_hashes, losses, label='Loss')
    plt.errorbar(mid_hashes, losses, yerr=losses_err, fmt='o')
    plt.axvline(n_pieces_ratio, color='red', label='n_pieces ratio')
    plt.xlabel("Bash segment ratio")
    plt.title("Loss against segment ratio")
    plt.legend()
    plt.savefig(f"{figure_save_path}/loss_vs_segment_ratio.png")
    plt.close()

if __name__ == "__main__":
    for trial in range(1, 5):
        train(trial, 3)
        plot(trial, 3)