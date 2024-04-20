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
MODEL = 'ngp'

# Training parameters
n_trials = 10
n_seeds = 3
n_samples = 50000
n = 1000
epoch = 25000
max_bandwidth = 100

# Animation parameters
nframes = 30


def train(trial, n_seeds):
    print("generating samples...")
    sample = torch.tensor(np.linspace(0, 1, n_samples)).to(torch.float32).to("cuda")

    analytical_save_path = f"vis/{experiment_name}/{MODEL}/analytical/{trial}"
    empirical_save_path = f"vis/{experiment_name}/{MODEL}/empirical/{trial}"
    create_subdirectories(analytical_save_path)
    create_subdirectories(empirical_save_path)

    # Generate full bandwidth signal
    full_band_signal, coeffs, freqs, phases = generate_fourier_signal(sample, max_bandwidth)
    for bandwidth in range(max_bandwidth, 1, -10):
        coeffs[bandwidth:] = 0
        signal, _, _, _ = generate_fourier_signal(sample, bandwidth, coeffs=coeffs, freqs=freqs, phases=phases)

        # Save data & configs
        save_data(sample.cpu().numpy(), signal.cpu().numpy(), f"{empirical_save_path}/data_{bandwidth}.npy")

        # Generate specific hash vals
        for seed in range(n_seeds):
            # Model training
            model, configs, model_loss, model_preds = trainer(MODEL, sample, signal, epoch, nframes, device="cuda")
            
            # Animate model predictions
            animate_model_preds(sample, signal, model_preds, nframes, f"{empirical_save_path}/preds_{bandwidth}_{seed}.mp4")
            # Save model configs
            save_configs(configs, f"{empirical_save_path}/configs.json")
            # Save model loss
            save_vals([model_loss], f"{empirical_save_path}/loss_{bandwidth}_{seed}.txt")
            # Save model weights
            torch.save(model.state_dict(), f"{empirical_save_path}/weights_{bandwidth}_{seed}.pth")
            print(f"model weights saved at {empirical_save_path}/weights_{bandwidth}_{seed}.pth")


def detect_flipping(vals: torch.tensor):
    """Detect the number of domain flipping incurred by the hash function of the model."""
    vals_diff = vals[1:] - vals[:-1]
    flips = torch.logical_xor((vals_diff[1:] < 0), (vals_diff[:-1] < 0))
    return flips.sum().item()


def plot_scatter(y, x, xlabel, ylabel, title, save_path):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"Scatter plot saved at {save_path}")
    


if __name__ == "__main__":
    # for trial in range(n_trials):
    #     train(trial, n_seeds=n_seeds)

    # Plot
    BASE_PATH = f"vis/bandwidth/{MODEL}"
    EMPIRICAL_PATH = f"{BASE_PATH}/empirical"
    FIGURE_PATH = f"{BASE_PATH}/figures"
    create_subdirectories(FIGURE_PATH)

    hash_flips_total = []
    mlp_flips_total = []
    pred_flips_total = []
    signal_flips_total = []
    bandwidths_total = []
    losses_total = []
    for trial in range(n_trials):
        hash_flips_trial = []
        mlp_flips_trial = []
        pred_flips_trial = []
        signal_flips_trial = []
        bandwidths_trial = []
        losses_trial = []
        config_path = f"{EMPIRICAL_PATH}/{trial}/configs.json"
        for bandwidth in range(max_bandwidth, 1, -10):
            data_path = f"{EMPIRICAL_PATH}/{trial}/data_{bandwidth}.npy"
            for seed in range(n_seeds):
                model_path = f"{EMPIRICAL_PATH}/{trial}/weights_{bandwidth}_{seed}.pth"
                loss_path = f"{EMPIRICAL_PATH}/{trial}/loss_{bandwidth}_{seed}.txt"
                hash_val_save_path = f"{FIGURE_PATH}/{trial}/visualization_{bandwidth}_{seed}.png"
                create_subdirectories(hash_val_save_path, is_file=True)

                # load trained_model
                model, x, y, configs = load_model_and_configs(data_path, config_path, model_path, MODEL)
                # plot hash vals
                # visualization_1d(model, x, y, one_to_one=True, save_path=hash_val_save_path)
                # load loss
                model_loss = load_vals(loss_path)[0]
                # get ngp model outputs
                hash_vals, mlp_vals, mlp_domain = get_model_hash_net_output(model, x)
                # get number of hash flips
                hash_flips = detect_flipping(model.hash_table.embeddings[0].weight)
                # get number of mlp flips
                mlp_flips = detect_flipping(mlp_vals)
                # get number of pred flips
                pred_flips = detect_flipping(model(x))
                # get number of signal flips
                signal_flips = detect_flipping(y)
                hash_flips_trial.append(hash_flips)
                mlp_flips_trial.append(mlp_flips)
                pred_flips_trial.append(pred_flips)
                signal_flips_trial.append(signal_flips)
                bandwidths_trial.append(bandwidth)
                losses_trial.append(model_loss)

        plot_scatter(hash_flips_trial,
                    bandwidths_trial,
                    "Bandwidth",
                    "Hash Flips",
                    "Hash Flips vs Bandwidth",
                    f"{FIGURE_PATH}/hash_flips_vs_bandwidth_{trial}.png")
        plot_scatter(mlp_flips_trial,
                    bandwidths_trial,
                    "Bandwidth",
                    "MLP Flips",
                    "MLP Flips vs Bandwidth",
                    f"{FIGURE_PATH}/mlp_flips_vs_bandwidth_{trial}.png")
        plot_scatter(pred_flips_trial,
                    bandwidths_trial,
                    "Bandwidth",
                    "Pred Flips",
                    "Pred Flips vs Bandwidth",
                    f"{FIGURE_PATH}/pred_flips_vs_bandwidth_{trial}.png")
        plot_scatter(signal_flips_trial,
                    bandwidths_trial,
                    "Bandwidth",
                    "Signal Flips",
                    "Signal Flips vs Bandwidth",
                    f"{FIGURE_PATH}/signal_flips_vs_bandwidth_{trial}.png")
        plot_scatter(losses_trial,
                    bandwidths_trial,
                    "Bandwidth",
                    "Loss",
                    "Loss vs Bandwidth",
                    f"{FIGURE_PATH}/loss_vs_bandwidth_{trial}.png")

        hash_flips_total += hash_flips_trial
        mlp_flips_total += mlp_flips_trial
        pred_flips_total += pred_flips_trial
        signal_flips_total += signal_flips_trial
        bandwidths_total += bandwidths_trial
        losses_total += losses_trial

    plot_scatter(hash_flips_total,
                bandwidths_total,
                "Bandwidth",
                "Hash Flips",
                "Hash Flips vs Bandwidth",
                f"{FIGURE_PATH}/hash_flips_vs_bandwidth.png")
    plot_scatter(mlp_flips_total,
                bandwidths_total,
                "Bandwidth",
                "MLP Flips",
                "MLP Flips vs Bandwidth",
                f"{FIGURE_PATH}/mlp_flips_vs_bandwidth.png")
    plot_scatter(pred_flips_total,
                bandwidths_total,
                "Bandwidth",
                "Pred Flips",
                "Pred Flips vs Bandwidth",
                f"{FIGURE_PATH}/pred_flips_vs_bandwidth.png")
    plot_scatter(signal_flips_total,
                bandwidths_total,
                "Bandwidth",
                "Signal Flips",
                "Signal Flips vs Bandwidth",
                f"{FIGURE_PATH}/signal_flips_vs_bandwidth.png")
    plot_scatter(losses_total,
                bandwidths_total,
                "Bandwidth",
                "Loss",
                "Loss vs Bandwidth",
                f"{FIGURE_PATH}/loss_vs_bandwidth.png")