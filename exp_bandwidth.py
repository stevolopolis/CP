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
MODEL_NAME = f"{MODEL}_piecewise"

# Training parameters
n_trials = 10
n_seeds = 3
n_samples = 50000
n = 1000
epoch = 25000
max_bandwidth = 500
bandwidth_decrement = 50

# Animation parameters
nframes = 30


def train(base_path, trial, n_seeds, device="cuda"):
    print("generating samples...")
    sample = torch.tensor(np.linspace(0, 1, n_samples)).to(torch.float32).to(device)

    empirical_save_path = f"{base_path}/{trial}"
    create_subdirectories(empirical_save_path)

    # Generate full bandwidth signal
    #full_band_signal, coeffs, freqs, phases = generate_fourier_signal(sample, max_bandwidth)
    for bandwidth in range(max_bandwidth, 1, -bandwidth_decrement):
        #coeffs[bandwidth:] = 0
        #signal, _, _, _ = generate_fourier_signal(sample, bandwidth, coeffs=coeffs, freqs=freqs, phases=phases)
        signal, _, _, _ = generate_piecewise_signal(sample, bandwidth)

        # Save data & configs
        save_data(sample.cpu().numpy(), signal.cpu().numpy(), f"{empirical_save_path}/data_{bandwidth}.npy")

        # Generate specific hash vals
        for seed in range(n_seeds):
            # Load default model configs
            configs = get_default_model_configs(MODEL)
            # Get model
            model = get_model(MODEL, 1, 1, 1, configs, device=device)
            # Initialize model weights
            model.init_weights(ordered=True)
            # Load default model optimizers and schedulers
            optim, scheduler = get_default_model_opts(MODEL, model, epoch)
            # Model training
            model_loss, model_preds = trainer(sample.unsqueeze(1), signal.unsqueeze(1), model, optim, scheduler, epoch, nframes, device=device)
            
            # Animate model predictions
            animate_model_preds(sample, signal, model_preds, nframes, f"{empirical_save_path}/preds_{bandwidth}_{seed}.mp4")
            # Save model configs
            save_configs(configs, f"{empirical_save_path}/configs.json")
            # Save model loss
            save_vals([model_loss], f"{empirical_save_path}/loss_{bandwidth}_{seed}.txt")
            # Save model weights
            torch.save(model.state_dict(), f"{empirical_save_path}/weights_{bandwidth}_{seed}.pth")
            print(f"model weights saved at {empirical_save_path}/weights_{bandwidth}_{seed}.pth")


def plot(empirical_path, figure_path, hashing=True):
    total = {"pred_flips": [], "pred_segs": [], "signal_flips": [], "signal_segs": [], "bandwidths": [], "losses": []}
    if hashing:
        total["hash_flips"] = []
        total["mlp_flips"] = []
        total["mlp_segs"] = []

    for trial in range(n_trials):
        trial_dict = {"pred_flips": [], "pred_segs": [], "signal_flips": [], "signal_segs": [], "bandwidths": [], "losses": []}
        if hashing:
            trial_dict["hash_flips"] = []
            trial_dict["mlp_flips"] = []
            trial_dict["mlp_segs"] = []
        config_path = f"{empirical_path}/{trial}/configs.json"
        for bandwidth in range(max_bandwidth, 1, -bandwidth_decrement):
            data_path = f"{empirical_path}/{trial}/data_{bandwidth}.npy"
            for seed in range(n_seeds):
                model_path = f"{empirical_path}/{trial}/weights_{bandwidth}_{seed}.pth"
                loss_path = f"{empirical_path}/{trial}/loss_{bandwidth}_{seed}.txt"
                hash_val_save_path = f"{figure_path}/{trial}/visualization_{bandwidth}_{seed}.png"
                create_subdirectories(hash_val_save_path, is_file=True)

                # load trained_model
                model, x, y, configs = load_model_and_configs(data_path, config_path, model_path, MODEL)
                # plot hash vals
                visualization_1d(model, x, y, one_to_one=True, save_path=hash_val_save_path)
                # load loss
                model_loss = load_vals(loss_path)[0]
                if hashing:
                    # get ngp model outputs
                    hash_vals, mlp_vals, mlp_domain = get_model_hash_net_output(model, x)
                    # get number of hash flips
                    hash_flips, _ = detect_segments(model.hash_table.embeddings[0].weight)
                    # get number of mlp flips
                    mlp_flips, mlp_segs = detect_segments(mlp_vals)
                    trial_dict["hash_flips"].append(hash_flips)
                    trial_dict["mlp_flips"].append(mlp_flips)
                    trial_dict["mlp_segs"].append(mlp_segs)

                # get number of pred flips
                pred_flips, pred_segs = detect_segments(model(x))
                # get number of signal flips
                signal_flips, signal_segs = detect_segments(y)
                trial_dict["pred_flips"].append(pred_flips)
                trial_dict["pred_segs"].append(pred_segs)
                trial_dict["signal_flips"].append(signal_flips)
                trial_dict["signal_segs"].append(signal_segs)
                trial_dict["bandwidths"].append(bandwidth)
                trial_dict["losses"].append(model_loss)

        for key in total.keys():
            if key == "bandwidths":
                continue
            plot_scatter(trial_dict["bandwidths"],
                         trial_dict[key],
                         "bandwidths",
                         key,
                         f"{key} vs bandwidths",
                         f"{figure_path}/{key}_vs_bandwidths_{trial}.png")

        total["pred_flips"].extend(trial_dict["pred_flips"])
        total["pred_segs"].extend(trial_dict["pred_segs"])
        total["signal_flips"].extend(trial_dict["signal_flips"])
        total["signal_segs"].extend(trial_dict["signal_segs"])
        total["bandwidths"].extend(trial_dict["bandwidths"])
        total["losses"].extend(trial_dict["losses"])
        if hashing:
            total["hash_flips"].extend(trial_dict["hash_flips"])
            total["mlp_flips"].extend(trial_dict["mlp_flips"])
            total["mlp_segs"].extend(trial_dict["mlp_segs"])

    for key in total.keys():
        if key == "bandwidths":
            continue
        plot_scatter(total["bandwidths"],
                    total[key],
                    "bandwidths",
                    key,
                    f"{key} vs bandwidths",
                    f"{figure_path}/{key}_vs_bandwidths.png")


def detect_segments(vals: torch.tensor):
    """Detect the number of linear segments and change of directions in an array of values"""
    eps = 1e-6

    slopes = vals[1:] - vals[:-1]
    slopes = smooth_signal(slopes)
    segs = torch.abs(slopes[1:] - slopes[:-1]) > eps
    flips = torch.logical_xor((slopes[1:] < 0), (slopes[:-1] < 0))
    
    n_segs = segs.sum().item() + 1
    n_flips = flips.sum().item()

    return n_flips, n_segs


def smooth_signal(vals):
    """Smooth a signal via a sliding window that assigns a new value with the mode of the window."""
    window_size = 10
    smoothed_vals = torch.zeros_like(vals)
    for i in range(len(vals)):
        start = max(0, i - window_size)
        end = min(len(vals), i + window_size)
        smoothed_vals[i] = torch.mode(vals[start:end], dim=0).values

    return smoothed_vals


def plot_scatter(x, y, xlabel, ylabel, title, save_path):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"Scatter plot saved at {save_path}")


if __name__ == "__main__":
    BASE_PATH = f"vis/bandwidth/{MODEL_NAME}"
    EMPIRICAL_PATH = f"{BASE_PATH}/empirical"
    FIGURE_PATH = f"{BASE_PATH}/figures"
    create_subdirectories(EMPIRICAL_PATH)
    create_subdirectories(FIGURE_PATH)

    # train
    #for trial in range(n_trials):
    #    train(EMPIRICAL_PATH, trial, n_seeds=n_seeds)

    # Plot
    plot(EMPIRICAL_PATH, FIGURE_PATH, hashing=MODEL=="ngp")

    