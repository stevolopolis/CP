import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import wandb
import random

from utils import *
from models import *


def get_model_hash_net_output(model, x):
    if model.hash_table.n_levels > 1:
        raise NotImplementedError("Only single-level grid is supported for now.")
    
    hash_vals = model.hash_table(x)
    # 1d hash_vals
    if model.hash_table.n_features_per_level == 1:
        min_hash = torch.min(hash_vals).item()
        max_hash = torch.max(hash_vals).item()
        mlp_domain = torch.linspace(min_hash, max_hash, 1000).unsqueeze(1).to(x.device)
        
        mlp_vals = model.net(mlp_domain)
    # 2d hash_vals
    elif model.hash_table.n_features_per_level == 2:
        min_x_hash = torch.min(hash_vals[:, 0]).item()
        min_y_hash = torch.min(hash_vals[:, 1]).item()
        max_x_hash = torch.max(hash_vals[:, 0]).item()
        max_y_hash = torch.max(hash_vals[:, 1]).item()
        mlp_domain_x = torch.linspace(min_x_hash, max_x_hash, 1000).to(x.device)
        mlp_domain_y = torch.linspace(min_y_hash, max_y_hash, 1000).to(x.device)
        model_x = torch.stack(torch.meshgrid((mlp_domain_x, mlp_domain_y), indexing='ij'), dim=-1).view(-1, 2)
        mlp_domain = torch.meshgrid((mlp_domain_x, mlp_domain_y), indexing='ij')

        mlp_vals = model.net(model_x)
    else:
        raise NotImplementedError("Only 1d and 2d hash_vals are supported for now.")


    return hash_vals, mlp_vals, mlp_domain


def visualization_1d(model, x, y, one_to_one=True, save_path="visualization.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    preds = model(x)
    ax[0].plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label="target", color='royalblue')
    ax[0].plot(x.detach().cpu().numpy(), preds.detach().cpu().numpy(), label="prediction", color='orange')
    ax[0].legend()
    ax[0].set_title("Model prediction & Target signal")

    if one_to_one:
        hash_vals, mlp_vals, mlp_domain = get_model_hash_net_output(model, x)
        # visualize the hash function
        hash_vals = hash_vals.detach().cpu().numpy()
        ax[1].plot(hash_vals, x.detach().cpu().numpy(), label="hash_vals", color='cadetblue')
        ax[1].legend()
        ax[1].set_xlabel("MLP domain")
        ax[1].set_ylabel("Hash/Input domain")
        
        # visualize the mlp function
        mlp_vals = mlp_vals.detach().cpu().numpy()
        ax2 = ax[1].twinx()
        ax2.plot(mlp_domain.cpu().numpy(), mlp_vals, label="mlp_vals", color='salmon')
        ax2.set_ylabel("MLP range")
        ax2.legend()
        
        plt.title("Hash and MLP function visualization")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    else:
        raise NotImplementedError("Only one-to-one hash function is supported for now.")


def visualization_2d(model, x, y, one_to_one=True, threeD=True, save_path="visualization.png"):
    fig, ax = plt.subplots(figsize=(15, 5))

    preds = model(x)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label="target", color='royalblue')
    ax1.plot(x.detach().cpu().numpy(), preds.detach().cpu().numpy(), label="prediction", color='orange')
    ax1.legend()
    ax1.set_title("Model prediction & Target signal")

    if one_to_one:
        hash_vals, mlp_vals, (mlp_domain_x, mlp_domain_y) = get_model_hash_net_output(model, x)
        hash_post_mlp = model.net(hash_vals)        # hash_vals -> mlp -> hash_post_mlp
        hash_post_mlp = hash_post_mlp.detach().cpu().numpy()
        # visualize the hash function
        hash_vals = hash_vals.detach().cpu().numpy()
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(hash_vals[:, 0], hash_vals[:, 1], label="hash_vals", color='cadetblue')
        ax2.grid(True)
        ax2.set_title("Hash Values")
        
        # visualize the mlp function
        mlp_vals = mlp_vals.detach().cpu().numpy()
        mlp_vals = np.reshape(mlp_vals, mlp_domain_x.shape)
        mlp_domain_x = mlp_domain_x.cpu().numpy()
        mlp_domain_y = mlp_domain_y.cpu().numpy()
        if not threeD:
            # Plot contour of MLP prediction
            ax3 = fig.add_subplot(1, 3, 3)
            cntr = ax3.contourf(mlp_domain_x, mlp_domain_y, mlp_vals, cmap='winter')
            ax3.set_title('MLP Plot')
            ax3.grid(True)
        else:
            # Plot surface of MLP prediction
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            cntr = ax3.plot_surface(mlp_domain_x, mlp_domain_y, mlp_vals, cmap='winter')
            ax3.plot(hash_vals[:, 0], hash_vals[:, 1], hash_post_mlp.squeeze(1), label="hash_vals", color='red')
            ax3.set_title('MLP Plot')
            ax3.grid(True)
            ax3.legend()

        fig.colorbar(cntr, ax=ax3)
        
        plt.savefig(save_path)
        plt.close()
        print("Visualization saved at", save_path)
    else:
        raise NotImplementedError("Only one-to-one hash function is supported for now.")


def visualization_2d_animated(model, x, y, one_to_one=True, threeD=True, animate=False, save_path="visualization.png"):
    fig, ax = plt.subplots(figsize=(15, 5))

    preds = model(x)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label="target", color='royalblue')
    ax1.plot(x.detach().cpu().numpy(), preds.detach().cpu().numpy(), label="prediction", color='orange')
    ax1.legend()
    ax1.set_title("Model prediction & Target signal")

    if one_to_one:
        hash_vals, mlp_vals, (mlp_domain_x, mlp_domain_y) = get_model_hash_net_output(model, x)
        hash_post_mlp = model.net(hash_vals)        # hash_vals -> mlp -> hash_post_mlp
        hash_post_mlp = hash_post_mlp.detach().cpu().numpy()
    
        # visualize the mlp function
        mlp_vals = mlp_vals.detach().cpu().numpy()
        mlp_vals = np.reshape(mlp_vals, mlp_domain_x.shape)
        mlp_domain_x = mlp_domain_x.cpu().numpy()
        mlp_domain_y = mlp_domain_y.cpu().numpy()
        if not threeD:
            # Plot contour of MLP prediction
            ax3 = fig.add_subplot(1, 3, 3)
            cntr = ax3.contourf(mlp_domain_x, mlp_domain_y, mlp_vals, cmap='winter')
            ax3.set_title('MLP Plot')
            ax3.grid(True)
        else:
            # Plot surface of MLP prediction
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            cntr = ax3.plot_surface(mlp_domain_x, mlp_domain_y, mlp_vals, cmap='winter')
            ax3.plot(hash_vals[:, 0], hash_vals[:, 1], hash_post_mlp.squeeze(1), label="hash_vals", color='red')
            ax3.set_title('MLP Plot')
            ax3.grid(True)
            ax3.legend()

        fig.colorbar(cntr, ax=ax3)

        # visualize the hash function
        ax2 = fig.add_subplot(1, 3, 2)
        hash_vals = hash_vals.detach().cpu().numpy()

        if animate:
            line, = ax2.plot([], [], marker='o', color='cadetblue', linestyle='-')
            points_per_grid = len(x) // model.hash_table.finest_resolution
            sparse_hash_vals = hash_vals[::points_per_grid]

            def init():
                ax2.set_xlim(min(p[0] for p in sparse_hash_vals) - 1, max(p[0] for p in sparse_hash_vals) + 1)
                ax2.set_ylim(min(p[1] for p in sparse_hash_vals) - 1, max(p[1] for p in sparse_hash_vals) + 1)
                return line,
            def update(num, x, y, line):
                line.set_data(x[:num], y[:num])
                return line,

            ax2.grid(True)
            ax2.set_title("Hash Values")

            ani = FuncAnimation(fig, update, frames=len(sparse_hash_vals), fargs=[sparse_hash_vals[:, 0], sparse_hash_vals[:, 1], line], init_func=init, blit=True)
        else:
            ax2.plot(hash_vals[:, 0], hash_vals[:, 1], label="hash_vals", color='cadetblue')
            ax2.grid(True)
            ax2.set_title("Hash Values")
        
        if animate:
            print(f"saving animation (frames: {len(sparse_hash_vals)} @ 1fps)")
            writervideo = animation.FFMpegWriter(fps=1)
            # ani.save(f"{save_path}", writer=writervideo)
            ani.save(f"{save_path}")
            print(f"Animation saved at {save_path}")
            plt.show()
            plt.close()
        else:
            plt.savefig(save_path)
            plt.close()
            print("Visualization saved at", save_path)
    else:
        raise NotImplementedError("Only one-to-one hash function is supported for now.")


def visualization_2d_surface(model, x, save_path="visualization.png"):
    fig, ax = plt.subplots(figsize=(15, 15))

    hash_vals, mlp_vals, (mlp_domain_x, mlp_domain_y) = get_model_hash_net_output(model, x)
    mlp_vals = mlp_vals.detach().cpu().numpy()
    mlp_vals = np.reshape(mlp_vals, mlp_domain_x.shape)
    mlp_domain_x = mlp_domain_x.cpu().numpy()
    mlp_domain_y = mlp_domain_y.cpu().numpy()
    hash_post_mlp = model.net(hash_vals)        # hash_vals -> mlp -> hash_post_mlp
    hash_post_mlp = hash_post_mlp.detach().cpu().numpy()
    hash_vals = hash_vals.detach().cpu().numpy()

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    cntr = ax.plot_surface(mlp_domain_x, mlp_domain_y, mlp_vals, cmap='winter', antialiased=True, alpha=.5)
    ax.plot(hash_vals[:, 0], hash_vals[:, 1], hash_post_mlp.squeeze(1), label="hash_vals", color='red', linewidth=2)
    ax.set_title('MLP Plot')
    ax.grid(True)
    ax.legend()

    fig.colorbar(cntr, ax=ax)
    plt.savefig(save_path)
    plt.close()

    print("Visualization saved at", save_path)


def visualization_hash_and_turning(model, x, turning="noturning", save_path="visualization.png"):
    fig, ax = plt.subplots(2, 1, figsize=(4, 8))

    # visualize the hash function
    hash_vals = model.hash_table(x)
    hash_vals = hash_vals.detach().cpu().numpy()
    min_hash = np.min(hash_vals)
    max_hash = np.max(hash_vals)

    if turning == "noturning":
        new_hash_vals = torch.linspace(min_hash, max_hash, len(x)).unsqueeze(1).to(x.device)
    elif turning == "scale":
        mid_hash = (max_hash - min_hash) * 0.8 + min_hash
        new_hash_vals1 = torch.linspace(min_hash, mid_hash, len(x) // 2).unsqueeze(1).to(x.device)
        new_hash_vals2 = torch.linspace(mid_hash, max_hash, len(x) // 2).unsqueeze(1).to(x.device)
        new_hash_vals = torch.cat((new_hash_vals1, new_hash_vals2), dim=0)
    elif turning == "flip":
        new_hash_vals1 = torch.linspace(min_hash, max_hash, len(x) // 2).unsqueeze(1).to(x.device)
        new_hash_vals2 = torch.linspace(max_hash, min_hash, len(x) // 2).unsqueeze(1).to(x.device)
        new_hash_vals = torch.cat((new_hash_vals1, new_hash_vals2), dim=0)
    else:
        raise NotImplementedError("Only no turning, scale, and flip are supported for now.")
    
    preds = model.net(new_hash_vals)

    ax[0].plot(x.detach().cpu().numpy(), new_hash_vals.detach().cpu().numpy(), label="hash_vals", color='cadetblue')
    ax[0].set_xlabel("Input domain")
    ax[0].set_ylabel("MLP domain")
    ax[0].set_title("Hash function")
    ax[0].legend(loc='upper left')

    # visualize the mlp function
    if turning == "scale":
        mid_point = len(x) // 2
        mid_y = model.net(new_hash_vals[mid_point]).item()
        ax[1].scatter((0.5), (mid_y), s=100, color='r', alpha=0.5, label="New turning point")
    ax[1].plot(x.detach().cpu().numpy(), preds.detach().cpu().numpy(), label="prediction", color='salmon')
    ax[1].set_xlabel("Input domain")
    ax[1].set_title("Model prediction")
    ax[1].legend(loc='upper left')

    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()


def visualize_isolated_ngp_hashing(trainer, x, y, data_shape, log):
    """
    Visualize the effects of each level of NGP hashing.

    NGP hashes coordinates in a multiresolution many-to-one manner.
    To understand the contributions of each level of hashing 
    indivdually, we zero out all the hash values except for one level
    and feed this augmented set of hashed values to the MLP. 
    We visualize the reconstructed image with this augmented hash vals.
    """
    for level in range(trainer.model.hash_table.n_levels):
        # get the prediction without the highest resolution hashing
        hashed_input = trainer.model.hash_table(x.to(trainer.args.device))
        # zero out the highest resolution hashing
        zero_hashed_input = torch.zeros_like(hashed_input)
        zero_hashed_input[:, 2*level:2*(level+1)] = hashed_input[:, 2*level:2*(level+1)]
        hashed_input = zero_hashed_input
        #hashed_input[:, 2*(level+1):] = 0
        alt_pred = trainer.model(hashed_input, hash=False)
        alt_loss = ((alt_pred - y.to(trainer.args.device)) ** 2).mean()
        alt_psnr = -10*np.log10(alt_loss.item())
        # generate the reconstructed image
        alt_pred = alt_pred.clamp(-1, 1).detach().cpu().numpy()
        alt_pred = alt_pred.reshape(data_shape)
        alt_pred = alt_pred / 2 + 0.5
        alt_im = Image.fromarray((alt_pred*255).astype(np.uint8), mode='RGB')

        log["psnr_w/o_L%s+" % level] = alt_psnr
        log["recon_w/o_L%s+" % level] = wandb.Image(alt_im)


def load_model_and_configs(data_path, config_path, model_path, model_type, dim_in=1, dim_out=1, data_dim=[1], device="cuda"):
    x = None
    y = None
    configs = None

    if data_path is not None:
        x, y = load_data(data_path)
        x = torch.tensor(x).to(device).unsqueeze(1)
        y = torch.tensor(y).to(device).unsqueeze(1)

    if config_path is not None:
        configs = load_configs(config_path)
    # Load default model configs
    else:
        configs = get_default_model_configs(model_type)

    # Get model
    assert dim_in == len(data_dim), "Input dimension does not match the data dimension."
    model = get_model(model_type, dim_in, dim_out, data_dim, configs, device=device)
    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))  
    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model, x, y, configs


def plot_segment_summary(empirical_path, figure_path, hashing=True, device="cuda"):
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
                model, x, y, configs = load_model_and_configs(data_path, config_path, model_path, MODEL, dim_in=1, dim_out=1, data_dim=[1], device=device)
                # load loss
                model_loss = load_vals(loss_path)[0]
                if hashing:
                    # plot hash vals
                    visualization_1d(model, x, y, one_to_one=True, save_path=hash_val_save_path)
                    # get ngp model outputs
                    hash_vals, mlp_vals, mlp_domain = get_model_hash_net_output(model, x)
                    # get number of hash flips
                    hash_flips, _ = detect_segments(model.hash_table.embeddings[0].weight, exact=True)
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
        x, y, yerr = scatter_to_errbar(total["bandwidths"], total[key])
        plot_errbar(x, y, yerr, "bandwidths", key, f"{key} vs bandwidths", f"{figure_path}/{key}_vs_bandwidths.png")


def scatter_to_errbar(x, y):
    """Convert scatter plot data to error bar data"""
    x = np.array(x).astype(int)
    y = np.array(y)
    x_uniq = np.unique(x)
    y_mean = [np.mean(y[x == i]) for i in x_uniq]
    y_std = [np.std(y[x == i]) for i in x_uniq]

    return x_uniq, y_mean, y_std


def detect_segments(vals: torch.tensor, exact=False):
    """Detect the number of linear segments and change of directions in an array of values"""
    eps = 1e-6

    slopes = vals[1:] - vals[:-1]
    if not exact:
        slopes = smooth_signal(slopes)
    segs = torch.abs(slopes[1:] - slopes[:-1]) > eps
    flips = torch.logical_xor((slopes[1:] < 0), (slopes[:-1] < 0))
    
    n_segs = segs.sum().item() + 1
    n_flips = flips.sum().item()

    return n_flips, n_segs


def smooth_signal(vals):
    """Smooth a signal via a sliding window that assigns a new value with the mode of the window."""
    window_size = 10
    vals_unfolded = vals.unfold(0, window_size, 1)
    smoothed_vals = torch.mode(vals_unfolded, dim=-1).values

    return smoothed_vals


def plot_scatter(x, y, xlabel, ylabel, title, save_path):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"Scatter plot saved at {save_path}")


def plot_errbar(x, y, yerr, xlabel, ylabel, title, save_path):
    plt.errorbar(x, y, yerr=yerr, fmt='o', color='salmon')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved at {save_path}")



MEGAPIXELS = ["pluto", "tokyo", "mars"]
DATA_ID = 0
MODEL = "ngp_feature2d"

EXPERIMENT_NAME = "bandwidth"
MODEL_NAME = f"{MODEL}"
BASE_PATH = f"vis/{EXPERIMENT_NAME}/{MODEL_NAME}"
EMPIRICAL_PATH = f"{BASE_PATH}/empirical"
FIGURE_PATH = f"{BASE_PATH}/figures"

MODEL_PATH = f'vis/{MODEL}/{MODEL}_{DATA_ID}'
VIS_PATH = "vis/experiments"
MODEL_WEIGHTS_PATH = f"{MODEL_PATH}/{MODEL}_{DATA_ID}.pth"
HASH_VIS_PATH = f"{MODEL_PATH}/visualization.png"

random.seed(1001)


if __name__ == "__main__":
    signal_type = "fourier"
    n_trials = 2
    n_seeds = 3
    max_bandwidth = 100 if signal_type == "fourier" else 500
    bandwidth_decrement = 10 if signal_type == "fourier" else 50

    for trial in range(n_trials):
        for bandwidth in range(max_bandwidth, 1, -bandwidth_decrement):
            for seed in range(n_seeds):
                empirical_save_path = f"{EMPIRICAL_PATH}/{trial}"
                # Load model configs
                model, x, y, configs = load_model_and_configs(f"{empirical_save_path}/data_{bandwidth}.npy",
                                                            f"{empirical_save_path}/configs.json",
                                                            f"{empirical_save_path}/weights_{bandwidth}_{seed}.pth",
                                                            MODEL)

                # visualize model prediction
                # visualization_1d(model, x, y, one_to_one=True, save_path=HASH_VIS_PATH)
                figure_save_path = f"{FIGURE_PATH}/{trial}"
                create_subdirectories(figure_save_path)
                visualization_2d(model, x, y, one_to_one=True, threeD=False, save_path=f"{figure_save_path}/visualization_{bandwidth}_{seed}.png")
                visualization_2d_surface(model, x, save_path=f"{figure_save_path}/visualization_surface_{bandwidth}_{seed}.png")
                visualization_2d_animated(model, x, y, one_to_one=True, threeD=False, animate=True, save_path=f"{figure_save_path}/visualization_animated_{bandwidth}_{seed}.gif")

    # for turning in ["noturning", "scale", "flip"]:  
    #     visualization_hash_and_turning(model, x, turning=turning, save_path=f"{VIS_PATH}/hash_turning_{DATA_ID}_{turning}.png")
        
