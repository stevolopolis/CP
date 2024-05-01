import numpy as np
import torch
import math
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
experiment_name = "image_strips"

# Model parameters
MODEL = 'ngp'
MODEL_NAME = f"{MODEL}_bigGrid"

# Training parameters
n_imgs = 1
n_seeds = 1
n = 1000
epoch = 10000
resolution = 50 if MODEL_NAME == "ngp_bigGrid" else 25

# Animation parameters
nframes = 30


def train(base_path, data_path, n_seeds, row=True, device="cuda", use_wandb=False):
    # get image
    img_name = data_path.split("/")[-1].split(".")[0]
    dataLoader = ImageFile(data_path, coord_mode=0, grayscale=True)
    H, W, C = dataLoader.get_data_shape()
    img_in, img_out = next(iter(dataLoader))
    img_in = img_in.view(H, W, -1).to(device)
    img_out = img_out.view(H, W, -1).to(device)
    n_strips = img_in.shape[0] if row else img_in.shape[1]

    signal_type = "row" if row else "col"

    # iterate through image strips
    for strip_idx in range(n_strips):
        # get strip
        sample = img_in[strip_idx, :, 1] if row else img_in[:, strip_idx, 0]
        signal = img_out[strip_idx].squeeze(1) if row else img_out[:, strip_idx].squeeze(1)

        # Save data & configs
        save_data(sample.cpu().numpy(), signal.cpu().numpy(),
                  f"{base_path}/data_{signal_type}_{strip_idx}.npy")

        # Generate specific hash vals
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            # Load default model configs
            configs = get_default_model_configs(MODEL)
            temp_c = configs.NET._replace(base_resolution=resolution)
            temp_c = temp_c._replace(finest_resolution=resolution)
            temp_c = temp_c._replace(log2_n_features=math.ceil(np.log2(resolution)))
            configs = configs._replace(NET = temp_c)

            # Get model
            model = get_model(MODEL, 1, 1, [W if row else H], configs, device=device)
            # Initialize model weights
            if "ordered" in MODEL_NAME:
                model.init_weights(ordered=True)
            else:
                model.init_weights()

            if "frozen" in MODEL_NAME:
                model.freeze_hash_table()
                print("hash table weights frozen")

            # Load default model optimizers and schedulers
            optim, scheduler = get_default_model_opts(MODEL, model, epoch)

            if use_wandb:
                wandb.init(
                    project="1d-image-strips",
                    entity="utmist-parsimony",
                    config=configs.NET._asdict(),
                    group=f"{MODEL_NAME}",
                    name=f"{img_name}_{signal_type}_{strip_idx}_{seed}",
                )

            # Model training
            model_loss, model_preds = trainer(sample.unsqueeze(1), signal.unsqueeze(1), model, optim, scheduler, epoch, nframes, use_wandb=use_wandb)
            
            # Animate model predictions
            animate_model_preds(sample, signal, model_preds, nframes, f"{base_path}/preds_{signal_type}_{strip_idx}_{seed}.mp4")
            # Save model configs
            save_configs(configs, f"{base_path}/configs.json")
            # Save model loss
            save_vals([model_loss], f"{base_path}/loss_{signal_type}_{strip_idx}_{seed}.txt")
            # Save model weights
            torch.save(model.state_dict(), f"{base_path}/weights_{signal_type}_{strip_idx}_{seed}.pth")
            print(f"model weights saved at {base_path}/weights_{signal_type}_{strip_idx}_{seed}.pth")


if __name__ == "__main__":
    DEVICE = "cuda:0"
    BASE_PATH = f"vis/{experiment_name}/{MODEL_NAME}"
    EMPIRICAL_PATH = f"{BASE_PATH}/empirical"
    FIGURE_PATH = f"{BASE_PATH}/figures"
    create_subdirectories(EMPIRICAL_PATH)
    create_subdirectories(FIGURE_PATH)

    # train
    for img_idx in range(1, n_imgs+1):
        empirical_save_path = f"{EMPIRICAL_PATH}/{img_idx}"
        create_subdirectories(empirical_save_path)
        data_path = f"../data/kodak/kodim{str(img_idx).zfill(2)}.png"
        train(empirical_save_path, data_path, n_seeds=n_seeds, device=DEVICE, use_wandb=True)

    # Plot
    plot_segment_summary(EMPIRICAL_PATH, FIGURE_PATH, hashing=MODEL=="ngp", device=DEVICE)

    