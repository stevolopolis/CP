import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

import wandb
import random

from utils import *
from models import *


def get_model_hash_net_output(model, x):
    hash_vals = model.hash_table(x)
    min_hash = torch.min(hash_vals)
    max_hash = torch.max(hash_vals)
    mlp_domain = torch.linspace(min_hash, max_hash, 1000).unsqueeze(1).to(x.device)
    mlp_vals = model.net(mlp_domain)

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


def load_model_and_configs(data_path, config_path, model_path, model_type, device="cuda"):
    x, y = load_data(data_path)
    x = torch.tensor(x).to(device).unsqueeze(1)
    y = torch.tensor(y).to(device).unsqueeze(1)

    # Load model configs
    if config_path is not None:
        configs = load_configs(config_path)
    # Load default model configs
    else:
        configs = get_default_model_configs(model_type)
    # Get model
    model = get_model(model_type, 1, 1, 1, configs, device=device)
    # load model
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model, x, y, configs



MEGAPIXELS = ["pluto", "tokyo", "mars"]
DATA_ID = 0
MODEL = "ngp"

MODEL_PATH = 'results/%s/%s_%s' % (MODEL, MODEL, DATA_ID)
MODEL_WEIGHTS_PATH = "%s/%s_%s.pth" % (MODEL_PATH, MODEL, DATA_ID)
MODEL_PRED_PATH = "%s/ngp_pred.png" % MODEL_PATH
MODEL_HASH_PATH = "%s/visualization.png" % MODEL_PATH

random.seed(1001)


if __name__ == "__main__":
    for DATA_ID in tqdm(range(0, 14)):
        MODEL_PATH = 'results/%s/%s_%s' % (MODEL, MODEL, DATA_ID)
        MODEL_WEIGHTS_PATH = "%s/weights.pth" % (MODEL_PATH)
        MODEL_PRED_PATH = "%s/ngp_pred.png" % MODEL_PATH
        MODEL_HASH_PATH = "%s/visualization.png" % MODEL_PATH
        EXP_VIS_PATH = "vis/experiments"

        # Load model configs
        model, x, y, configs = load_model_and_configs(f"{MODEL_PATH}/data.npy", f"{MODEL_PATH}/configs.json", MODEL_WEIGHTS_PATH, MODEL)

        # visualize model prediction
        visualization_1d(model, x, y, one_to_one=True, save_path=MODEL_HASH_PATH)

        for turning in ["noturning", "scale", "flip"]:  
            visualization_hash_and_turning(model, x, turning=turning, save_path=f"{EXP_VIS_PATH}/hash_turning_{DATA_ID}_{turning}.png")
    
