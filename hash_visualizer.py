import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

import wandb
import random

from misc import *
from models.model_loader import load_default_models

def visualize_model_1d_prediction(model, x, y, save_path="ngp_pred.png"):
    """Generic model prediction."""
    # get the original prediction of the model
    preds = model(x)
    
    # visualize the target signal and the prediction
    plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label="target")
    plt.plot(x.detach().cpu().numpy(), preds.detach().cpu().numpy(), label="prediction")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def visualization_1d(model, x, one_to_one=True, save_path="visualization.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    preds = model(x)
    ax[0].plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), label="target")
    ax[0].plot(x.detach().cpu().numpy(), preds.detach().cpu().numpy(), label="prediction")
    ax[0].legend()
    ax[0].set_title("Model prediction & Target signal")

    if one_to_one:
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # # visualize the hash function
        # hash_vals = model.hash_table(x)
        # hash_vals = hash_vals.detach().cpu().numpy()
        # ax[1].plot(hash_vals, x.detach().cpu().numpy(), label="hash_vals")
        # ax[1].legend()
        # # visualize the mlp function
        # min_hash = np.min(hash_vals)
        # max_hash = np.max(hash_vals)
        # samples = torch.linspace(min_hash, max_hash, 1000).unsqueeze(1).to(x.device)
        # mlp_vals = model.net(samples)
        # mlp_vals = mlp_vals.detach().cpu().numpy()
        # ax[0].plot(samples.cpu().numpy(), mlp_vals, label="mlp_vals")
        # ax[0].legend()

        # visualize the hash function
        hash_vals = model.hash_table(x)
        hash_vals = hash_vals.detach().cpu().numpy()
        ax[1].plot(hash_vals, x.detach().cpu().numpy(), label="hash_vals", color='b')
        ax[1].legend()
        ax[1].set_xlabel("MLP domain")
        ax[1].set_ylabel("Hash/Input domain")
        
        # visualize the mlp function
        min_hash = np.min(hash_vals)
        max_hash = np.max(hash_vals)
        samples = torch.linspace(min_hash, max_hash, 1000).unsqueeze(1).to(x.device)
        mlp_vals = model.net(samples)
        mlp_vals = mlp_vals.detach().cpu().numpy()
        ax2 = ax[1].twinx()
        ax2.plot(samples.cpu().numpy(), mlp_vals, label="mlp_vals", color='orange')
        ax2.set_ylabel("MLP range")
        
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

    ax[0].plot(x.detach().cpu().numpy(), new_hash_vals.detach().cpu().numpy(), label="hash_vals", color='b')
    ax[0].set_xlabel("Input domain")
    ax[0].set_ylabel("MLP domain")
    ax[0].set_title("Hash function")
    ax[0].legend(loc='upper left')

    # visualize the mlp function
    if turning == "scale":
        mid_point = len(x) // 2
        mid_y = model.net(new_hash_vals[mid_point]).item()
        ax[1].scatter((0.5), (mid_y), s=100, color='r', alpha=0.5, label="New turning point")
    ax[1].plot(x.detach().cpu().numpy(), preds.detach().cpu().numpy(), label="prediction", color='orange')
    ax[1].set_xlabel("Input domain")
    ax[1].set_title("Model prediction")
    ax[1].legend(loc='upper left')

    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()


def visualize_model_prediction(trainer, x, y, data_shape, log):
    """Generic model prediction."""
    # get the original prediction of the model
    preds = trainer.model(x.to(trainer.args.device))
    loss = ((preds - y.to(trainer.args.device)) ** 2).mean()
    psnr = -10*np.log10(loss.item())
    # generate the reconstructed image
    preds = preds.clamp(-1, 1).detach().cpu().numpy()
    preds = preds.reshape(data_shape)
    preds = preds / 2 + 0.5
    im = Image.fromarray((preds*255).astype(np.uint8), mode='RGB')

    log["psnr"] = psnr
    log["recon"] = wandb.Image(im)


def visualize_diner_hash_function(trainer, data_shape, log):
    """
    Visualize the hash map of DINER.
    
    Note that DINER hash values are 2D, while we visualize in 3D. 
    We do so by padding the R channel with zeros.
    """
    # Visualizing the hash operation
    hashed_input = trainer.model.table.data     # diner
    hashed_input = hashed_input - torch.min(hashed_input, dim=0)[0]
    hashed_input = hashed_input / torch.max(hashed_input, dim=0)[0]
    zeros = torch.zeros((hashed_input.shape[0], 1)).to(trainer.args.device)
    hashed_input = torch.cat((zeros, hashed_input), dim=1)
    hashed_im = hashed_input.reshape(data_shape).detach().cpu().numpy()
    hashed_im = Image.fromarray((hashed_im*255).astype(np.uint8), mode='RGB')

    log["hashing"] = wandb.Image(hashed_im)


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


MEGAPIXELS = ["pluto", "tokyo", "mars"]
DATA_ID = 0
MODEL = "ngp"

MODEL_PATH = 'results/%s/%s_%s' % (MODEL, MODEL, DATA_ID)
MODEL_WEIGHTS_PATH = "%s/%s_%s.pth" % (MODEL_PATH, MODEL, DATA_ID)
MODEL_PRED_PATH = "%s/ngp_pred.png" % MODEL_PATH
MODEL_HASH_PATH = "%s/visualization.png" % MODEL_PATH

random.seed(1001)


if __name__ == "__main__":
    for DATA_ID in tqdm(range(13, 14)):
        MODEL_PATH = 'results/%s/%s_%s' % (MODEL, MODEL, DATA_ID)
        MODEL_WEIGHTS_PATH = "%s/weights.pth" % (MODEL_PATH)
        MODEL_PRED_PATH = "%s/ngp_pred.png" % MODEL_PATH
        MODEL_HASH_PATH = "%s/visualization.png" % MODEL_PATH
        EXP_VIS_PATH = "vis/experiments"

        # Load model configs
        x, y = load_data("%s/data.npy" % (MODEL_PATH))
        x = torch.tensor(x).to("cuda").unsqueeze(1)
        y = torch.tensor(y).to("cuda").unsqueeze(1)
        configs = load_configs("%s/configs.json" % MODEL_PATH)

        
        model, optim, scheduler, configs = load_default_models(MODEL, configs=configs)

        # load model
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

        # visualize model prediction
        visualization_1d(model, x, one_to_one=True, save_path=MODEL_HASH_PATH)

        for turning in ["noturning", "scale", "flip"]:  
            visualization_hash_and_turning(model, x, turning=turning, save_path=f"{EXP_VIS_PATH}/hash_turning_{DATA_ID}_{turning}.png")
    
