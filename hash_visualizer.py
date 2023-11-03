from argparse import ArgumentParser
import numpy as np
from PIL import Image
import torch

import wandb
import random

from main_image import get_data


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
IMG_ID = 20


random.seed(1001)


def main():
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("model")
    parser.add_argument("saved_model_path")
    common_arg = parser.parse_args()
    
    # Import model specific packages
    if common_arg.model == "diner":
        # Diner experiment package
        import diner_experiments.utils as MODE
    elif common_arg.model == "ngp":
        # NGP experiment package
        import ngp_experiments.utils as MODE

    # load model specific config file    
    args = MODE.load_config(common_arg.config_file)

    # make dataset and loader
    dataset, data_idx = get_data(args.dataset, args.batch_size, idx=IMG_ID, coord_mode=args.coord_mode)
    data_shape = dataset.get_data_shape()
    print(data_shape)

    # make trainer
    trainer = MODE.Trainer(dataset, data_shape, args)
    # load model
    trainer.load_model(common_arg.saved_model_path)
    print("Model parameters: ", trainer.get_model_size())

    # init wanbd report for visualizing the experiment results
    wandb.init(project=args.wandb_project,
               entity=args.wandb_entity,
               group="hash_visualization",
               name="%s_visuals" % common_arg.model
               )

    # understand what the highest resolution hashing is contributing to the reconstruction
    x, y = next(iter(dataset))

    # Train on diner hash
    #hashing_trainer = MODE.HashTrainer(dataset, trainer.model, data_shape, args)
    #hashing_trainer.train()
    # Visualization set
    log = {}
    visualize_model_prediction(trainer, x, y, data_shape, log)
    #visualize_hash_function(trainer, data_shape, log, model=common_arg.model)
    visualize_isolated_ngp_hashing(trainer, x, y, data_shape, log)
    wandb.log(log)

    # visualizing the r-slice of the MLP space


if __name__ == "__main__":
    main()
    
