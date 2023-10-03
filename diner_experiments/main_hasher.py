import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import numpy as np
from PIL import Image
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import wandb
from pathlib import Path
import pprint

from diner_experiments.models import DinerMLP, DinerSiren , DinerSiren_Hasher, MLP
from vanilla_mri_experiments.mri import SIRENWrapper
from data import ImageFile, PointCloud

        
def train(hasher, model, loader, opt, model_opt, img_shape, iterations=10000, use_wandb=None, device="cuda"):
    hash_losses = []
    losses = []
    psnrs = []
    for it in range(1, iterations + 1):
        if it < 5000:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        else:
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        for x, y in loader:
            # load data, pass through model, get loss
            hash = hasher(x.to(device))
            preds = model.hasher_forward(hash)

            hash_loss = ((hash - model.table) ** 2).mean()
            loss = ((preds - y.to(device)) ** 2).mean()
            psnr = -10*np.log10(loss.item())

            sum_loss = hash_loss + .1*loss

            # backprop
            if it < 5000:
                opt.zero_grad()
                sum_loss.backward()
                opt.step()
            else:
                model_opt.zero_grad()
                loss.backward()
                model_opt.step()

            hash_losses.append(hash_loss.item())
            losses.append(loss.item())
            psnrs.append(psnr)

            if use_wandb:
                log_dict = {"hash_loss": hash_loss.item(),
                            "loss": loss.item(),
                            "psnr": psnr}
                if it % 500 == 1:
                    if it == 1:
                        predicted_img = get_img_pred(hasher, model, loader, img_shape, gt=True)
                    else:
                        predicted_img = get_img_pred(hasher, model, loader, img_shape, gt=False)
                    output_image  = Image.fromarray((predicted_img*255).astype(np.uint8))
                    log_dict["Reconstruction"] = wandb.Image(output_image)

                wandb.log(log_dict, step=it)

    return hash_losses, losses

def get_img_pred(hasher, model, loader, img_shape, device="cuda", gt=False):
    # predict
    with torch.no_grad():
        x, y = next(iter(loader))
        if gt:
            pred = y.detach().cpu().numpy()
        else:
            pred = model.hasher_forward(hasher(x.to(device))).clamp(-1,1)
            pred = pred.detach().cpu().numpy()
    
    # reshape, convert to image, return
    pred = pred.reshape(img_shape)
    # Normalize from [-1, 1] to [0, 1]
    pred = pred / 2 + 0.5

    return pred


def get_args():
    parser = ArgumentParser(fromfile_prefix_chars="@", description="Image representation task")

    parser.add_argument(
        "--model", type=str, help="name of model", default="siren"
    )
    parser.add_argument(
        "--dataset", type=str, help="dataset", default="lake"
    )
    parser.add_argument(
        "--normalize", type=int, help="whether to normalize input coordinates", default=0
    )
    parser.add_argument(
        "--lr", type=float, help="learning rate", default=1e-4
    )
    parser.add_argument(
        "--w_0", type=float, help="SIREN frequency constant", default=30.
    )
    parser.add_argument(
        "--hidden_dim", type=int, help="SIREN hidden dimesion", default=256
    )
    parser.add_argument(
        "--dim_in", type=int, help="SIREN input dimesion", default=2
    )
    parser.add_argument(
        "--dim_out", type=int, help="SIREN output dimesion", default=3
    )
    parser.add_argument(
        "--n_layers", type=int, help="number of SIREN layers", default=5
    )
    parser.add_argument(
        "--hasher_hidden_dim", type=int, help="Diner Hasher hidden dimesion", default=64
    )
    parser.add_argument(
        "--hasher_n_layers", type=int, help="number of Diner Hasher layers", default=2
    )
    parser.add_argument(
        "--base", type=int, help="base for mri", default=2
    )
    parser.add_argument(
        "--save_dir", default="results", type=str, help="directory in which to save image & model"
    )
    parser.add_argument(
        "--log-dir", default="logs", type=str, help="directory to store tensorboard logs"
    )
    parser.add_argument(
        "--iterations", default=10000, type=int, help="Number of training iterations"
    )
    parser.add_argument(
        "--use_wandb", type=int, help="boolean for whether to use w&b for logging", default=1
    )
    parser.add_argument(
        "--wandb_project", default="", type=str, help="W&B project name."
    )
    parser.add_argument(
        "--wandb_entity", default="stevolopolis", type=str, help="W&B project entity."
    )
    parser.add_argument(
        "--device", default="cuda:0", type=str, help="training and inferencing device ('cuda', 'cpu', etc.)."
    )

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    device = args.device

    import random
    random.seed(21)
    random_img_idx = [random.randint(1, 70000) for _ in range(25)]

    for img_idx in random_img_idx:
        # make dataset and loader
        if args.dataset == "FFHQ":
            # Total instances = 70000
            img_idx_str = str(img_idx).zfill(5)
            dataset = ImageFile("/mnt/share/images1024x1024/%s.png" % img_idx_str, normalize=args.normalize)
        elif args.dataset == "ImageNet":
            # Total instances = 100000
            img_idx_str = str(img_idx).zfill(8)
            dataset = ImageFile("/home/mnt/data/imagenet/val/ILSVRC2012_val_%s.JPEG" % (img_idx_str), normalize=True)
        elif args.dataset == "sdf.armadillo":
            dataset = PointCloud("D:\\sdf\\Armadillo.xyz", 4000)
        loader = DataLoader(
            dataset, shuffle=True, batch_size=1, pin_memory=True
        )
        img_shape = (dataset.get_img_h(), dataset.get_img_w(), dataset.get_img_c())
        print(img_shape)

        model = DinerSiren( 
            hash_table_length = dataset.get_img_h() * dataset.get_img_w(),
            in_features = args.dim_in,
            hidden_features = args.hidden_dim,
            hidden_layers = args.n_layers,
            out_features = args.dim_out,
            outermost_linear = True,
            first_omega_0 = 30.0,
            hidden_omega_0 = 30.0
        ).to(device)
        model.load_state_dict(torch.load("results/ImageNet00021621_DinerSiren_layer2_hidden64_w_030.0/trained_model_DinerSiren.pt"))

        hasher = DinerSiren_Hasher(
                in_features=args.dim_in,
                hidden_features=args.hasher_hidden_dim,
                hidden_layers=args.hasher_n_layers,
                outermost_linear=True,
                first_omega_0=args.w_0,
                hidden_omega_0=args.w_0
        ).to(device)
        
        experiment_name = "%s%s_DinerSirenHasher_layer%s_hidden%s_w_0%s" % (args.dataset, img_idx_str, args.hasher_n_layers, args.hasher_hidden_dim, args.w_0)

        args.model_parameters = sum([p.numel() for p in hasher.parameters()])
        
        args.save_dir = os.path.join(args.save_dir, experiment_name)
        args.log_dir = os.path.join(args.log_dir, experiment_name)

        if args.use_wandb:
            # Initialize wandb experiment
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=args,
                group="%s_DINER" % (args.dataset),
                name=experiment_name
            )

            # Save ENV variables
            with (Path(wandb.run.dir) / "env.txt").open("wt") as f:
                pprint.pprint(dict(os.environ), f)

            # Define path where model will be saved
            model_path = Path(wandb.run.dir) / "model.pt"

        print(f"No. of parameters: {args.model_parameters}")
        opt = Adam(hasher.parameters(), lr=args.lr)
        model_opt = Adam(model.parameters(), lr=args.lr)

        # Optionally save model
        if args.use_wandb:
            torch.save({"args": args, "state_dict": hasher.state_dict()}, model_path)
            wandb.save(str(model_path.absolute()), base_path=wandb.run.dir, policy="live")

        train(hasher, model, loader, opt, model_opt, img_shape, iterations=args.iterations, use_wandb=args.use_wandb, device=device)

        # save model state dict and represented image as np array, if path is supplied
        if args.save_dir is not None:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            predicted_img = get_img_pred(hasher, model, loader, img_shape)
            # np.save(os.path.join(args.save_dir, "represented_image.npy"), predicted_img)
            output_image  = Image.fromarray((predicted_img*255).astype(np.uint8))
            output_image.save(os.path.join(args.save_dir, "output.png"))
            torch.save(hasher.state_dict(), os.path.join(args.save_dir, f"trained_model_{args.model}.pt"))

        if args.use_wandb:
            wandb.save(str(model_path.absolute()), base_path=wandb.run.dir, policy="live")
            wandb.finish()


if __name__ == "__main__":
    main()
    
