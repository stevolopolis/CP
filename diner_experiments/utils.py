import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func

import wandb

from torch.optim import Adam

from utils import AbstractTrainer
from diner_experiments.models import DinerSiren, DinerMLP

import configparser
import argparse


def load_config(configfile):
    '''
        Load configuration file
    '''
    config_dict = configparser.ConfigParser()
    config_dict.read(configfile)
    
    config = argparse.Namespace()
    
    # Specific modifications    
    config.dim_in = int(config_dict['NETWORK']['dim_in'])
    config.dim_out = int(config_dict['NETWORK']['dim_out'])
    config.hidden_dim = int(config_dict['NETWORK']['hidden_dim'])
    config.n_layers = int(config_dict['NETWORK']['n_layers'])
    config.w_0 = float(config_dict['NETWORK']['w_0'])
    
    config.lr = float(config_dict['TRAINING']['lr'])
    config.iterations = int(config_dict['TRAINING']['iterations'])
    config.device = str(config_dict['TRAINING']['device'])
    config.dataset = str(config_dict['TRAINING']['dataset'])
    config.coord_mode = int(config_dict['TRAINING']['coord_mode'])
    config.model = str(config_dict['TRAINING']['model'])
    config.batch_size = int(config_dict['TRAINING']['batch_size'])

    config.use_wandb = int(config_dict['WANDB']['use_wandb'])
    config.wandb_project = str(config_dict['WANDB']['wandb_project'])
    config.wandb_entity = str(config_dict['WANDB']['wandb_entity'])

    config.save_dir = "results"
    config.log_dir = "logs"
    
    return config



class Trainer(AbstractTrainer):
    def __init__(self, loader, img_shape, args):
        self.loader = loader
        self.img_shape = img_shape
        self.args = args

        self.model = self.get_model().to(self.args.device)
        self.opt = Adam(self.model.parameters(), lr=self.args.lr)

    def train(self):
        losses = []
        psnrs = []
        ssims = []
        for it in range(1, self.args.iterations + 1):
            x, y = next(iter(self.loader))
            # load data, pass through model, get loss
            preds = self.model(x.to(self.args.device))

            loss = ((preds - y.to(self.args.device)) ** 2).mean()
            psnr = -10*np.log10(loss.item())

            # backprop
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            ssim = ssim_func(preds.cpu().detach().numpy(), y.cpu().detach().numpy(), data_range=2, channel_axis=-1)
            losses.append(loss.item())
            psnrs.append(psnr)
            ssims.append(ssim)

            if self.args.use_wandb:
                log_dict = {"loss": loss.item(),
                            "psnr": psnr}
                if it % 500 == 1:
                    if it == 1:
                        predicted_img = self.generate_image(gt=True)
                    else:
                        predicted_img = self.generate_image(gt=False)
                    
                    mlp_space = self.mlp_space_visualizer()
                    hash_space = self.hash_visualizer()
                    log_dict["Reconstruction"] = wandb.Image(predicted_img)
                    log_dict["MLP_space"] = wandb.Image(mlp_space)
                    log_dict["hash_space"] = wandb.Image(hash_space)

                wandb.log(log_dict, step=it)
                self.save_model()

        return losses
    
    def get_model(self):
        if self.args.model == "diner_relu":
            model = DinerMLP( 
                hash_table_length = self.img_shape[0] * self.img_shape[1],
                in_features = self.args.dim_in,
                hidden_features = self.args.hidden_dim,
                hidden_layers = self.args.n_layers,
                out_features = self.args.dim_out
            )
        elif self.args.model == "diner_siren":
            model = DinerSiren( 
            hash_table_length = self.img_shape[0] * self.img_shape[1],
            in_features = self.args.dim_in,
            hidden_features = self.args.hidden_dim,
            hidden_layers = self.args.n_layers,
            out_features = self.args.dim_out,
            outermost_linear=True,
            first_omega_0=self.args.w_0,
            hidden_omega_0=self.args.w_0
            )

        return model

    def generate_image(self, gt=True):
        predicted_img = get_img_pred(self.model, self.loader, self.img_shape, gt=gt, device=self.args.device)
        output_image  = Image.fromarray((predicted_img*255).astype(np.uint8), mode='RGB')

        return output_image
    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, f"trained_model_{self.args.model}.pt"))

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.args.device))

    def hash_visualizer(self):
        import io
        import matplotlib.pyplot as plt

        x, y = next(iter(self.loader))
        
        hash_vals = self.model.table.data.detach().clone().cpu().numpy()
        c = y.detach().cpu().numpy() / 2 + 0.5
        plt.figure(figsize=(6, 6))
        plt.scatter(hash_vals[:, 0], hash_vals[:, 1], color=c, alpha=.5, s=.5)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)
        plt.close()

        return im

    def mlp_space_visualizer(self):
        x, y = next(iter(self.loader))
        # Get hashed coordinates range
        hash_vals = self.model.table.data.detach().clone().cpu().numpy()
        hash_min_x = hash_vals[:, 0].min()
        hash_max_x = hash_vals[:, 0].max()
        hash_min_y = hash_vals[:, 1].min()
        hash_max_y = hash_vals[:, 1].max()

        # Mesh based on hashed coordinates range
        x_hashed_mesh = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(hash_min_x, hash_max_x, self.img_shape[0]),
                        torch.linspace(hash_max_y, hash_min_y, self.img_shape[1]),
                    ]
                ),
                dim=-1,
            ).view(-1, 2).to(self.args.device)
        
        # Generate learned INR  
        preds = self.model.net(x_hashed_mesh)
        preds = preds / 2 + 0.5
        preds = preds.clamp(0, 1).detach().cpu().numpy()
        preds = preds.reshape(self.img_shape)
        preds = Image.fromarray((preds*255).astype(np.uint8), mode='RGB')

        return preds


def get_img_pred(model, loader, img_shape, device="cuda", gt=False):
    # predict
    with torch.no_grad():
        x, y = next(iter(loader))
        if gt:
            pred = y
        else:
            pred = model(x.to(device))
    
    # reshape, convert to image, return
    pred = pred.reshape(img_shape)
    # Normalize from [-1, 1] to [0, 1]
    pred = pred / 2 + 0.5

    return pred.clamp(0, 1).detach().cpu().numpy()


class HashTrainer(AbstractTrainer):
    def __init__(self, loader, model, img_shape, args):
        self.loader = loader
        self.img_shape = img_shape
        self.args = args

        # Get hash table from model
        self.model = model
        self.model.eval()
        self.hash_gt = self.model.table.data
        self.hash_min = torch.min(self.hash_gt, dim=0)[0]
        self.hash_gt = self.hash_gt - self.hash_min
        self.hash_max = torch.max(self.hash_gt, dim=0)[0]
        self.hash_gt = self.hash_gt / self.hash_max
        # Get model
        self.hash_model = self.get_model().to(self.args.device)
        self.opt = Adam(self.hash_model.parameters(), lr=self.args.lr)

    def train(self):
        losses = []
        psnrs = []
        ssims = []
        for it in range(1, self.args.iterations + 1):
            x, y = next(iter(self.loader))
            # load data, pass through model, get loss
            hash_preds = self.hash_model(x.to(self.args.device))
            preds = self.model.net(hash_preds * self.hash_max + self.hash_min)

            hash_loss = ((hash_preds - self.hash_gt.to(self.args.device)) ** 2).mean()
            loss = ((preds - y.to(self.args.device)) ** 2).mean()
            psnr = -10*np.log10(loss.item())

            # backprop
            self.opt.zero_grad()
            hash_loss.backward()
            self.opt.step()

            ssim = ssim_func(preds.cpu().detach().numpy(), y.cpu().detach().numpy(), data_range=2, channel_axis=-1)
            losses.append(loss.item())
            psnrs.append(psnr)
            ssims.append(ssim)

            if self.args.use_wandb:
                log_dict = {"hash_loss": hash_loss.item(),
                            "loss": loss.item(),
                            "psnr": psnr}
                if it % 500 == 1:
                    if it == 1:
                        predicted_img = self.generate_image(gt=True)
                        prediced_hash = self.generate_hash(gt=True)
                    else:
                        predicted_img = self.generate_image(gt=False)
                        prediced_hash = self.generate_hash(gt=False)
                    
                    hash_space = self.hash_visualizer()
                    log_dict["Reconstruction"] = wandb.Image(predicted_img)
                    log_dict["hash_space"] = wandb.Image(hash_space)
                    log_dict["hash"] = wandb.Image(prediced_hash)

                wandb.log(log_dict, step=it)
                self.save_model()

        return losses
    
    def get_model(self):
        if self.args.model == "diner_relu":
            model = DinerMLP( 
                hash_table_length = self.img_shape[0] * self.img_shape[1],
                in_features = self.args.dim_in,
                hidden_features = self.args.hidden_dim,
                hidden_layers = self.args.n_layers,
                out_features = 2
            )
        elif self.args.model == "diner_siren":
            model = DinerSiren( 
            hash_table_length = self.img_shape[0] * self.img_shape[1],
            in_features = self.args.dim_in,
            hidden_features = self.args.hidden_dim,
            hidden_layers = self.args.n_layers,
            out_features = 2,
            outermost_linear=True,
            first_omega_0=self.args.w_0,
            hidden_omega_0=self.args.w_0
            )

        return model
    
    def generate_hash(self, gt=True):
        # predict
        with torch.no_grad():
            x, y = next(iter(self.loader))
            if gt:
                pred = self.hash_gt
            else:
                pred = self.hash_model(x.to(self.args.device)) * self.hash_max + self.hash_min

        zeros = torch.zeros((pred.shape[0], 1)).to(self.args.device)
        pred = torch.cat((zeros, pred), dim=1)
        # reshape, convert to image, return
        pred = pred.reshape(self.img_shape).detach().cpu().numpy()

        output_image  = Image.fromarray((pred*255).astype(np.uint8), mode='RGB')

        return output_image
    

    def generate_image(self, gt=True):
        # predict
        with torch.no_grad():
            x, y = next(iter(self.loader))
            if gt:
                pred = y
            else:
                hash = self.hash_model(x.to(self.args.device)) * self.hash_max + self.hash_min
                pred = self.model.net(hash)
        
        # reshape, convert to image, return
        pred = pred.reshape(self.img_shape)
        # Normalize from [-1, 1] to [0, 1]
        pred = pred / 2 + 0.5
        pred = pred.clamp(0, 1).detach().cpu().numpy()

        output_image  = Image.fromarray((pred*255).astype(np.uint8), mode='RGB')

        return output_image
    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, f"trained_model_{self.args.model}.pt"))

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.args.device))

    def hash_visualizer(self):
        import io
        import matplotlib.pyplot as plt

        x, y = next(iter(self.loader))
        
        hash_vals = self.hash_model.table.data.detach().clone().cpu().numpy()
        c = self.hash_gt.detach().cpu().numpy() / 2 + 0.5
        zeros = np.zeros((hash_vals.shape[0], 1))
        c = np.concatenate((c, zeros), axis=1)
        plt.figure(figsize=(6, 6))
        plt.scatter(hash_vals[:, 0], hash_vals[:, 1], c=c, alpha=.5, s=.5)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)
        plt.close()

        return im