import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

import wandb
import io
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from utils import AbstractTrainer
from ngp_experiments.models import HashEmbedder, NGP

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
    config.n_levels = int(config_dict['HASH']['n_levels'])
    config.log2_n_features = int(config_dict['HASH']['log2_n_features'])
    config.feature_dim = int(config_dict['HASH']['feature_dim'])
    config.base_resolution = int(config_dict['HASH']['base_resolution'])
    config.finest_resolution = int(config_dict['HASH']['finest_resolution'])
    
    config.dim_in = int(config_dict['NETWORK']['dim_in'])
    config.dim_out = int(config_dict['NETWORK']['dim_out'])
    config.hidden_dim = int(config_dict['NETWORK']['hidden_dim'])
    config.n_layers = int(config_dict['NETWORK']['n_layers'])
    
    config.lr = float(config_dict['TRAINING']['lr'])
    config.iterations = int(config_dict['TRAINING']['iterations'])
    config.device = str(config_dict['TRAINING']['device'])
    config.dataset = str(config_dict['TRAINING']['dataset'])
    config.coord_mode = int(config_dict['TRAINING']['coord_mode'])
    config.model = str(config_dict['TRAINING']['model'])
    config.batch_size = int(config_dict['TRAINING']['batch_size'])

    config.exp_seed = int(config_dict['EXPERIMENT']['exp_seed'])
    config.exp_data_idx = int(config_dict['EXPERIMENT']['exp_data_idx'])
    config.exp_type = int(config_dict['EXPERIMENT']['exp_type'])
    config.exp_p1 = int(config_dict['EXPERIMENT']['exp_p1'])

    config.use_wandb = int(config_dict['WANDB']['use_wandb'])
    config.wandb_project = str(config_dict['WANDB']['wandb_project'])
    config.wandb_entity = str(config_dict['WANDB']['wandb_entity'])
    config.wandb_group = str(config_dict['WANDB']['wandb_group'])

    config.save_dir = "results"
    config.log_dir = "logs"
    
    return config



class Trainer(AbstractTrainer):
    def __init__(self, loader, img_shape, args):
        self.loader = loader
        self.img_shape = img_shape
        self.args = args
        self.args.device = torch.device(self.args.device)

        self.hasher = self.get_hasher()
        self.model = self.get_model(self.hasher).to(self.args.device)
        #self.model = torch.nn.DataParallel(self.model).to(self.args.device)

        self.opt = Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.99), eps=1e-15)
        self.scheduler = StepLR(self.opt, step_size=1000, gamma=0.1)

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
            self.scheduler.step()

            preds = preds.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            psnr = psnr_func(preds, y, data_range=1)
            ssim = ssim_func(preds, y, data_range=1, channel_axis=-1)
            losses.append(loss.item())
            psnrs.append(psnr)
            ssims.append(ssim)

            if self.args.use_wandb:
                log_dict = {"loss": loss.item(),
                            "psnr": psnr,
                            'ssim': ssim}
                if it % 500 == 1:
                    if it == 1:
                        predicted_img = self.generate_image(self.args.dim_in, gt=True)
                    else:
                        predicted_img = self.generate_image(self.args.dim_in, gt=False)
                    
                    hash_plot_ls, hash_axis = self.hash_visualizer(self.args.dim_in)
                    mlp_space = self.mlp_space_visualizer(hash_axis, self.args.dim_in)
                    for i, hash_plot in enumerate(hash_plot_ls):
                        log_dict[f"hash_space_{i}"] = wandb.Image(hash_plot)
                    log_dict["MLP_space"] = wandb.Image(mlp_space)
                    log_dict["Reconstruction"] = wandb.Image(predicted_img)
                    self.save_model()

                wandb.log(log_dict, step=it)

        return losses
    
    def train_megapixels(self):
        losses = []
        psnrs = []
        #ssims = []
        for it in range(1, self.args.iterations + 1):
            instance_mse = 0
            n = 0
            iter_loader = iter(self.loader)
            for batch in range(len(self.loader)):
                x, y = next(iter_loader)
                # load data, pass through model, get loss
                preds = self.model(x.to(self.args.device))

                mse = (preds - y.to(self.args.device)) ** 2
                loss = mse.mean()

                # backprop
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.scheduler.step()
                
                instance_mse += mse.sum().item()
                n += len(x)

            instance_loss = instance_mse / n
            instance_psnr = -10*np.log10(instance_loss)
            losses.append(instance_loss)
            psnrs.append(instance_psnr)
            #ssims.append(ssim)

            if self.args.use_wandb:
                log_dict = {"loss": instance_loss,
                            "psnr": instance_psnr}
                if it % 50 == 1:
                    if it == 1:
                        print("generating image...")
                        predicted_img, ssim = self.generate_megaimage(gt=True)
                    else:
                        predicted_img, ssim = self.generate_megaimage(gt=False)
                    
                    log_dict["Reconstruction"] = wandb.Image(predicted_img)
                    log_dict["ssim"] = ssim

                wandb.log(log_dict, step=it)
                self.save_model()

        return losses
    
    
    def get_model(self, hasher):
        model = NGP( 
            hash_table = hasher,
            hidden_features = self.args.hidden_dim,
            hidden_layers = self.args.n_layers,
            out_features = self.args.dim_out
        )
        return model
    
    def get_hasher(self):
        hasher = HashEmbedder(self.img_shape,
                              n_levels=self.args.n_levels,
                              n_features_per_level=self.args.feature_dim,
                              log2_hashmap_size=self.args.log2_n_features,
                              base_resolution=self.args.base_resolution,
                              finest_resolution=self.args.finest_resolution)
        return hasher

    def generate_image(self, dim, gt=True):
        x, pred = get_img_pred(self.model, self.loader, self.img_shape, gt=gt, device=self.args.device)
        if dim == 2:
            output_image  = Image.fromarray((pred*255).astype(np.uint8), mode='RGB')
        elif dim == 1:
            plt.plot(x, pred)
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            output_image = Image.open(img_buf)
            plt.close()

        return output_image
    
    def generate_megaimage(self, gt=False):
        predicted_img, ssim = get_megaimg_pred(self.model, self.loader, self.img_shape, gt=gt)
        output_image  = Image.fromarray((predicted_img*255).astype(np.uint8))
        resize_x = 512
        resize_y = int(self.img_shape[1] / self.img_shape[0] * 512)

        return output_image.resize((resize_x, resize_y), Image.Resampling.NEAREST), ssim

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, f"trained_model_{self.args.model}.pt"))
        print("Saved checkpoint at %s" % str(os.path.join(self.args.save_dir, f"trained_model_{self.args.model}.pt")))

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.args.device))

    def hash_visualizer(self, dim):
        x, y = next(iter(self.loader))
        
        plot_ls = []
        plot_axis = []
        # Get hashed coordinates (self.args.n_levels different levels)
        hash_vals = self.model.hash_table(x.to(self.args.device)).detach().clone().cpu().numpy()
        for i in range(self.args.n_levels):
            if torch.min(y) < 0:
                c = y / 2 + 0.5
            else: 
                c = y
            hash_vertices = self.locate_hash_vertices(i)
            c = c.detach().cpu().numpy()
            
            if dim == 2:
                im, ax_lims = self.hash_2d(hash_vals[:, 2*i:2*i+2], hash_vertices, c)
            elif dim == 1:
                im, ax_lims = self.hash_1d(x, hash_vals)

            plot_ls.append(im)
            plot_axis.append(ax_lims)

        return plot_ls, plot_axis

    def hash_2d(self, hash_vals, hash_vertices, c):
        fig = plt.figure(figsize=(7, 7))
        ax1 = fig.add_subplot(111)
        ax1.scatter(hash_vals[:, 0], hash_vals[:, 1], color=c, alpha=0.5, s=0.5)
        ax1.scatter(hash_vertices[:, 0], hash_vertices[:, 1], color='r', alpha=1, s=2)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)

        xl, xr = ax1.get_xlim()
        yl, yr = ax1.get_ylim()

        plt.close()

        return im, [xl, xr, yl, yr]

    def hash_1d(self, x, hash_vals):
        xl = np.min(hash_vals)
        xr = np.max(hash_vals)

        plt.plot(hash_vals, x)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)

        plt.close()

        return im, [xl, xr]
    
    def locate_hash_vertices(self, level):
        return self.model.hash_table.embeddings[level].weight.data.clone().detach().cpu().numpy()
    
    def mlp_space_visualizer(self, axis, dim):
        if dim == 2:
            return self.mlp_2d(axis)
        elif dim == 1:
            return self.mlp_1d(axis)
        else:
            raise NotImplementedError

    def mlp_2d(self, axis):
        (xmin, xmax, ymin, ymax) = axis[0]
        # Mesh based on hashed coordinates range
        x_hashed_mesh = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(xmin, xmax, 512),
                        torch.linspace(ymax, ymin, 512),
                    ]
                ),
                dim=-1,
            ).view(-1, 2).to(self.args.device)
        # Generate learned INR  
        preds = self.model.net(x_hashed_mesh)
        preds = preds.clamp(-1, 1).detach().cpu().numpy()
        if np.min(preds) < 0:
            preds = preds / 2 + 0.5
        preds = preds.reshape((512, 512, 3))
        preds = np.moveaxis(preds, 0, 1)
        preds = Image.fromarray((preds*255).astype(np.uint8), mode='RGB')

        return preds

    def mlp_1d(self, axis):
        (xmin, xmax) = axis[0]
        # Mesh based on hashed coordinates range
        x_hashed_mesh = torch.linspace(xmin, xmax, 512).view(-1, 1).to(self.args.device)
        # Generate learned INR  
        preds = self.model.net(x_hashed_mesh)
        preds = preds.clamp(-1, 1).detach().cpu().numpy()
        if np.min(preds) < 0:
            preds = preds / 2 + 0.5
        plt.plot(x_hashed_mesh.detach().cpu().numpy(), preds)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)

        plt.close()

        return im


def get_img_pred(model, loader, img_shape, device="cuda", gt=False):
    # predict
    with torch.no_grad():
        x, y = next(iter(loader))
        y = y.detach().cpu().numpy()
        if gt:
            pred = y
        else:
            pred = model(x.to(device)).clamp(-1,1)
            pred = pred.detach().cpu().numpy()
    
    # reshape, convert to image, return
    pred = pred.reshape(img_shape)
    if np.min(y) < 0:
        pred = pred / 2 + 0.5

    return x, pred


def get_megaimg_pred(model, loader, img_shape, device="cuda", gt=False):
    # predict
    empty_image = torch.zeros(img_shape).to(device)
    empty_image_gt = torch.zeros(img_shape).to(device)
    with torch.no_grad():
        iter_loader = iter(loader)
        for batch in range(len(loader)):
            x, y = next(iter_loader)
            if gt:
                pred = y.to(device)
            else:
                pred = model(x.to(device)).clamp(-1,1)

            # Unnormalized coords
            x0_coord = x[:, 0].type(torch.long)
            x1_coord = x[:, 1].type(torch.long)
            # Normalized coords in range [-1, 1]
            #x0_coord = torch.round((x[:, 0] * 0.5 + 0.5) * img_shape[0]).long().clamp(0, img_shape[0]-1)
            #x1_coord = torch.round((x[:, 1] * 0.5 + 0.5) * img_shape[1]).long().clamp(0, img_shape[1]-1)

            empty_image[x0_coord, x1_coord] = pred.clamp(0, 1)  #(pred * 0.5 + 0.5).clamp(0, 1)
            empty_image_gt[x0_coord, x1_coord] = y.to(device).clamp(0, 1)

    empty_image = empty_image.detach().cpu().numpy()
    empty_image_gt = empty_image_gt.detach().cpu().numpy()
    ssim = ssim_func(empty_image, empty_image_gt, data_range=2, channel_axis=-1)
            
    if gt:
        return empty_image_gt, ssim
    else:
        return empty_image, ssim