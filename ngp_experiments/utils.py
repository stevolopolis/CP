import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

import wandb

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from utils import AbstractTrainer
from ngp_experiments.models import HashEmbedder, ConsistentHashEmbedder, NGP

import configparser
import argparse

from loguru import logger
import io
import matplotlib.pyplot as plt


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
    config.consistent = int(config_dict['HASH']['consistent'])
    
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

    config.use_wandb = int(config_dict['WANDB']['use_wandb'])
    config.wandb_project = str(config_dict['WANDB']['wandb_project'])
    config.wandb_entity = str(config_dict['WANDB']['wandb_entity'])
    config.experiment_name = str(config_dict['WANDB']['name'])

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
        densify_until_iter = 2000

        losses = []
        psnrs = []
        ssims = []
        for it in range(1, self.args.iterations + 1):
            x, y = next(iter(self.loader))
            # load data, pass through model, get loss
            preds, hashed_grid_indices = self.model(x.to(self.args.device))

            loss = ((preds - y.to(self.args.device)) ** 2).mean()
            psnr = -10*np.log10(loss.item())

            # backprop
            self.opt.zero_grad()
            loss.backward()

            # update table grad accum
            if self.args.consistent:
                hash_table = self.model.hash_table
                for i in range(hash_table.start_level, hash_table.n_levels):
                    hgi_unique = hashed_grid_indices[i].unique()
                    mask = torch.zeros_like(hash_table.table_grad_accum[i - hash_table.start_level]).to(bool)
                    mask[hgi_unique] = True
                    hash_table.update_table_grad_accum(i - hash_table.start_level, mask)

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
                        predicted_img = self.generate_image(gt=True)
                    else:
                        predicted_img = self.generate_image(gt=False)
                    mlp_space = self.mlp_space_visualizer()
                    hash_plot_ls = self.hash_visualizer()
                    for i, hash_plot in enumerate(hash_plot_ls):
                        log_dict[f"hash_space_{i}"] = wandb.Image(hash_plot)
                    log_dict["MLP_space"] = wandb.Image(mlp_space)
                    log_dict["Reconstruction"] = wandb.Image(predicted_img)
                    self.save_model()

                if it % 100 == 0 and self.args.consistent:
                    table_grad_accum_plots = self.table_grad_accum_visualizer()
                    for i, grad_accum_plot in enumerate(table_grad_accum_plots):
                        log_dict[f"table_grad_accum_level_{i + self.model.hash_table.start_level}"] \
                            = wandb.Image(grad_accum_plot)

                wandb.log(log_dict, step=it)

            if it < densify_until_iter and it % 400 == 0 and self.args.consistent:
                pass
                self.model.hash_table.densify()

                self.model.hash_table.embeddings[1].weight = torch.nn.Parameter(
                    torch.zeros_like(self.model.hash_table.embeddings[1].weight)
                )
                self.model.hash_table.embeddings[2].weight = torch.nn.Parameter(
                    torch.zeros_like(self.model.hash_table.embeddings[2].weight)
                )
                self.model.hash_table.embeddings[3].weight = torch.nn.Parameter(
                    torch.zeros_like(self.model.hash_table.embeddings[3].weight)
                )
                # self.opt = Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.99), eps=1e-15)

                self.opt = Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.99), eps=1e-15)
                self.scheduler = StepLR(self.opt, step_size=1000, gamma=0.1)

            # 100 iterations to settle down
            if it >= 400 and it % 100 == 0 and self.args.consistent:
                for i in range(len(self.model.hash_table.table_grad_accum)):
                    self.model.hash_table.reset_table_grad_accum(i)

        return losses
    
    def train_megapixels(self):
        if self.args.consistent:
            raise NotImplementedError

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

                if it % 500 == 0 and self.args.consistent:
                    self.model.hash_table.densify()

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
        if self.args.consistent:
            logger.info(f"Using consistent hashing")
            hashing_mode = ConsistentHashEmbedder
        else:
            logger.info(f"Using base hashing")
            hashing_mode = HashEmbedder
        hasher = hashing_mode(self.img_shape,
                              n_levels=self.args.n_levels,
                              n_features_per_level=self.args.feature_dim,
                              log2_hashmap_size=self.args.log2_n_features,
                              base_resolution=self.args.base_resolution,
                              finest_resolution=self.args.finest_resolution)
        return hasher

    def generate_image(self, gt=True):
        predicted_img = get_img_pred(self.model, self.loader, self.img_shape, gt=gt, device=self.args.device)
        output_image  = Image.fromarray((predicted_img*255).astype(np.uint8), mode='RGB')

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

    def hash_visualizer(self):
        x, y = next(iter(self.loader))
        
        plot_ls = []
        # Get hashed coordinates (self.args.n_levels different levels)
        hash_vals = self.model.hash_table(x.to(self.args.device))[0].detach().clone().cpu().numpy()
        for i in range(self.args.n_levels):
            if torch.min(y) < 0:
                c = y / 2 + 0.5
            else: 
                c = y
            hash_vertices = self.locate_hash_vertices(i)
            c = c.detach().cpu().numpy()
            fig = plt.figure(figsize=(7, 7))
            ax1 = fig.add_subplot(111)
            ax1.scatter(hash_vals[:, 2*i], hash_vals[:, 2*i+1], color=c, alpha=0.5, s=0.5)
            ax1.scatter(hash_vertices[:, 0], hash_vertices[:, 1], color='r', alpha=1, s=2)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            im = Image.open(img_buf)
            plot_ls.append(im)
            plt.close()

        return plot_ls
    
    def locate_hash_vertices(self, level):
        return self.model.hash_table.embeddings[level].weight.data.clone().detach().cpu().numpy()

    def table_grad_accum_visualizer(self):
        assert self.args.consistent
        plots = []
        hash_table = self.model.hash_table
        denoms = hash_table.denom
        table_grad_accum = hash_table.table_grad_accum
        for i in range(len(table_grad_accum)):
            fig, ax = plt.subplots()
            denom = denoms[i]
            grads = table_grad_accum[i]
            grads = grads / denom
            grads[grads.isnan()] = 0
            ax.bar(np.arange(len(grads)), grads.detach().cpu().sort()[0].numpy())
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png')
            im = Image.open(img_buf)
            plots.append(im)
            plt.close(fig)

        return plots

    
    def mlp_space_visualizer(self):
        x, y = next(iter(self.loader))
        # Get hashed coordinates (self.args.n_levels different levels)
        hash_vals = self.model.hash_table(x.to(self.args.device))[0]
        for i in range(1):
            hash_min_x = hash_vals[:, 2*i].min().item()
            hash_max_x = hash_vals[:, 2*i].max().item()
            hash_min_y = hash_vals[:, 2*i+1].min().item()
            hash_max_y = hash_vals[:, 2*i+1].max().item()

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
            x_hashed_mesh_filler = torch.zeros_like(hash_vals).to(self.args.device)
            x_hashed_mesh_filler[:, 2*i:2*i+2] = x_hashed_mesh
            
            # Generate learned INR  
            preds = self.model.net(x_hashed_mesh_filler)
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
            pred = y.detach().cpu().numpy()
        else:
            pred = model(x.to(device))[0].clamp(-1,1)
            pred = pred.detach().cpu().numpy()
    
    # reshape, convert to image, return
    pred = pred.reshape(img_shape)
    if np.min(pred) < 0:
        pred = pred / 2 + 0.5

    return pred


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