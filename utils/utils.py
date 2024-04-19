import random
import os
import pprint

import torch
import numpy as np
import wandb
from pathlib import Path

from data import ImageFile, PointCloud, BigImageFile, CIFAR10, CameraDataset, WaveFile


BASE_PATH = "./data"


class AbstractTrainer:
    def __init__(self):
        self.model = None
    
    @classmethod
    def train(self):
        return NotImplemented

    def get_model(self):
        return NotImplemented

    def get_model_size(self):
        if self.model is None:
            print("Model is not initialized.")
            return None
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_model(self):
        return NotImplemented

    def generate_image(self):
        return NotImplemented


def get_data(dataset, batch_size, coord_mode='0', idx=None, p1=None):
    if dataset == "FFHQ":
        # Total instances = 70000
        data_idx = random.randint(1, 70001) if idx is None else idx
        data_idx = str(data_idx).zfill(5)
        dataset = ImageFile("%s/images1024x1024/%s.png" % (BASE_PATH, data_idx), coord_mode=coord_mode)
    elif dataset == "wave":
        data_idx = random.randint(0, 2) if idx is None else idx
        dataset = WaveFile(1024, coord_mode=coord_mode, wave_type=data_idx, p1=p1)
    elif dataset == "cifar10":
        data_idx = random.randint(0, 1000) if idx is None else idx
        dataset = CIFAR10("%s/CIFAR10" % BASE_PATH, data_idx) 
    elif dataset == "celebA":
        data_idx = random.randint(1, 202600) if idx is None else idx
        data_idx = str(data_idx).zfill(6)
        dataset = ImageFile("%s/celeba/celeba/img_align_celeba/%s.jpg" % (BASE_PATH, data_idx), coord_mode=coord_mode)  # For Paperspace
    elif dataset == "kodak":
        data_idx = random.randint(1, 25) if idx is None else idx
        data_idx = str(data_idx).zfill(2)
        dataset = ImageFile("%s/kodak/kodim%s.png" % (BASE_PATH, data_idx), coord_mode=coord_mode)
    elif dataset == "ImageNet":
        # Total instances = 100000
        data_idx = random.randint(1, 100001) if idx is None else idx
        data_idx = str(data_idx).zfill(8)
        dataset = ImageFile("/mnt/share/ILSVRC/Data//CLS-LOC/test/ILSVRC2012_test_%s.JPEG" % (data_idx), normalize=True, coord_mode=coord_mode)  # For Paperspace
        #dataset = ImageFile("/home/mnt/data/imagenet/val/ILSVRC2012_val_%s.JPEG" % (data_idx), normalize=True)  # For HKU
    elif dataset == "pluto":
        # 16,000,000 is the max number of pixels two A5000 can handle in one batch for L16_nfeat18_featdim2
        dataset = BigImageFile("%s/megapixels/pluto.png" % (BASE_PATH), max_coords=16000000, coord_mode=coord_mode)
        data_idx = "0"
    elif dataset == "sdf.armadillo":
        dataset = PointCloud("%s/sdf/Armadillo.xyz" % (BASE_PATH), batch_size)
        data_idx = "0"
    elif dataset == "cameraman":
        dataset = CameraDataset(side_length=512, normalize=True)
        data_idx = "0"

    return dataset, data_idx


def set_seeds(seed):
    torch.manual_seed(seed) 
    random.seed(seed)
    np.random.seed(seed)


def setup_wandb(args, experiment_name):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args,
        group=args.wandb_group,
        name=experiment_name
    )

    # Save ENV variables
    with (Path(wandb.run.dir) / "env.txt").open("wt") as f:
        pprint.pprint(dict(os.environ), f)

    # Define path where model will be saved
    args.model_path = Path(wandb.run.dir) / "model.pt"