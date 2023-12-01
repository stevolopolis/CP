import torch
import numpy as np

from argparse import ArgumentParser
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import wandb
from pathlib import Path
import pprint
import random

from data import ImageFile, PointCloud, BigImageFile, CIFAR10, CameraDataset
from loguru import logger


MEGAPIXELS = ["pluto", "tokyo", "mars"]
BASE_PATH = os.environ["BASE_PATH"]

random.seed(21)


def get_data(dataset, batch_size, coord_mode='0', idx=None):
    if dataset == "FFHQ":
        # Total instances = 70000
        data_idx = random.randint(1, 70001) if idx is None else idx
        data_idx = str(data_idx).zfill(5)
        dataset = ImageFile("%s/images1024x1024/%s.png" % (BASE_PATH, data_idx), coord_mode=coord_mode)
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


def main():
    # Set random seed (default: 21)
    torch.manual_seed(10000) 
    random.seed(10000)
    np.random.seed(10000)

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("model")
    common_arg = parser.parse_args()
    
    # Import model specific packages
    if common_arg.model == "diner":
        # Diner experiment package
        import diner_experiments.utils as MODE
    elif common_arg.model == "ngp":
        # NGP experiment package
        import ngp_experiments.utils as MODE

    # Load model specific config file    
    args = MODE.load_config(common_arg.config_file)
    model_type = args.model

    for idx in range(20, 19, -1):
        for run in range(1):
            # make dataset and loader
            dataset, data_idx = get_data(args.dataset, args.batch_size, idx=idx, coord_mode=args.coord_mode)
            data_shape = dataset.get_data_shape()
            print(data_shape)

            if common_arg.model in ['diner', 'siren']:
                experiment_name = "%s_%s%s_run%s_layer%s_hidden%s_w_0%s" % (model_type, args.dataset, data_idx, run, args.n_layers, args.hidden_dim, args.w_0)
            elif common_arg.model in ['ngp']:
                experiment_name = "%s_%s%s_L%s_nfeat%s_nlayer%s_hidden%s_%s-%s_v3" % (model_type,
                                                                                    args.dataset,
                                                                                    data_idx,
                                                                                    args.n_levels,
                                                                                    args.log2_n_features,
                                                                                    args.n_layers,
                                                                                    args.hidden_dim,
                                                                                    args.base_resolution,
                                                                                    args.finest_resolution)

            args.save_dir = os.path.join(args.save_dir, experiment_name)
            args.log_dir = os.path.join(args.log_dir, experiment_name)

            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)

            if args.use_wandb:
                # Initialize wandb experiment
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    config=args,
                    group="train_image",
                    name=experiment_name
                )

                # Save ENV variables
                with (Path(wandb.run.dir) / "env.txt").open("wt") as f:
                    pprint.pprint(dict(os.environ), f)

                # Define path where model will be saved
                args.model_path = Path(wandb.run.dir) / "model.pt"

            # trainer = siren.Trainer(dataset, data_shape, args)
            trainer = MODE.Trainer(dataset, data_shape, args)
            print("Model parameters: ", trainer.get_model_size())

            if args.dataset in MEGAPIXELS:
                trainer.train_megapixels()
            else:
                trainer.train()

            '''# save model state dict and represented image as np array
            if args.dataset in MEGAPIXELS:
                predicted_img = trainer.generate_megaimage(gt=False)
            else:
                predicted_img = trainer.generate_image(gt=False)

            predicted_img.save(os.path.join(args.save_dir, "output.png"))
            #trainer.save_model()'''

            if args.use_wandb:
                wandb.save(str(args.model_path.absolute()), base_path=wandb.run.dir, policy="live")
                wandb.finish()

            # input("Continue?")


if __name__ == "__main__":
    main()
    
