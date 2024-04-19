from argparse import ArgumentParser
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import wandb

from utils import *

MEGAPIXELS = ["pluto", "tokyo", "mars"]


def experiment_factory(dataset, args):
    if args.exp_type == 1:
        dataset.disconnected_double_patch(0.1, exp_seed=args.exp_seed)
        return "disconnected_patch_data%s_%s_seed%s_res%s" \
            % (args.exp_data_idx, args.exp_p1, args.exp_seed, args.base_resolution)
    elif args.exp_type == 10:
        dataset.scramble_data()
        dataset.disconnected_double_patch(0.1, exp_seed=args.exp_seed)
        return "disconnected_patch_scrambled_data%s_%s_seed%s_res%s" \
            % (args.exp_data_idx, args.exp_p1, args.exp_seed, args.base_resolution)
    elif args.exp_type == 0:
        return "original_data%s_%s_seed%s_res%s" \
            % (args.exp_data_idx, args.exp_p1, args.exp_seed, args.base_resolution)
    else:
        return "unknown_type"

    
    # Scramble the data randomly (same color dist, different locations)
    #dataset.scramble_data()
    # Blackout a patch of the image (dim of patch = 0.1 * min(H, W))
    #dataset.local_decolorize(0.5)
    # Blackout the entire image 
    #dataset.global_decolorize()
    # Shift the color of a patch the image (dim of patch = 0.1 * min(H, W), color shift = 0.1 * original color)
    #dataset.local_color_shift(0.1, 0.75)
    # Shift the color of the entire image (color shift = 0.1 * original color)
    #dataset.global_color_shift(0.75)
    # Two disconnected color pathces (path_size)
    #dataset.disconnected_double_patch(0.1)




def main():
    # Set random seed
    set_seeds(10000)

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("--model")
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

    # make dataset and loader
    dataset, data_idx = get_data(args.dataset, args.batch_size, idx=args.exp_data_idx, coord_mode=args.coord_mode, p1=args.exp_p1)
    data_shape = dataset.get_data_shape()
    print(data_shape)

    experiment_name = experiment_factory(dataset, args)

    args.save_dir = os.path.join(args.save_dir, experiment_name)
    args.log_dir = os.path.join(args.log_dir, experiment_name)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Initialize wandb experiment
    if args.use_wandb:
        setup_wandb(args, experiment_name)

    trainer = MODE.Trainer(dataset, data_shape, args)
    print("Model parameters: ", trainer.get_model_size())

    if args.dataset in MEGAPIXELS:
        trainer.train_megapixels()
    else:
        trainer.train()

    if args.use_wandb:
        wandb.save(str(args.model_path.absolute()), base_path=wandb.run.dir, policy="live")
        wandb.finish()


if __name__ == "__main__":
    main()
    
