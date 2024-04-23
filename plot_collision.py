import matplotlib.pyplot as plt
import pandas as pd
import torch

from utils import *
from scorers import *
from models import *
from hash_visualizer import *


def load_data(path):
    """Load csv into pandas dataframe"""
    return pd.read_csv(path)


def summarize_data(data):
    """Calculate the mean and std psnr of each finest_resolution"""
    return data.groupby('finest_resolution').agg({'psnr': ['mean', 'std']})


def load_model(model_path):
    """Load model from path"""
    return torch.load(model_path)


def get_hash_table_size(model):
    sizes = {}
    hash_table = model.hash_table
    for level in range(hash_table.n_levels):
        sizes[level] = hash_table.embeddings[level].weight.size(0)

    return sizes


def get_resolution(model):
    resolutions = {}
    for level in range(model.hash_table.n_levels):
        resolution = (torch.floor(model.hash_table.base_resolution * model.hash_table.b**level) + 1) ** model.dim_in
        resolutions[level] = resolution

    return resolutions


def single_level_collision_rate(resolution, hash_table_size):
    """Calculate the collision rate of a given resolution"""
    return resolution / hash_table_size


def get_theoretical_collision_rate(model):
    """Calculate the collision rate of a multi-level model"""
    collision_rate = {}
    hash_table_sizes = get_hash_table_size(model)
    for level, size in hash_table_sizes.items():
        resolution = (torch.floor(model.hash_table.base_resolution * model.hash_table.b**level) + 1) ** model.dim_in
        collision_rate[level] = single_level_collision_rate(resolution, size)

    return collision_rate


def get_empirical_collision_rate(model, x):
    non_interp_hash = model.hash_table(x, interp=False)
    unique_hashes, unique_counts = torch.unique(non_interp_hash, dim=0, return_counts=True)
    coordinates_to_hash_ratio = x.size(0) / unique_hashes.size(0)
    average_collision_rate = torch.mean(unique_counts.float()).item()

    return coordinates_to_hash_ratio, average_collision_rate


def plot_psnr(data, save_path, hash_table_size=128):
    """
    Line plot of the mean psnr of each finest_resolution with error bars.
    Red vertical line at the one-to-one hashing limit (default: 64x64)    
    """
    fig, ax = plt.subplots()
    data['psnr']['mean'].plot(ax=ax, yerr=data['psnr']['std'], marker='o', color='salmon', label='Mean PSNR')
    ax.axvline(x=hash_table_size, color='cadetblue', linestyle='--', label='Hash table size')
    ax.set_xlabel('Finest Resolution')
    ax.set_ylabel('PSNR')
    ax.set_title('PSNR vs Finest Resolution')

    ax.legend()
    plt.savefig(save_path)
    plt.close()


def plot_key_against_values(dictionary, xlabel, ylabel, title, save_path):
    fig, ax = plt.subplots()
    ax.plot(dictionary.keys(), dictionary.values(), marker='o', color='salmon')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    # model_name = 'ngp_single_level'
    # if model_name == 'ngp_single_level':
    #     hash_table_size = 128
    # elif model_name == 'ngp_multi_level':
    #     hash_table_size = 64
    # data = load_data(f'{model_name}_psnr.csv')
    # summary = summarize_data(data)
    # plot_psnr(summary, f'{model_name}_psnr_vs_finest_resolution.png', hash_table_size=hash_table_size)

    MODEL = "ngp_multilevel_2d"
    MODEL_NAME = f"{MODEL}"
    BASE_PATH = f'vis/collision_error/{MODEL_NAME}'
    DATA_PATH = '../data/kodak'
    EMPIRICAL_PATH = f"{BASE_PATH}/empirical"
    FIGURE_PATH = f"{BASE_PATH}/figures"
    DEVICE = 'cuda:0'

    resolution_logger = {}
    hash_table_size_logger = {}
    theoretical_collision_rate_logger = {}
    coord_to_hash_logger = {}
    collision_rate_logger = {}
    for img_idx in range(1, 5):
        model_path = f"{EMPIRICAL_PATH}/{img_idx}"
        data_path = f"{DATA_PATH}/kodim{str(img_idx).zfill(2)}.png"

        dataLoader = ImageFile(data_path, coord_mode=0)
        H, W, C = dataLoader.get_data_shape()
        sample, signal = next(iter(dataLoader))
        sample = sample.to(DEVICE)
        signal = signal.to(DEVICE)

        for finest_resolution in range(50, 500, 50):
            if finest_resolution not in coord_to_hash_logger:
                coord_to_hash_logger[finest_resolution] = []
            if finest_resolution not in collision_rate_logger:
                collision_rate_logger[finest_resolution] = []
                
            for seed in range(3):
                config_path = f"{model_path}/configs_{finest_resolution}.json"
                weights_path = f"{model_path}/weights_{img_idx}_{finest_resolution}_{seed}.pth"

                model, x, y, configs = load_model_and_configs(None, config_path, weights_path, MODEL, data_size=(H, W), device=DEVICE)
                hash_table_sizes = get_hash_table_size(model)
                resolutions = get_resolution(model)
                empirical_coord_to_hash_ratio, empirical_hash_collision_ratio = get_empirical_collision_rate(model, sample)
                theorertical_collision_rate = get_theoretical_collision_rate(model)
                coord_to_hash_logger[finest_resolution].append(empirical_coord_to_hash_ratio) 
                collision_rate_logger[finest_resolution].append(empirical_hash_collision_ratio)

                if finest_resolution not in hash_table_size_logger:
                    hash_table_size_logger[finest_resolution] = sum(hash_table_sizes.values())  
                if finest_resolution not in resolution_logger:
                    resolution_logger[finest_resolution] = sum(resolutions.values())  
                
                for level in theorertical_collision_rate.keys():
                    if level not in theoretical_collision_rate_logger:
                        theoretical_collision_rate_logger[level] = {}
                    if finest_resolution not in theoretical_collision_rate_logger[level]:
                        theoretical_collision_rate_logger[level][finest_resolution] = []
                    theoretical_collision_rate_logger[level][finest_resolution].append(theorertical_collision_rate[level])

    plot_key_against_values(coord_to_hash_logger, "Resolution", "Coordinates to hash ratio", "Coordinates to hash ratio VS Resolution", f"{FIGURE_PATH}/coord_to_hash_ratio.png")
    plot_key_against_values(collision_rate_logger, "Resolution", "Collision ratio", "Collision ratio VS Resolution", f"{FIGURE_PATH}/collision_rate.png")
    for level in theoretical_collision_rate_logger.keys():
        plot_key_against_values(theoretical_collision_rate_logger[level], "Resolution", "Collision ratio", "Collision ratio VS Resolution", f"{FIGURE_PATH}/theoretical_collision_rate_{level}.png")
    plot_key_against_values(hash_table_size_logger, "Total number of hash entries", "Resolution", "Hash entries VS Resolution", f"{FIGURE_PATH}/hash_table_size.png")
    plot_key_against_values(resolution_logger, "Total number of grid vertices", "Resolution", "Grid vertices VS Resolution", f"{FIGURE_PATH}/resolution.png")
