import matplotlib.pyplot as plt
import pandas as pd
import torch

from hash_visualizer import load_model_and_configs


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
    pass


def single_level_collision_rate(resolution, hash_table_size, dims):
    """Calculate the collision rate of a given resolution"""
    return (resolution / hash_table_size) ** dims


def multi_level_collision_rate(resolution, hash_table_size, dims):
    pass



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


def plot_collision(data, save_path, hash_table_size=128):
    collision_rates = [single_level_collision_rate(resolution, hash_table_size, 2) for resolution in data.index]

    fig, ax = plt.subplots()
    ax.plot(data.index.to_list(), collision_rates, color='darkorange', label='Collision Rate')
    ax.axhline(y=1, color='darkorange', linestyle='--', label='One-to-one hashing limit')
    ax.set_ylabel('Collision Rate')
    ax.set_xlabel('Finest Resolution')

    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    MODEL_PATH = 'vis/collision'
    MODEL_WEIGHTS_PATH = "%s/weights.pth" % (MODEL_PATH)
    MODEL_PRED_PATH = "%s/ngp_pred.png" % MODEL_PATH
    MODEL_HASH_PATH = "%s/visualization.png" % MODEL_PATH
    EXP_VIS_PATH = "vis/experiments"

    model_name = 'ngp_single_level'
    if model_name == 'ngp_single_level':
        hash_table_size = 128
    elif model_name == 'ngp_multi_level':
        hash_table_size = 64
    data = load_data(f'{model_name}_psnr.csv')
    summary = summarize_data(data)
    plot_psnr(summary, f'{model_name}_psnr_vs_finest_resolution.png', hash_table_size=hash_table_size)


    model, x, y, configs = load_model_and_configs(f"{MODEL_PATH}/data.npy", f"{MODEL_PATH}/configs.json", MODEL_WEIGHTS_PATH, MODEL)
