import matplotlib.pyplot as plt
import numpy as np

import os
import json

from collections import namedtuple


# ========================
# Directory management functions
# ========================

def create_subdirectories(directory_path, is_file=False):
    """
    Create directories for each subdirectory that doesn't exist.

    Args:
    directory_path (str): The directory path.

    Returns:
    None
    """
    # Split the directory path into individual directory names
    directories = directory_path.split(os.path.sep)
    if is_file:
        directories = directories[:-1]

    # Initialize a variable to keep track of the current directory being created
    current_directory = ''

    # Iterate through each directory in the path
    for directory in directories:
        # Append the current directory to the path
        current_directory = os.path.join(current_directory, directory)
        
        # Check if the current directory exists
        if not os.path.exists(current_directory):
            # If it doesn't exist, create the directory
            os.makedirs(current_directory)



# ========================
# Data processing functions
# ========================

def save_vals(vals, path):
    """Save a list of vals to file."""
    with open(path, "w") as f:
        for val in vals:
            f.write(str(val) + "\n")
    print("vals saved at %s" % path)


def load_vals(path):
    """Load a list of vals from a file."""
    with open(path, "r") as f:
        vals = f.readlines()
    return [float(val) for val in vals]


def save_data(x, y, path):
    """Save the data to a file."""
    np.save(path, np.array([x, y]))
    print("data saved at %s" % path)


def load_data(path):
    """Load the data from a file."""
    data = np.load(path)
    return data[0], data[1]
    

def save_configs(configs, path):
    """Save the configs to a file."""
    j = json.dumps(configs.NET._asdict())
    with open(path, "w") as f:
        json.dump(j, f)
    print("configs saved at %s" % path)


def convert(json):
    """Convert a json object to a namedtuple."""
    for key, value in json.items():
        if isinstance(value, dict):
            json[key] = convert(value)

    return namedtuple('NET', json.keys())(**json)

def load_configs(path):
    """Load the configs from a file and store as a namedtuple."""
    Config = namedtuple("config", ["NET"])

    with open(path, "r") as f:
        j = json.load(f)
        
    return Config(convert(json.loads(j)))


# ========================
# Plotting functions
# ========================

def plot_signal(sample, signal, label, color='b', save_path=None):
    """Plot the signal."""
    # Plot signal
    plt.plot(sample.cpu().numpy(), signal.cpu().numpy(), label=label, color=color)
    
    if save_path is not None:
        plt.legend(loc='upper right')
        plt.savefig(save_path)
        print("Saved to %s" % save_path)
        plt.close()


def plot_analytical(sample, peaks=None, analytical_segments=None, save_path=None):
    """Plot the analytical segments."""
    sample = sample.cpu().numpy()

    if peaks is not None:
        for peak in peaks:
            plt.axvline(x=peak, color='r', linestyle='--')

    if analytical_segments is not None:
        slopes, b, knots = analytical_segments
        knots = knots.cpu().numpy().astype(np.int32)

        for i in range(len(slopes)):
            if i == 0:
                plt.plot(sample[knots[i]:knots[i+1]], slopes[i] * sample[knots[i]:knots[i+1]] + b[i], label='analytical', linewidth=1, color='g')
            else:
                plt.plot(sample[knots[i]:knots[i+1]], slopes[i] * sample[knots[i]:knots[i+1]] + b[i], linewidth=1, color='g')


    if save_path is not None:
        plt.legend(loc='upper right')
        plt.savefig(save_path)
        print("Saved to %s" % save_path)
        plt.close()


def plot_with_points(points, save_path=None):
    """
    Plot a polyline with points.
    """    
    # Extract the x and y coordinates of the points
    x, y = zip(*points)
    
    # Plot the polyline
    plt.plot(x, y, label='ramer_douglas_peucker', linewidth=1, color='red')
    
    # Display the plot
    if save_path is not None:
        plt.legend(loc='upper right')
        plt.savefig(save_path)
        print("Saved to %s" % save_path)
        plt.close()

