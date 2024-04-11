import torch

from scorers.sim_score import *


def exponential_moving_average(data, window):
    """
    Calculate the exponential moving average of a given list.

    Parameters:
    data (list): The input data.
    window (int): The size of the window for the moving average.

    Returns:
    list: Exponential moving average of the input data.
    """
    alpha = 2 / (window + 1)
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return torch.tensor(ema)


def find_turning_points(ema):
    """
    Find the turning points in a given list representing the exponential moving average.

    Parameters:
    ema (list): Exponential moving average of a given list.

    Returns:
    list: Indices of turning points.
    """
    turning_points = []
    for i in range(1, len(ema) - 1):
        if ema[i] > ema[i - 1] and ema[i] > ema[i + 1]:
            turning_points.append(i)
        elif ema[i] < ema[i - 1] and ema[i] < ema[i + 1]:
            turning_points.append(i)

    degree = []
    for j in range(len(turning_points)):
        if j == 0:
            degree.append(turning_points[j+1])
        elif j == len(turning_points)-1:
            degree.append(len(ema) - turning_points[j-1])
        else:
            degree.append(turning_points[j+1] - turning_points[j-1])

    return turning_points, degree


def ema_scorer(sample: torch.tensor, signal: torch.tensor, n_samples: int,
               slopes: torch.tensor, b: torch.tensor, knot_idx: torch.tensor,
               n_pieces: int, ema_window: int = 10, return_points=False):
    """
    Get similarity score with EMA peak finder.
    
    Parameters:
    - sample (torch.tensor): Sample data points.
    - signal (torch.tensor): Signal data points.
    - n_samples (int): Number of samples.
    - slopes (torch.tensor): Slopes of each segment of the signal.
    - b (torch.tensor): y-intercepts of each segment of the signal.
    - knot_idx (torch.tensor): Indices of the knots. Knots are the endpoints of each segment.
    - n_pieces (int): Number of segments to divide the signal into.
    - ema_window (int): The size of the window for the moving average.
    - return_points (bool): Whether to return the turning points."""
    # Convert knot indices to knot x-coordinates
    knots = sample[knot_idx]

    # Calculate the exponential moving average of the signal
    ema = exponential_moving_average(signal.cpu().numpy(), ema_window)
    # Find the turning points in the exponential moving average
    # Each turning point has a corresponding degree, which measure the distance of the peak from the neighboring peaks
    peaks, degree = find_turning_points(ema)
    peaks = torch.tensor(peaks)
    degree = torch.tensor(degree)
    # Sort the peaks by degree in descending order
    sorted_peak_idx = torch.argsort(degree, descending=True)

    if n_pieces < len(peaks) + 2:   # If there are more pieces than wanted, then we only choose the top n_pieces peaks
        turning_points = torch.zeros(n_pieces+2)
        max_turning_points = peaks[sorted_peak_idx[:n_pieces]]
        max_turning_points = torch.sort(max_turning_points)[0]
        turning_points[1:-1] = max_turning_points
        turning_points[-1] = n_samples-1
        
        turning_points_idx = convert_data_point_to_piece_idx(turning_points, knot_idx)
    else:   # If there are less pieces than wanted, then we choose all the peaks and fill the rest with the endpoints
        turning_points = torch.zeros(len(peaks)+2)
        turning_points[1:-1] = peaks
        turning_points[-1] = n_samples-1
        turning_points_idx = convert_data_point_to_piece_idx(turning_points, knot_idx)

    # Return the similarity score
    score, ema_h, ema_b = signal_similarity_score_with_turning_points(slopes, b, knots, turning_points_idx)

    if return_points:
        return score, ema_h, ema_b, ema
    
    return score, ema_h, ema_b, None