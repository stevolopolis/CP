import matplotlib.pyplot as plt
import numpy as np


def plot_signal_with_ema(sample, signal, ema, peaks=None, analytical_segments=None, save_path=None):
    # Plot signal with ema
    plt.plot(sample.cpu().numpy(), signal.cpu().numpy(), label='signal', linewidth=1, color='b')
    plt.plot(sample.cpu().numpy(), ema, label='ema', linewidth=1, color='orange')
    plt.legend(loc='upper right')

    if peaks is not None:
        for peak in peaks:
            plt.axvline(x=peak, color='r', linestyle='--')

    if analytical_segments is not None:
        slopes, b, knots = analytical_segments
        knots = knots.cpu().numpy().astype(np.int32)

        for i in range(len(slopes)):
            plt.plot(sample[knots[i]:knots[i+1]].cpu().numpy(), slopes[i] * sample[knots[i]:knots[i+1]].cpu().numpy() + b[i], label='segment %s' % i, linewidth=1, color='g')

    plt.savefig(save_path)
    print("Saved to %s" % save_path)
    plt.close()


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
    return ema


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


def convert_data_point_to_piece_idx(data_points, knot_idx):
    piece_idx = []
    knot_counter = 0
    
    for data_point in data_points:
        while not (data_point >= knot_idx[knot_counter] and data_point < knot_idx[knot_counter+1]):
            if knot_counter == len(knot_idx)-2:
                break
            knot_counter += 1

        piece_idx.append(knot_counter)

    if len(piece_idx) < len(data_points):
        raise ValueError("Some data points are not within the range of the knots")
        
    return piece_idx


# def coeff2diner(sample, coeff, freqs):
#     coord = torch.tensor([coord for coord in range(len(sample))])
#     coord_perm = torch.zeros_like(coord)
#     for i, freq in enumerate(freqs):
#         period = abs(2*math.pi/freq)
#         n_coord_per_period = int((period / (torch.max(sample)-torch.min(sample))) * len(sample))
#         n = 0
#         for j in tqdm(range(n_coord_per_period)):
#             idx = torch.tensor([id for id in range(j, len(sample), n_coord_per_period)])
#             coord_perm[n : n+len(idx)] = coord[idx]
#             n += len(idx)
    
#     return coord_perm


# def permute_signal(signal, coord_perm):
#     signal_perm = signal[coord_perm]

#     fig, ax = plt.subplots()
#     signal_plot = ax.plot(sample.cpu().numpy(), signal.cpu().numpy(), label='signal', linewidth=1, color='b')
#     #signal_perm_plot = ax.plot(sample.cpu().numpy(), signal_perm.detach().cpu().numpy(), label='perm_signal', linewidth=1, color='orange')
#     plt.show()