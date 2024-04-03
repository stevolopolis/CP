import numpy as np
import torch
import math
import random
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from models.mlp import MLP
from models.linear import LinearModel

# Set random seed
random.seed(21)
torch.manual_seed(21)
np.random.seed(21)


def generate_fourier_signal(sample, n):
    coefficients = []
    phase = []
    freqs = []
    signal = torch.zeros_like(sample).to("cuda")
    print("generating signal...")
    for i in range(1, n+1):
        coeff = random.gauss(0.0, 1.0)
        freq = 2 * math.pi * i
        signal += random.gauss(0.0, 1.0) * torch.sin(freq * sample) / n
        coefficients.append(coeff)
        freqs.append(freq)

    return signal, coefficients, freqs


def generate_piecewise_signal(sample, n, seed=42):
    torch.manual_seed(seed)
    signal = torch.zeros_like(sample).to("cuda")
    print("generating signal...")
    knots = (torch.rand(n+1) * len(sample)).int()
    knots[0] = 0
    knots[-1] = len(sample)-1
    knots = torch.sort(knots)[0]
    slopes = torch.randn(n)
    init_y = torch.randn(1)
    b = []
    for i in range(n+1):
        if i == 0:
            signal[:knots[i]] = init_y
        elif i == n-1:
            signal[knots[i-1]:] = slopes[i-1] * (sample[knots[i-1]:] - sample[knots[i-1]]) + signal[knots[i-1]-1]
            b.append(signal[knots[i-1]-1] - slopes[i-1] * sample[knots[i-1]-1])
        elif i == 1:
            signal[knots[i-1]:knots[i]] = slopes[i-1] * (sample[knots[i-1]:knots[i]] - sample[knots[i-1]]) + init_y.to('cuda')
            b.append(init_y)
        else:
            signal[knots[i-1]:knots[i]] = slopes[i-1] * (sample[knots[i-1]:knots[i]] - sample[knots[i-1]]) + signal[knots[i-1]-1]
            b.append(signal[knots[i-1]-1] - slopes[i-1] * sample[knots[i-1]-1])

    return signal, knots, slopes, torch.tensor(b)


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


def get_weighted_preds(slopes, b, knots):
    weights = [knots[i+1] - knots[i] for i in range(len(knots)-1)]
    sq_weights = [knots[i+1]**2 - knots[i]**2 for i in range(len(knots)-1)]
    cube_weights = [knots[i+1]**3 - knots[i]**3 for i in range(len(knots)-1)]

    slopes = slopes.to("cuda")
    b = b.to("cuda")
    weights = torch.tensor(weights).to("cuda")
    sq_weights = torch.tensor(sq_weights).to("cuda")
    cube_weights = torch.tensor(cube_weights).to("cuda")

    mean_b = b @ weights
    sq_mean_b = b @ sq_weights
    sq_mean_slope = slopes @ sq_weights
    cube_mean_slope = slopes @ cube_weights
    norm_factor = 1 - (3/4 * sq_weights.sum()**2 / weights.sum() / cube_weights.sum())

    slope_remainders = (mean_b + 1/2*sq_mean_slope) / weights.sum()
    b_remainders = (cube_mean_slope + 3/2*sq_mean_b) / cube_weights.sum()

    pred_slopes = (cube_mean_slope + 3/2*((b - slope_remainders) @ sq_weights)) / cube_weights.sum() / norm_factor
    pred_b = (mean_b + 1/2*((slopes - b_remainders) @ sq_weights)) / weights.sum() / norm_factor 

    return pred_slopes, pred_b


def weighted_var(slopes, b, pred_slopes, pred_b, knots):
    weights = [abs(knots[i+1] - knots[i]) for i in range(len(knots)-1)]
    sq_weights = [abs(knots[i+1]**2 - knots[i]**2) for i in range(len(knots)-1)]
    cube_weights = [abs(knots[i+1]**3 - knots[i]**3) for i in range(len(knots)-1)]

    slopes = slopes.to("cuda")
    b = b.to("cuda")
    pred_slopes = pred_slopes.to("cuda")
    pred_b = pred_b.to("cuda")
    weights = torch.tensor(weights).to("cuda")
    sq_weights = torch.tensor(sq_weights).to("cuda")
    cube_weights = torch.tensor(cube_weights).to("cuda")

    e1 = (slopes - pred_slopes)**2 @ cube_weights / 3
    e2 = ((slopes - pred_slopes) * (b - pred_b)) @ sq_weights
    e3 = (b - pred_b)**2 @ weights

    return e1 + e2 + e3



def signal_similarity_score(slopes, b, knots, n_pieces):
    knots_per_pieces = len(slopes) // n_pieces
    score = []
    ranges = []
    for i in range(n_pieces):
        if i == n_pieces-1:
            pred_h, pred_b = get_weighted_preds(slopes[i*knots_per_pieces:], b[i*knots_per_pieces:], knots[i*knots_per_pieces:])
            score.append(weighted_var(slopes[i*knots_per_pieces:],
                                      b[i*knots_per_pieces:],
                                      pred_h, pred_b,
                                      knots[i*knots_per_pieces:]))
            ranges.append(knots[-1] - knots[i*knots_per_pieces])
        else:
            pred_h, pred_b = get_weighted_preds(slopes[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                                 b[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                                 knots[i*knots_per_pieces:(i+1)*knots_per_pieces+1])
            score.append(weighted_var(slopes[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                      b[i*knots_per_pieces:(i+1)*knots_per_pieces],
                                      pred_h, pred_b,
                                      knots[i*knots_per_pieces:(i+1)*knots_per_pieces+1]))
            ranges.append(knots[(i+1)*knots_per_pieces] - knots[i*knots_per_pieces])

    score_tensor = torch.tensor(score).to("cuda")
    ranges = torch.tensor(ranges).to("cuda")

    mean_score = score_tensor @ ranges / ranges.sum()

    return mean_score.item()


def signal_similarity_score_with_turning_points(slopes, b, knots, turning_points):
    score = []
    ranges = []

    pred_hs = []
    pred_bs = []
    
    for i in range(len(turning_points)-1):
        if i == len(turning_points)-1:
            pred_h, pred_b = get_weighted_preds(slopes[turning_points[i]:], b[turning_points[i]:], knots[turning_points[i]:])
            score.append(weighted_var(slopes[turning_points[i]:],
                                      b[turning_points[i]:],
                                      pred_h, pred_b,
                                      knots[turning_points[i]:]))
            ranges.append(knots[-1] - knots[turning_points[i]])
        else:
            pred_h, pred_b = get_weighted_preds(slopes[turning_points[i]:turning_points[i+1]],
                                                 b[turning_points[i]:turning_points[i+1]],
                                                 knots[turning_points[i]:turning_points[i+1]+1])
            pred_hs.append(pred_h.item())
            pred_bs.append(pred_b.item())
            score.append(weighted_var(slopes[turning_points[i]:turning_points[i+1]],
                                      b[turning_points[i]:turning_points[i+1]],
                                      pred_h, pred_b,
                                      knots[turning_points[i]:turning_points[i+1]+1]))
            ranges.append(knots[turning_points[i+1]] - knots[turning_points[i]])

    score_tensor = torch.tensor(score).to("cuda")
    ranges = torch.tensor(ranges).to("cuda")

    mean_score = score_tensor @ ranges / ranges.sum()

    return mean_score.item(), pred_hs, pred_bs


def coeff2diner(sample, coeff, freqs):
    coord = torch.tensor([coord for coord in range(len(sample))])
    coord_perm = torch.zeros_like(coord)
    for i, freq in enumerate(freqs):
        period = abs(2*math.pi/freq)
        n_coord_per_period = int((period / (torch.max(sample)-torch.min(sample))) * len(sample))
        n = 0
        for j in tqdm(range(n_coord_per_period)):
            idx = torch.tensor([id for id in range(j, len(sample), n_coord_per_period)])
            coord_perm[n : n+len(idx)] = coord[idx]
            n += len(idx)
    
    return coord_perm

def permute_signal(signal, coord_perm):
    signal_perm = signal[coord_perm]

    fig, ax = plt.subplots()
    signal_plot = ax.plot(sample.cpu().numpy(), signal.cpu().numpy(), label='signal', linewidth=1, color='b')
    #signal_perm_plot = ax.plot(sample.cpu().numpy(), signal_perm.detach().cpu().numpy(), label='perm_signal', linewidth=1, color='orange')
    plt.show()


# Model parameters
MODEL = 'relu'
dim_in = 1
dim_out = 1
hidden_dim = 64
n_layers = 5
w0 = 30.0
w0_initial = 30.0

# Training parameters
n_samples = 50000
n = 1000
epoch = 5000

# Animation parameters
nframes = 30


if MODEL == 'relu':
    Config = namedtuple("config", ["NET"])
    NetworkConfig = namedtuple("NET", ["num_layers", "dim_hidden", "use_bias"])
    c_net = NetworkConfig(num_layers=n_layers, dim_hidden=hidden_dim, use_bias=True)
    c = Config(NET=c_net)
    model = MLP(dim_in, dim_out, c).to("cuda")
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-5)


if MODEL == 'linear':
    model = LinearModel().to("cuda")
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-2)

print("Number of parameters:")
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


print("generating samples...")
sample = torch.tensor(np.linspace(0, 1, n_samples)).to(torch.float32).to("cuda")

train_loops = False
n_pieces = 7
ema_window = n_samples // 10
dynamic_pieces = True

sim_score_ls = []
loss_score_ls = []
for trial in range(10):
    # Generate signal
    #signal, coeff, freqs = generate_fourier_signal(sample, n)
    signal, knot_idx, slopes, b = generate_piecewise_signal(sample, n, seed=trial)

    # Find natural turning points with ema
    if dynamic_pieces:
        ema = exponential_moving_average(signal.cpu().numpy(), ema_window)
        peaks, degree = find_turning_points(ema)
        peaks = torch.tensor(peaks)
        degree = torch.tensor(degree)
        sorted_peak_idx = torch.argsort(degree, descending=True)
        if n_pieces < len(peaks):
            turning_points = torch.zeros(n_pieces+2)
            max_turning_points = peaks[sorted_peak_idx[:n_pieces]]
            max_turning_points = torch.sort(max_turning_points)[0]
            turning_points[1:-1] = max_turning_points
            turning_points[-1] = n_samples-1
            
            turning_points_idx = convert_data_point_to_piece_idx(turning_points, knot_idx)
        else:
            turning_points = torch.zeros(len(peaks)+2)
            turning_points[1:-1] = peaks
            turning_points[-1] = n_samples-1
            turning_points_idx = convert_data_point_to_piece_idx(turning_points, knot_idx)
    else:
        turning_points_idx = None

    # Measure signal self-similarity score
    knots = sample[knot_idx]
    if dynamic_pieces:
        score, pred_hs, pred_bs = signal_similarity_score_with_turning_points(slopes, b, knots, turning_points_idx)
    else:
        score = min([signal_similarity_score(slopes, b, knots, i) for i in range(1, 20)])
    sim_score_ls.append(score)

    analytical_save_path = "vis/analytical/toy_signal_%ssegs_%s.png" % (n_pieces, trial)
    plot_signal_with_ema(sample, signal, ema,
                         peaks=turning_points/n_samples,
                         analytical_segments=(pred_hs, pred_bs, turning_points),
                         save_path=analytical_save_path)

    # # Perm signal 
    # print("calculating permutation matrix")
    # coord_perm = coeff2diner(sample, coeff, freqs)
    # print('plotting permuted signal')
    # permute_signal(signal, coord_perm)

    # input()

    # # dummy prediction and loss
    # model_prediction = torch.mean(signal).repeat(len(sample))
    # loss = ((model_prediction - signal) ** 2).mean()
    # print(loss.item())
    # loss_score_ls.append(loss.item())

    # analytical prediction
    # h, yint = get_weighted_preds(slopes, b, knots)
    # print("mean h: %s\t mean b: %s" % (torch.mean(slopes).item(), torch.mean(b).item()))
    # print("h: %s\t b: %s" % (h.item(), yint.item()))
    # model_prediction = h * sample + yint
    # print("anal loss: %s" % ((model_prediction - signal)**2).mean().item())

    # # plot dummy prediction
    # plt.plot(sample.cpu().numpy(), signal.cpu().numpy(), label='signal', linewidth=1, color='b')
    # plt.plot(sample.cpu().numpy(), model_prediction.detach().cpu().numpy(), label='model', linewidth=1, color='orange')
    # plt.legend(loc='upper right')
    # plt.savefig("vis/toy_signal_%s_%s.png" % (MODEL, trial))
    # plt.close()


    if train_loops:
        # Initialize model weights
        if MODEL == 'relu':
            model.init_weights()

        # Initial model prediction
        model_prediction = model(sample.unsqueeze(1)).squeeze(1)

        # Plotting
        print('plotting initial signal...')
        fig, ax = plt.subplots()
        signal_plot = ax.plot(sample.cpu().numpy(), signal.cpu().numpy(), label='signal', linewidth=1, color='b')
        model_pred_plot = ax.plot(sample.cpu().numpy(), model_prediction.detach().cpu().numpy(), label='model', linewidth=1, color='orange')[0]
        ax.legend(loc='upper right')

        print("training model...")
        # Train model
        model_pred_history = []
        tbar = tqdm(range(epoch))
        for i in tbar:
            model_prediction = model(sample.unsqueeze(1)).squeeze(1)
            loss = ((model_prediction - signal)**2).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            if i % int(epoch / nframes) == 0:
                model_pred_history.append(model_prediction.detach().cpu().numpy())

            tbar.set_description(f'loss: {loss.item()}')

        loss_score_ls.append(loss.item())
        print("model loss: %s" % loss.item())
        # print("model h: %s\t model b: %s" % (model.h.item(), model.b.item()))

        # Animation
        def update(frame):
            # Update model predictions
            model_pred = model_pred_history[frame]
            model_pred_plot.set_ydata(model_pred)

            model_pred_min = model_pred.min().item()
            model_pred_max = model_pred.max().item()
            signal_min = signal.min().item()
            signal_max = signal.max().item()
            plot_min = min(model_pred_min, signal_min)
            plot_max = max(model_pred_max, signal_max)
            plot_range = plot_max - plot_min
            epsilon = plot_range * 0.1
            ax.set_ylim(plot_min - epsilon, plot_max + epsilon)

            return (signal_plot, model_pred_plot)


        print("animating...")
        ani = animation.FuncAnimation(fig=fig, func=update, frames=nframes, interval=500)
        plt.show()
        writervideo = animation.FFMpegWriter(fps=2)
        ani.save("vis/empirical/toy_animation_%s_%s.mp4" % (MODEL, trial), writer=writervideo)
        plt.close()


# Save losses and self-similarity scores
if dynamic_pieces:
    print("Saving similarity scores to logs/sim_scores_%s_knots_%s_dynamic.npy" % (MODEL, n_pieces))
    np.save("logs/sim_scores_%s_knots_%s_dynamic.npy" % (MODEL, n_pieces), np.array(sim_score_ls))
else:
    print("Saving similarity scores to logs/sim_scores_%s_knots_%s.npy" % (MODEL, n_pieces))
    np.save("logs/sim_scores_%s_knots_%s.npy" % (MODEL, n_pieces), np.array(sim_score_ls))

if train_loops:
    print("Saving losses to logs/losses_%s.npy" % MODEL)
    np.save("logs/losses_%s.npy" % MODEL, np.array(loss_score_ls))
