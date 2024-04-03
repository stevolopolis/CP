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
from models.ngp import NGP

from sim_score import *
from data import generate_fourier_signal, generate_piecewise_signal
from misc import exponential_moving_average, find_turning_points, plot_signal_with_ema, convert_data_point_to_piece_idx


# Set random seed
random.seed(21)
torch.manual_seed(21)
np.random.seed(21)

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
elif MODEL == 'linear':
    model = LinearModel().to("cuda")
    optim = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-2)
elif MODEL == "ngp":
    Config = namedtuple("config", ["NET"])
    NetworkConfig = namedtuple("NET", ["dim_hidden", "n_levels", "feature_dim", "log2_n_features", "base_resolution", "finest_resolution", "num_layers"])
    c_net = NetworkConfig(dim_hidden=hidden_dim, n_levels=5, feature_dim=4, log2_n_features=10, base_resolution=10, finest_resolution=1, num_layers=n_layers)
    c = Config(NET=c_net)
    model = NGP(dim_in, dim_out, n_samples, c).to("cuda")
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-5)


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
