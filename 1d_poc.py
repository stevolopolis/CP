import numpy as np
import torch
import math
import random
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from models.mlp import MLP

# Model parameters
MODEL = 'relu'
dim_in = 1
dim_out = 1
hidden_dim = 32
n_layers = 2
w0 = 30.0
w0_initial = 30.0

# Training parameters
n = 100
epoch = 100000

# Animation parameters
nframes = 30

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


def generate_piecewise_signal(sample, n):
    signal = torch.zeros_like(sample).to("cuda")
    print("generating signal...")
    knots = (torch.rand(n) * len(sample)).int()
    knots = torch.sort(knots)[0]
    slopes = torch.randn(n)
    init_y = torch.randn(1)
    b = []
    for i in range(n):
        if i == 0:
            signal[:knots[i]] = init_y
            b.append(init_y)
        elif i == n-1:
            signal[knots[i-1]:] = slopes[i-1] * (sample[knots[i-1]:] - sample[knots[i-1]]) + signal[knots[i-1]-1]
            b.append(signal[knots[i-1]-1] - slopes[i-1] * sample[knots[i-1]-1])
        else:
            signal[knots[i-1]:knots[i]] = slopes[i-1] * (sample[knots[i-1]:knots[i]] - sample[knots[i-1]]) + signal[knots[i-1]-1]
            b.append(signal[knots[i-1]-1] - slopes[i-1] * sample[knots[i-1]-1])

    return signal / torch.max(torch.abs(signal)), knots, slopes, torch.tensor(b)


def weighted_variance(arr, knots, squared=False):
    if squared:
        weights = [abs(knots[i+1]**2 - knots[i]**2) for i in range(len(knots)-1)]
    else:
        weights = [abs(knots[i+1] - knots[i]) for i in range(len(knots)-1)]    

    e = arr @ weights
    e_squared = torch.pow(arr, 2) @ weights

    return e_squared - e**2


def signal_similarity_score(slopes, b, n_knots, n_pieces):
    knots_per_pieces = n_knots // n_pieces
    h_var = []
    b_var = []
    for i in range(n_pieces):
        if i == n_pieces-1:
            h_var.append(weighted_variance(slopes[i*knots_per_pieces:], squared=True))
            b_var.append(weighted_variance(b[i*knots_per_pieces:]))
        else:
            h_var.append(weighted_variance(slopes[i*knots_per_pieces:(i+1)*knots_per_pieces], squared=True))
            b_var.append(weighted_variance(b[i*knots_per_pieces:(i+1)*knots_per_pieces]))

    return h_var, b_var


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


if MODEL == 'relu':
    Config = namedtuple("config", ["NET"])
    NetworkConfig = namedtuple("NET", ["num_layers", "dim_hidden", "use_bias"])
    c_net = NetworkConfig(num_layers=n_layers, dim_hidden=hidden_dim, use_bias=True)
    c = Config(NET=c_net)
    model = MLP(dim_in, dim_out, c).to("cuda")
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epoch, eta_min=1e-3)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))


print("generating samples...")
sample = torch.tensor(np.linspace(0, 1, 50000)).to(torch.float32).to("cuda")

# Generate signal
#signal, coeff, freqs = generate_fourier_signal(sample, n)
signal = generate_piecewise_signal(sample, n)

# # Perm signal 
# print("calculating permutation matrix")
# coord_perm = coeff2diner(sample, coeff, freqs)
# print('plotting permuted signal')
# permute_signal(signal, coord_perm)

# input()


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
    loss = ((model_prediction - signal) ** 2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()
    scheduler.step()

    if i % int(epoch / nframes) == 0:
        model_pred_history.append(model_prediction.detach().cpu().numpy())

    tbar.set_description(f'loss: {loss.item()}')


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
ani.save("toy_animation_%s.mp4" % MODEL, writer=writervideo)
plt.close()