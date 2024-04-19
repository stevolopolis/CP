import numpy as np
import torch
import random
from tqdm import tqdm

from utils import *
from scorers import *
from models import *


# Set random seed
random.seed(21)
torch.manual_seed(21)
np.random.seed(21)

# Model parameters
MODEL = 'ngp'

# Training parameters
n_samples = 50000
n = 1000
epoch = 25000

# Animation parameters
nframes = 30

print("generating samples...")
sample = torch.tensor(np.linspace(0, 1, n_samples)).to(torch.float32).to("cuda")

train_loops = False
n_pieces = 50
ema_window = n_samples // 10
dynamic_pieces = True

for trial in range(0, 25):
    model_path = f"results/{MODEL}/{MODEL}_{trial}" 
    analytical_save_path = "vis/toy_signal/analytical/%s" % trial
    empirical_save_path = "vis/toy_signal/empirical/%s" % trial
    create_subdirectories(model_path)
    create_subdirectories(analytical_save_path)
    create_subdirectories(empirical_save_path)

    # Generate signal
    signal, knot_idx, slopes, b = generate_piecewise_signal(sample, n, seed=trial)
    knots = sample[knot_idx]

    # Save data & configs
    save_data(sample.cpu().numpy(), signal.cpu().numpy(), f"{model_path}/data.npy")

    # Calculate similarity score with modified ramer-douglas-peucker
    rdp_scores, rdp_hs, rdp_bs, rdp_points = rdp_scorer(sample, signal, n_samples, slopes, b, knot_idx, n_pieces, return_points=True)
    rdp_score, rdp_anal_score = rdp_scores
    rdp_h, rdp_anal_h = rdp_hs
    rdp_b, rdp_anal_b = rdp_bs
    print(f"RDP similarity score: {rdp_score}")
    print(f"RDP analytical similarity score: {rdp_anal_score}")
    save_vals([rdp_score], f"{analytical_save_path}/rdp{n_pieces}_score.txt")
    save_vals([rdp_anal_score], f"{analytical_save_path}/rdpAnal{n_pieces}_score.txt")

    # Calculate similarity score with EMA-based turning points
    # ema_score, ema_h, ema_b, ema = ema_scorer(sample, signal, n_samples, slopes, b, knot_idx, n_pieces, ema_window=ema_window, return_points=True)
    # print(f"EMA similarity score: {ema_score}")
    # save_score(ema_score, f"{analytical_save_path}/ema{n_pieces}_score.txt")

    # plot_signal(sample, signal, "target", color='cornflowerblue')
    # plot_signal(sample, ema, "ema", color='orange')
    # plot_with_points(rdp_points, save_path=f"{analytical_save_path}/segs{n_pieces}.png")
    # plot_analytical(sample,
    #                 peaks=None, #turning_points/n_samples,
    #                 analytical_segments=(pred_hs, pred_bs, turning_points),
    #                 save_path=analytical_save_path)
    

    if train_loops:
        model, configs, model_loss, model_preds = trainer(MODEL, sample, signal, epoch, nframes)
        
        # Animate model predictions
        animate_model_preds(sample, signal, model_preds, nframes, f"{empirical_save_path}/predictions.mp4")
        # Save model configs
        save_configs(configs, f"{model_path}/configs.json")
        # Save model loss
        save_vals([model_loss], f"{empirical_save_path}/loss.txt")
        # Save model weights
        torch.save(model.state_dict(), f"{model_path}/weights.pth")
        print(f"model weights saved at {model_path}/weights.pth")

