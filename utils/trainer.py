import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from models import *
from utils import *


def trainer(model_type, sample, signal, epoch, nframes, hash_vals=None):    
    # Load model
    model, optim, scheduler, configs = load_default_models(model_type, epoch=epoch)
    print("Number of parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Initialize model weights
    model.init_weights()

    # For storing model prediction history
    model_pred_history = []

    # Initial model prediction
    model_prediction = model(sample.unsqueeze(1)).squeeze(1)
    model_pred_history.append(model_prediction.detach().cpu().numpy())

    print("training model...")
    # Train model
    for i in tqdm(range(epoch)):
        # If hash_vals is provided, use it as input and skip the hash table
        if hash_vals is not None:
            model_prediction = model.net(hash_vals).squeeze(1)
        else:
            model_prediction = model(sample.unsqueeze(1)).squeeze(1)
        loss = ((model_prediction - signal)**2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        if i % int(epoch / nframes) == 0:
            model_pred_history.append(model_prediction.detach().cpu().numpy())

    print("Training completed.")
    print("Model loss: %s" % loss.item())

    return model, configs, loss.item(), model_pred_history


def animate_model_preds(sample, signal, model_pred_history, nframes, empirical_save_path):
    # Plot initial prediction
    fig, ax = plt.subplots()
    signal_plot = ax.plot(sample.cpu().numpy(), signal.cpu().numpy(), label='signal', linewidth=1, color='b')
    model_pred_plot = ax.plot(sample.cpu().numpy(), model_pred_history[0], label='model', linewidth=1, color='orange')[0]
    ax.legend(loc='upper right')

    # Plot the rest of the predictions
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
    ani.save(f"{empirical_save_path}", writer=writervideo)
    print(f"animation saved at {empirical_save_path}")
    plt.close()
