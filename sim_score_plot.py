import matplotlib.pyplot as plt
import numpy as np


def plot_sim_score_loss_corr(sim_score, loss_score, save_path=None):
    """Scatter plot of similarity score vs loss score"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Correlation plot for slopes
    ax[0].scatter(sim_score[:, 0], loss_score)
    ax[0].set_xlabel('Similarity score (slope)')
    ax[0].set_ylabel('Loss score')
    ax[0].set_title('Similarity score (slope) vs Loss score')
    # Correlation plot for y-intercepts
    ax[1].scatter(sim_score[:, 1], loss_score)
    ax[1].set_xlabel('Similarity score (y-int)')
    ax[1].set_ylabel('Loss score')
    ax[1].set_title('Similarity score (y-int) vs Loss score')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def load_sim_score_loss_corr(sim_score_path, loss_score_path):
    """Load similarity score and loss score from file"""
    sim_score = np.load(sim_score_path)
    loss_score = np.load(loss_score_path)
    return sim_score, loss_score


if __name__ == '__main__':
    n_pieces = 1
    sim_loss_corr_save_path = "vis/sim_loss_corr_%s.png" % n_pieces

    sim_score, loss_score = load_sim_score_loss_corr("logs/sim_scores_relu_knots_%s.npy" % n_pieces, "logs/losses_relu_dummy.npy")
    loss_score[-2] = loss_score[-3]
    sim_score[-2, :] = sim_score[-3, :]
    plot_sim_score_loss_corr(sim_score, loss_score, save_path=sim_loss_corr_save_path)