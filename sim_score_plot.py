import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr   


def plot_sim_score_loss_corr_combined(sim_score, loss_score, save_path=None):
    """Scatter plot of similarity score vs loss score"""
    
    corr = pearsonr(sim_score, loss_score)

    plt.scatter(sim_score, loss_score)
    plt.xlabel('Similarity score')
    plt.ylabel('Loss score')
    plt.title('Similarity score vs Loss score\n(corr: %0.2f, prob null: %0.2f)' % (corr[0], corr[1]))

    plt.ylim(0, max(loss_score) * 1.25)

    for i in range(len(sim_score)):
        plt.annotate(i, (sim_score[i], loss_score[i]))

    if save_path is not None:
        plt.savefig(save_path)
        print("Saved to %s" % save_path)
    else:
        plt.show()

    plt.close()


def load_sim_score_loss_corr(sim_score_path, loss_score_path):
    """Load similarity score and loss score from file"""
    sim_score = np.load(sim_score_path)
    loss_score = np.load(loss_score_path)
    return sim_score, loss_score


if __name__ == '__main__':
    n_pieces = 7
    dynamic = True
    model = 'relu'

    if dynamic:
        sim_score_path = "logs/sim_scores_%s_knots_%s_dynamic.npy" % (model, n_pieces)
        sim_loss_corr_save_path = "vis/correlation/sim_loss_corr_%s_%s_dynamic.png" % (model, n_pieces)
    else:
        sim_score_path = "logs/sim_scores_%s_knots_%s.npy" % (model, n_pieces)
        sim_loss_corr_save_path = "vis/correlation/sim_loss_corr_%s_%s.png" % (model, n_pieces)
    loss_path = "logs/losses_%s.npy" % model

    sim_score, loss_score = load_sim_score_loss_corr(sim_score_path, loss_path)
    # plot_sim_score_loss_corr(sim_score, loss_score, save_path=sim_loss_corr_save_path)
    # sim_score[7] = sim_score[6]
    # loss_score[7] = loss_score[6]
    plot_sim_score_loss_corr_combined(sim_score, loss_score, save_path=sim_loss_corr_save_path)