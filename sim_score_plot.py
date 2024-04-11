import matplotlib.pyplot as plt
from scipy.stats import pearsonr   

from misc import *

def plot_analytical_score_loss_corr_combined(analytical_score, loss_score, save_path=None):
    """Scatter plot of similarity score vs loss score"""
    corr = pearsonr(analytical_score, loss_score)

    plt.scatter(analytical_score, loss_score)
    plt.xlabel('Similarity score')
    plt.ylabel('Loss score')
    plt.title('Similarity score vs Loss score\n(corr: %0.2f, prob null: %0.2f)' % (corr[0], corr[1]))

    plt.ylim(0, max(loss_score) * 1.25)

    for i in range(len(analytical_score)):
        plt.annotate(i, (analytical_score[i], loss_score[i]))

    if save_path is not None:
        plt.savefig(save_path)
        print("Saved to %s" % save_path)
    else:
        plt.show()

    plt.close()


def load_val(path):
    with open(path, 'r') as f:
        return float(f.read().strip())


if __name__ == '__main__':
    n_trials = 100
    n_pieces = 50
    dynamic = True
    MODEL = 'relu'

    corr_save_path = f"vis/toy_signal/correlation"
    create_subdirectories(corr_save_path)

    rdp = []
    ema = []
    loss = []
    for trial in range(n_trials):
        model_path = f"results/{MODEL}/{MODEL}_{trial}" 
        analytical_save_path = "vis/toy_signal/analytical/%s" % trial
        empirical_save_path = "vis/toy_signal/empirical/%s" % trial

        rdp_score = load_val(f"{analytical_save_path}/rdp{n_pieces}_score.txt")
        ema_score = load_val(f"{analytical_save_path}/ema{n_pieces}_score.txt")
        model_loss = load_val(f"{empirical_save_path}/loss.txt")

        rdp.append(rdp_score)
        ema.append(ema_score)
        loss.append(model_loss)
        
    plot_analytical_score_loss_corr_combined(ema, loss, save_path=f"{corr_save_path}/ema{n_pieces}_loss.png")
    plot_analytical_score_loss_corr_combined(rdp, loss, save_path=f"{corr_save_path}/rdp{n_pieces}_loss.png")