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


def plot_corr_vs_n_pieces(n_pieces_ls, corr_ls, save_path=None):
    """Line plot of correlation vs n_pieces"""
    plt.plot(n_pieces_ls, corr_ls, label=['Correlation', 'NULL Proc'])
    plt.xlabel('n_pieces')
    plt.ylabel('Correlation')
    plt.title('Correlation vs n_pieces')
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        print("Saved to %s" % save_path)
    else:
        plt.show()

    plt.close()


def load_val(path):
    with open(path, 'r') as f:
        val = float(f.read().strip())
        # check if val is infs or nans
        if val == float('inf') or val == float('-inf') or val != val:
            print(f"Value at {path} is {val}")
            return -1
        return val


if __name__ == '__main__':
    n_trials = 70
    n_pieces = 30
    dynamic = True
    MODEL = 'relu'

    corr_save_path = f"vis/toy_signal/correlation"
    create_subdirectories(corr_save_path)

    # ===============================================
    # Correlation between analytical score and loss
    # ===============================================
    # rdp = []
    # rdpAnal = []
    # loss = []
    # for trial in range(n_trials):
    #     model_path = f"results/{MODEL}/{MODEL}_{trial}" 
    #     analytical_save_path = "vis/toy_signal/analytical/%s" % trial
    #     empirical_save_path = "vis/toy_signal/empirical/%s" % trial

    #     rdp_score = load_val(f"{analytical_save_path}/rdp{n_pieces}_score.txt")
    #     rdpAnal_score = load_val(f"{analytical_save_path}/rdpAnal{n_pieces}_score.txt")
    #     model_loss = load_val(f"{empirical_save_path}/loss.txt")

    #     if rdp_score == -1 or rdpAnal_score == -1 or model_loss == -1:
    #         continue

    #     rdp.append(rdp_score)
    #     rdpAnal.append(rdpAnal_score)
    #     loss.append(model_loss)
        
    # plot_analytical_score_loss_corr_combined(rdp, loss, save_path=f"{corr_save_path}/rdp{n_pieces}_loss.png")
    # plot_analytical_score_loss_corr_combined(rdpAnal, loss, save_path=f"{corr_save_path}/rdpAnal{n_pieces}_loss.png")


    # # ===============================================
    # # Relationship between score-loss correlation and n_pieces
    # # ===============================================
    # rdp_corr_ls = []
    # rdpAnal_corr_ls = []
    # n_pieces_ls = [n for n in range(10, 101, 10)]
    # for n_pieces in n_pieces_ls:
    #     rdp = []
    #     rdpAnal = []
    #     ema = []
    #     loss = []
    #     for trial in range(n_trials):
    #         model_path = f"results/{MODEL}/{MODEL}_{trial}" 
    #         analytical_save_path = "vis/toy_signal/analytical/%s" % trial
    #         empirical_save_path = "vis/toy_signal/empirical/%s" % trial

    #         rdp_score = load_val(f"{analytical_save_path}/rdp{n_pieces}_score.txt")
    #         rdpAnal_score = load_val(f"{analytical_save_path}/rdpAnal{n_pieces}_score.txt")
    #         model_loss = load_val(f"{empirical_save_path}/loss.txt")

    #         if rdp_score == -1 or rdpAnal_score == -1 or model_loss == -1:
    #             continue

    #         rdp.append(rdp_score)
    #         rdpAnal.append(rdpAnal_score)
    #         loss.append(model_loss)
        
    #     rdp_corr_ls.append(pearsonr(rdp, loss))
    #     rdpAnal_corr_ls.append(pearsonr(rdpAnal, loss))

    # plot_corr_vs_n_pieces(n_pieces_ls, rdp_corr_ls, save_path=f"{corr_save_path}/rdp_corr_vs_n_pieces.png")
    # plot_corr_vs_n_pieces(n_pieces_ls, rdpAnal_corr_ls, save_path=f"{corr_save_path}/rdpAnal_corr_vs_n_pieces.png")
    


    # ===============================================
    # Matched correlation between analytical score and loss
    # ===============================================
    rdp = []
    loss = []
    n_pieces_ls = [n for n in range(10, 101, 10)]
    for trial in range(n_trials):
        model_path = f"results/{MODEL}/{MODEL}_{trial}" 
        analytical_save_path = "vis/toy_signal/analytical/%s" % trial
        empirical_save_path = "vis/toy_signal/empirical/%s" % trial

        model_loss = load_val(f"{empirical_save_path}/loss.txt")
        loss.append(model_loss)

        rdp_score_ls = []
        for n_pieces in n_pieces_ls:
            rdp_score = load_val(f"{analytical_save_path}/rdp{n_pieces}_score.txt")

            if rdp_score == -1 or model_loss == -1:
                continue

            rdp_score_ls.append(rdp_score)
        
        matching_idx = np.argmin(np.abs(np.array(rdp_score_ls) - model_loss))
        rdp.append(rdp_score_ls[matching_idx])

    plot_analytical_score_loss_corr_combined(rdp, loss, save_path=f"{corr_save_path}/rdp_matched_loss.png")
        
