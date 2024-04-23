# A New Perspective To Understanding Multi-resolution Hash Grid Neural Fields

> __ASMR - Activation-Sharing Multi-Resolution Coordinate Networks for Efficient Inference__  
> [STS Luo](https://www.cs.toronto.edu/~stevenlts)  
> _UofT CSC494 Winter 2024 Project_  
> __[Report](https://drive.google.com/file/d/1DtmzJPLqtytvwh5QF2XE_vQfQpVxN-Qe/view?usp=sharing)__

## Structure of the repo
This repo consists of all the code used to generate the figures in the report. Each figure is treated as an individual experiment and the corresponding code file contains both the training and plotting code.

Currently, there are four sets of experiments:
- Domain manipulation visualizer (`hash_visualizer.py`)
- Domain scaling and performance (`exp_scale_up.py`)
- Domain flipping, prediction segments, and signal bandwidth (`exp_bandwidth.py`)
- Collision rates and performance (`exp_collision_error.py`) (not included in the report)
- Visualizing how the MLP resolve hash collision via gradients (`exp_collision_gradient.py`) (not included in the report)

Each experiment code is structure as follow:
- `train(path, trial, n_seeds, **)` 
- `plot(model_path, figure_path, **)`
- `if __name__ == "__main__"` block which defines and creates the directories, and iterates through the number of trials to call `train()` and `plot()`


## To reproduce the figures
Simply run with a GPU the following code
```python exp_<experiment_name>.py```

Or download from [dropbox](https://www.dropbox.com/scl/fo/0p4zsga84of5f31aessny/AM4EPBJa94YIOSg892TXgt8?rlkey=mwrka4jnime4pth06p3vtied0&st=gijswjmr&dl=0)

## Legacy code
Attempted to develop a similarity score for signals that could determine how suitable a signal is for NGP to fit analytically. Related codes:
- `main.py`
- `sim_score_plot.py`
- `scorers/`
