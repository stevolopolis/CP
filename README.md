# Coordinate Permutation Is All You Need

## Structure of the repo
- `main_image.py`: main() to train on 2D images
- `main_sdf.py`: main() to train on 3D SDFs (under construction)
- `hash_visualizer.py`: 
- `data.py`: dataLoaders
- `<model_type>_experiments`: 
    - `models.py`: nn.modules
    - `utils.py`: model specific trainer class 
- `configs`:
    - `<model_type>.ini`: model specific configurations

We use W&B to log training stats and visualizations. W&B configs are specified in the config .ini files:
```
[WANDB]
use_wandb: 1
wandb_project: playground
wandb_entity: utmist-parsimony
```
Do not change `wandb_entity`, this is our base W&B directory. If you are running a big cluster of experiments that is working towards your own dedicated research question, create a new W&B project by specifying a different string for `wandb_project`. For any small PoC experiments, feel free to just dump it into `wandb_project: playground`.

## Get started -- Data preparation
All data are loaded using the function `get_data()` defined in `main_image.py`. All data should be stored in a common directory. Change the global variable `BASE_PATH` in `main_image.py` to the common directory once setup.

## Get started -- train a new model
```python main_image.py configs/<config.ini> <model_type>``` 

## Get started -- visualize a trained model
```python hash_visualizer.py configs/<config.ini> <model_type> <vis_folder>```

## Questions seeking answers
- Coordinate permutation (CP) perspective in one resolution (observed; not proved yet)
- How do we prove that NGP is doing CP instead of embedding learning? Or, is embedding learning = CP? We need a rigorous definition and explanation
- CP when concatenated in multiple resolutions
- CP on other hybrid methods
- CP with or without hashing
- CP in 3D or higher dimensions
- CP param to expressivity trade-off (rigorous math)
- Is hash embedding learning permuted coordinates or saving multi-resolution pixel values
- Deformable NGP grid
- CP as a constructive method of INRs
    - Can expand a whole topic here…
