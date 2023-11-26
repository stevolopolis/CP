from typing import Optional

import lightning.pytorch as pl
import pydantic_cli
from lightning.pytorch.loggers import WandbLogger

from sandbox.lshash.dataset import SingleImageDataModule
from sandbox.lshash.lit import LitNGPConfig, LitNGP


class TrainDelaunayNGPConfig(LitNGPConfig):
    #
    seed: int = 100
    accelerator: str = "cpu"
    devices: int = 1
    strategy: Optional[str] = "auto"

    # =================
    # Datamodule Fields
    # =================

    image_path: str = "kodim20.png"
    batch_size: Optional[int] = 1e10
    num_workers: int = 8

    # ===============
    # Training Fields
    # ===============

    max_epochs: int = 1000

    # ==============
    # Logging Fields
    # ==============

    wandb: bool = False
    wandb_project: str = "delaunay_ngp"
    progress_bar: bool = False

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def train(config: TrainDelaunayNGPConfig):
    cfg = config

    # Seeding
    pl.seed_everything(cfg.seed, workers=True)

    # Load dataset
    image = SingleImageDataModule(path=cfg.image_path)
    cfg.image_shape = image.shape

    # Initialize and load model
    # We use dict(cfg) so checkpointing does not rely on TrainDelaunayNGPConfig
    model = LitNGP(config=dict(cfg))

    # Initialize trainer
    if cfg.wandb:
        logger = WandbLogger(project=cfg.wandb_project, config=dict(cfg))
    else:
        logger = False

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        enable_checkpointing=False,
        logger=logger,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=1,
        enable_progress_bar=cfg.progress_bar,
        use_distributed_sampler=True,
    )

    # Start training
    trainer.fit(model=model, datamodule=image)

    return 0


if __name__ == "__main__":
    pydantic_cli.run_and_exit(TrainDelaunayNGPConfig, train)
