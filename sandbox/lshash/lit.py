from typing import List

import lightning.pytorch as pl
import pydantic
import torch.nn.functional as F
import torch.optim
from skimage.metrics import peak_signal_noise_ratio

from ngp_experiments.models import HashEmbedder, NGP
from sandbox.lshash.model import LSHashNGP


class LitNGPConfig(pydantic.BaseModel):
    #
    image_shape: List[int] = None
    architecture: str = "lsh"

    lr: float = 5e-3


class LitNGP(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.full_config = config
        self.config: LitNGPConfig = LitNGPConfig.parse_obj(config)
        cfg = self.config

        if cfg.architecture == "ngp":
            hasher = HashEmbedder(
                cfg.image_shape,
                n_levels=4,
                n_features_per_level=2,
                log2_hashmap_size=12,
                base_resolution=16,
                finest_resolution=256,
            )
            self.ngp = NGP(
                hash_table=hasher,
                hidden_features=64,
                hidden_layers=2,
                out_features=3,
            )
        elif cfg.architecture == "lsh":
            self.ngp = LSHashNGP()
        else:
            raise ValueError()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    def training_step(self, batch, batch_idx):
        coords, colors, hashes = batch
        input = coords if (self.config.architecture == "ngp") else hashes
        preds = self.ngp(input)

        loss = F.mse_loss(preds, colors)
        psnr = peak_signal_noise_ratio(
            colors.detach().numpy(),
            preds.detach().numpy(),
            data_range=1,
        )

        self.log("loss", loss, batch_size=coords.shape[0])
        self.log("psnr", psnr, batch_size=coords.shape[0])

        return loss
