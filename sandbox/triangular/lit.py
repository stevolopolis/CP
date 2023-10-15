import lightning.pytorch as pl
import pydantic
import torch.nn.functional as F
import torch.optim

from sandbox.triangular.model import DelaunayNGP


class LitDelaunayNGPConfig(pydantic.BaseModel):
    #
    # ============
    # Model Fields
    # ============

    hash_features: int = 2
    hash_num_points: int = 1000
    hash_num_heads: int = 1

    mlp_features: int = 32
    mlp_hidden_layers: int = 2

    learned_anchors: bool = True

    # ===============
    # Training Fields
    # ===============

    lr: float = 5e-3


class LitDelaunayNGP(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.full_config = config
        self.config: LitDelaunayNGPConfig = LitDelaunayNGPConfig.parse_obj(config)
        cfg = self.config

        self.ngp = DelaunayNGP(
            hash_features=cfg.hash_features,
            hash_num_points=cfg.hash_num_points,
            hash_num_heads=cfg.hash_num_heads,
            mlp_features=cfg.mlp_features,
            mlp_hidden_layers=cfg.mlp_hidden_layers,
            learned_anchors=cfg.learned_anchors,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    def training_step(self, batch, batch_idx):
        coords, colors = batch
        preds = self.ngp(coords)

        loss = F.mse_loss(preds, colors)
        psnr = -10 * torch.log10(loss)

        self.log("loss", loss, batch_size=coords.shape[0])
        self.log("psnr", psnr, batch_size=coords.shape[0])

        return loss
