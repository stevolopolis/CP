import pathlib

import einops
import lightning.pytorch as pl
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate

NUM_HASHES = 32
HASH_SIZE = 8


class SingleImageDataset(Dataset):

    def __init__(self, path):
        ROOT = pathlib.Path(__file__).parent / "images"
        image = Image.open(ROOT / path)

        image = T.ToTensor()(image)
        _, self.H, self.W = image.shape
        colors = einops.rearrange(image, "d h w -> (h w) d")

        y, x = torch.meshgrid(
            torch.arange(self.H),
            torch.arange(self.W),
            indexing="ij",
        )
        coords = torch.stack([y, x], dim=-1)
        coords = einops.rearrange(coords, "h w d -> (h w) d")

        self.coords = coords
        self.colors = colors

        # Minhash
        bounds = torch.tensor([self.H, self.W])
        pixels = torch.cat([coords / bounds, colors], dim=-1)  # (N 5)
        planes = torch.rand([5, NUM_HASHES * HASH_SIZE]) - 0.5  # (5 HxD)
        bins = (pixels @ planes) > 0  # (N HxD)
        bins = einops.rearrange(bins, "n (h d) -> n h d", h=NUM_HASHES, d=HASH_SIZE)
        hashes = torch.sum((2 ** torch.arange(HASH_SIZE)) * bins, dim=-1)
        self.hashes = hashes

    def __len__(self):
        return 1

    def __getitem__(self, item):
        assert item == 0
        return self.coords, self.colors, self.hashes


class SingleImageDataModule(pl.LightningDataModule):

    def __init__(self, path):
        super().__init__()

        self.dataset = SingleImageDataset(path)
        self.shape = (self.dataset.H, self.dataset.W)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        return [x.squeeze(0) for x in default_collate(batch)]
