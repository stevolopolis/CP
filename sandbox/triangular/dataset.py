import pathlib

import einops
import lightning.pytorch as pl
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, default_collate


class SingleImageDataset(Dataset):

    def __init__(self, path):
        ROOT = pathlib.Path(__file__).parent / "images"
        image = Image.open(ROOT / path)

        image = T.ToTensor()(image)
        image = einops.rearrange(image, "c y x -> x y c")
        self.W, self.H, _ = image.shape

        x, y = torch.meshgrid(
            torch.linspace(-1, 1, self.W),
            torch.linspace(-1, 1, self.H),
            indexing="ij",
        )
        coords = torch.stack([x, y], dim=-1)
        assert coords.shape[:2] == image.shape[:2]

        self.coords = einops.rearrange(coords, "w h d -> (w h) d")
        self.image = einops.rearrange(image, "w h d -> (w h) d")

    def __len__(self):
        return 1

    def __getitem__(self, item):
        assert item == 0
        return self.coords, self.image


class SingleImageDataModule(pl.LightningDataModule):

    def __init__(self, path):
        super().__init__()

        self.dataset = SingleImageDataset(path)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        return [x.squeeze(0) for x in default_collate(batch)]
