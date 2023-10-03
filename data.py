import errno
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import numpy as np
from PIL import Image
import skimage
import os

import urllib3


class ImageFile(Dataset):
    def __init__(self, filename, coord_mode='1', url=None, grayscale=False):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = 1000000000
        file_exists = os.path.isfile(filename)

        if not file_exists:
            if url is None:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), filename)
            else:
                print('Downloading image file...')
                urllib3.request.urlretrieve(url, filename)

        self.img = Image.open(filename)
        if grayscale:
            self.img = self.img.convert('L')

        self.img_channels = len(self.img.mode)
        self.W, self.H = self.img.size
        
        self.transform = Compose(
            [
                ToTensor(),
                # Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5])), # [0, 1] -> [-1, 1]
            ]
        )
        
        if coord_mode == 0:
            self.coords = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(0.0, self.H-1, self.H),
                        torch.linspace(0.0, self.W-1, self.W),
                    ]
                ),
                dim=-1,
            ).view(-1, 2)
        elif coord_mode == 1:
            self.coords = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(-1.0, 1 - 1e-6, self.H),
                        torch.linspace(-1.0, 1 - 1e-6, self.W),
                    ]
                ),
                dim=-1,
            ).view(-1, 2)
        elif coord_mode == 2:
            self.coords = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(0.0, 1 - 1e-6, self.H),
                        torch.linspace(0.0, 1 - 1e-6, self.W),
                    ]
                ),
                dim=-1,
            ).view(-1, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.transform(self.img).permute(1,2,0).view(-1, self.img_channels)
    
    def get_img_h(self):
        return self.H
    
    def get_img_w(self):
        return self.W
    
    def get_img_c(self):
        return self.img_channels

    def get_data_shape(self):
        return (self.H, self.W, self.img_channels)


class BigImageFile(Dataset):
    def __init__(self, filename, max_coords=500000, coord_mode='0', url=None, grayscale=False):
        super().__init__()
        Image.MAX_IMAGE_PIXELS = 1000000000
        file_exists = os.path.isfile(filename)

        if not file_exists:
            if url is None:
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), filename)
            else:
                print('Downloading image file...')
                urllib3.request.urlretrieve(url, filename)

        self.gpu_max_coords = max_coords
        self.img = Image.open(filename)
        if grayscale:
            self.img = self.img.convert('L')

        self.img_channels = len(self.img.mode)

        self.W, self.H = self.img.size
        
        self.img = torch.tensor(np.array(self.img)).view(-1, self.img_channels)
        self.img = self.img / 255      #(self.img / 255 - 0.5) * 2

        if coord_mode == 0:
            self.coords = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(0.0, self.H-1, self.H),
                        torch.linspace(0.0, self.W-1, self.W),
                    ]
                ),
                dim=-1,
            ).view(-1, 2)
        elif coord_mode == 1:
            self.coords = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(-1.0, 1 - 1e-6, self.H),
                        torch.linspace(-1.0, 1 - 1e-6, self.W),
                    ]
                ),
                dim=-1,
            ).view(-1, 2)
        elif coord_mode == 2:
            self.coords = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(0.0, 1 - 1e-6, self.H),
                        torch.linspace(0.0, 1 - 1e-6, self.W),
                    ]
                ),
                dim=-1,
            ).view(-1, 2)

        self.random_idx = torch.randperm(self.H*self.W)
        self.coords = self.coords[self.random_idx]
        self.img = self.img[self.random_idx]

    def __len__(self):
        return int(self.H * self.W // self.gpu_max_coords) + 1

    def __getitem__(self, idx):
        return self.coords[idx*self.gpu_max_coords:min((idx+1)*self.gpu_max_coords, self.H*self.W-1)], \
                self.img[idx*self.gpu_max_coords:min((idx+1)*self.gpu_max_coords, self.H*self.W-1)]
    
    def get_img_h(self):
        return self.H
    
    def get_img_w(self):
        return self.W
    
    def get_img_c(self):
        return self.img_channels

    def get_data_shape(self):
        return (self.H, self.W, self.img_channels)


class CIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR10 dataset without labels."""

    def __init__(self, path, idx):
        super().__init__(root=path,
                         train=False,
                         transform=torchvision.transforms.ToTensor())
        self.idx = idx
        self.coords = torch.stack(
            torch.meshgrid(
                [
                    torch.linspace(0.0, 1 - 1e-6, 32),
                    torch.linspace(0.0, 1 - 1e-6, 32),
                ]
            ),
            dim=-1,
        ).view(-1, 2)

    def __getitem__(self, idx):
        if self.transform:
            return self.coords, self.transform(self.data[self.idx]).permute(1, 2, 0).view(-1, 3)
        else:
            return self.coords, self.data[self.idx].view(-1, 3)
        
    def get_data_shape(self):
        return self.data[self.idx].shape


class CameraDataset(Dataset):
    def __init__(self, side_length=None, normalize=True):

        self.image = Image.fromarray(skimage.data.camera())

        self.img_channels = len(self.image.mode)

        if side_length is None:
            side_length = self.image.size[-1]
        
        self.side_length = side_length

        self.transform = Compose(
            [
                Resize(side_length),
                ToTensor(),
                #Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])), # [0, 1] -> [-1, 1]
            ]
        )

        if not normalize:
            self.coords = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(0.0, side_length, self.H),
                        torch.linspace(0.0, side_length, self.W),
                    ]
                ),
                dim=-1,
            ).view(-1, 2)
        else:
            self.coords = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(-1.0, 1 - 1e-6, side_length),
                        torch.linspace(-1.0, 1 - 1e-6, side_length),
                    ]
                ),
                dim=-1,
            ).view(-1, 2)

        return

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.transform(self.image).permute(1,2,0).view(-1, self.img_channels)
    
    def get_img_h(self):
        return self.side_length
    
    def get_img_w(self):
        return self.side_length
    
    def get_img_c(self):
        return self.img_channels

    def get_data_shape(self):
        return (self.side_length, self.side_length, self.img_channels)


class VideoDataset(Dataset):
    """
    The video argument should be a 3d numpy array with time/frame as the first dimension.
    """
    def __init__(self, video):

        self.video = torch.tensor(np.load(video))
        self.video = 2*self.video - 1. # normalize to [-1, 1]
        self.timesteps = self.video.shape[0]
        self.side_length = self.video.shape[1]
        mesh_sizes = self.video.shape[:-1]
        self.video = self.video.view(-1, 3)

        self.coords = torch.stack(
            torch.meshgrid([torch.linspace(-1.0, 1.0, s) for s in mesh_sizes]), dim=-1
        ).view(-1, 3)

        return

    def __len__(self):
        return self.video.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.video[idx]


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        print("Loading point cloud")
        self.point_cloud = np.genfromtxt(pointcloud_path)
        print("Finished loading point cloud")

        coords = self.point_cloud[:, :3]
        self.normals = self.point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        coords -= np.mean(coords, axis=0, keepdims=True)
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return torch.from_numpy(coords).float(), {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}

    def get_data_shape(self):
        return self.coords.shape

    def get_pointcloud(self):
        return self.point_cloud[:, :3]
