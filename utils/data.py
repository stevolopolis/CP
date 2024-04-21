import errno
import random
import torch
import sys
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import numpy as np
from PIL import Image
import skimage
import os
import math

import urllib3

###########################
# Toy signal generators
###########################
def generate_fourier_signal(sample, n, coeffs=None, phases=None, freqs=None, device="cuda"):
    if coeffs is None:
        coeffs = torch.randn(n)
    if phases is None:
        phases = torch.rand(n) * 2 * torch.pi
    if freqs is None:
        freqs = torch.linspace(1, n, n) * 2 * torch.pi
    signal = torch.zeros_like(sample).to(device)
    print("generating signal...")
    for i in range(n):
        signal += coeffs[i] * torch.sin(freqs[i] * sample + phases[i]) / n

    return signal, coeffs, freqs, phases


def generate_piecewise_signal(sample, n, seed=42, device="cuda"):
    """Generates a piecewise signal with n pieces."""
    torch.manual_seed(seed)
    signal = torch.zeros_like(sample).to(device)
    print("generating signal...")
    if isinstance(n, list):
        n_segs = len(n)
        samples_per_seg = len(sample) // n_segs
        knots = (torch.rand(n[0]+1) * len(sample) // n_segs).int()
        for i, sub_n in enumerate(n[1:], 1):
            knots = torch.cat((knots, (torch.rand(sub_n+1) * samples_per_seg + (samples_per_seg * i)).int()), dim=0)
        n_pieces = sum(n)
    else:
        knots = (torch.rand(n+1) * len(sample)).int()
        n_pieces = n
    knots[0] = 0
    knots[-1] = len(sample)-1
    knots = torch.sort(knots)[0].to(device)
    #slopes = torch.randn(n_pieces).to(device)
    slopes = (torch.rand(n_pieces).to(device) - 0.5) * 4
    init_y = torch.randn(1).to(device)
    b = []
    for i in range(n_pieces+1):
        if i == 0:
            signal[0] = init_y
        elif i == n_pieces:
            signal[knots[i-1]:] = slopes[i-1] * (sample[knots[i-1]:] - sample[knots[i-1]-1]) + signal[knots[i-1]-1]
            b.append(signal[knots[i-1]-1] - slopes[i-1] * sample[knots[i-1]-1])
        elif i == 1:
            signal[knots[i-1]:knots[i]] = slopes[i-1] * (sample[:knots[i]] - sample[0]) + init_y.to(device)
            b.append(init_y)
        else:
            signal[knots[i-1]:knots[i]] = slopes[i-1] * (sample[knots[i-1]:knots[i]] - sample[knots[i-1]-1]) + signal[knots[i-1]-1]
            b.append(signal[knots[i-1]-1] - slopes[i-1] * sample[knots[i-1]-1])

    return signal, knots, slopes, torch.tensor(b)


def generate_piecewise_segmented_signal(samples, ns, seed=42):
    """Generates a piecewise signal len(samples) consecutive regions, each with n[i] pieces."""
    new_samples = None
    new_signals = None
    new_knots = None
    new_slopes = None
    new_bs = None
    for i, (sample, n) in enumerate(zip(samples, ns)):
        signal, knot, slope, b = generate_piecewise_signal(sample, n, seed=seed)
        if i == 0:
            new_samples = sample
            new_signals = signal
            new_knots = knot
            new_slopes = slope
            new_bs = b
        else:
            new_samples = torch.cat((new_samples, sample))
            new_signals = torch.cat((new_signals, signal))
            new_knots = torch.cat((new_knots, knot))
            new_slopes = torch.cat((new_slopes, slope))
            new_bs = torch.cat((new_bs, b))

    return new_samples, new_signals, new_knots, new_slopes, new_bs


###########################
# Toy signal generators END
###########################

class ImageFile(Dataset):
    def __init__(self, filename, coord_mode=2, url=None, grayscale=False):
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

        self.img = self.transform(self.img).permute(1,2,0).view(-1, self.img_channels)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, self.img
    
    def gauss_data(self):
        self.img = torch.randn_like(self.img)

    def scramble_data(self):
        self.random_idx = torch.randperm(self.H*self.W)
        self.img = self.img[self.random_idx]

    def local_decolorize(self, patch_size=.1):
        dim, x, y = self.get_local_patch_coords(patch_size)

        self.img = self.img.view(self.H, self.W, self.img_channels)
        self.img[x:x+dim, y:y+dim, :] = 0
        self.img = self.img.view(-1, self.img_channels)
    
    def global_decolorize(self):
        self.img = torch.zeros_like(self.img)

    def local_color_shift(self, patch_size=.1, shift=.25):
        dim, x, y = self.get_local_patch_coords(patch_size)

        self.img = self.img.view(self.H, self.W, self.img_channels)
        self.img[x:x+dim, y:y+dim, :] = self.img[x:x+dim, y:y+dim, :] * shift
        self.img = self.img.view(-1, self.img_channels)

    def global_color_shift(self, shift=.25):
        self.img = self.img * shift

    def disconnected_double_patch(self, patch_size=.1, exp_seed=0):
        coords = self.get_n_local_patch_coords(patch_size, 2, exp_seed=exp_seed)
        dim1_x, dim1_y, x1, y1 = coords[0]
        dim2_x, dim2_y, x2, y2 = coords[1]

        self.img = self.img.view(self.H, self.W, self.img_channels)
        self.img[y1:y1+dim1_y, x1:x1+dim1_x, :] = torch.tensor([0.0, 1.0, 0.0])
        self.img[y2:y2+dim2_y, x2:x2+dim2_x, :] = torch.tensor([0.0, 1.0, 0.0])
        self.img = self.img.view(-1, self.img_channels)

    def get_local_patch_coords(self, patch_size, x_pos='random'):
        dim_x = int(self.W * patch_size)
        dim_y = int(self.H * patch_size)
        top_left_corner_y = random.randint(0, self.H - dim_y)
        if x_pos == "random":
            top_left_corner_x = random.randint(0, self.W - dim_x)
        elif x_pos == "left":
            top_left_corner_x = random.randint(0, (self.W - dim_x) // 2)
        elif x_pos == "right":
            top_left_corner_x = random.randint((self.W - dim_x) // 2, self.W - dim_x)
        return dim_x, dim_y, top_left_corner_x, top_left_corner_y

    def get_n_local_patch_coords(self, patch_size, n, exp_seed=0):
        coords = []
        seed = random.randrange(sys.maxsize)
        for i in range(n):
            random.seed(seed+i+exp_seed)
            x_pos = "left" if i % 2 == 0 else "right"
            coords.append(self.get_local_patch_coords(patch_size, x_pos))
        return coords

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


class WaveFile(Dataset):
    def __init__(self, size, coord_mode='1', wave_type=0, p1=None):
        super().__init__()
        self.size = size

        if coord_mode == 0:
            self.coords = torch.linspace(0.0, size, size)
        elif coord_mode == 1:
            self.coords = torch.linspace(-1.0, 1 - 1e-6, size)
        elif coord_mode == 2:
            self.coords = torch.linspace(0.0, 1 - 1e-6, size)

        self.coords = self.coords.view(-1, 1)
        self.data = self.wave_factory(self.coords, wave_type, p1)

    def __len__(self):
        return 1

    def wave_factory(self, x, wave_type, n=None):
        if wave_type == 0:
            return self.generate_gauss_wave(x)
        elif wave_type == 1:
            return self.generate_square_wave(x, n)
        elif wave_type == 2:
            return self.generate_fourier_wave(x, n)
        else:
            raise NotImplementedError

    def generate_gauss_wave(self, x):
        data = torch.randn_like(x)
        return data

    def generate_square_wave(self, x, n):
        if n is None: n = 2
        domain_length = len(x)
        max_interval = domain_length // n
        data = torch.zeros_like(x)
        for i in range(n):
            random.seed(i)
            interval = random.randint(0, max_interval)
            start = random.randint(0, domain_length - interval)
            val = random.randint(0, 255)
            data[start:start + interval] = val

        return data

    def generate_fourier_wave(self, x, n):
        if n is None: n = 10
        data = torch.zeros_like(x)
        for i in range(1, n + 1):
            freq = 2 * math.pi * i
            data += random.gauss(0.0, 1.0) * torch.sin(freq * x) / n

        return data

    def disconnected_double_patch(self, patch_size=.1, exp_seed=0):
        coords = self.get_n_local_patch_coords(patch_size, 2, exp_seed=exp_seed)
        val = random.random()
        dim1_x, x1 = coords[0]
        dim2_x, x2 = coords[1]

        self.data[x1:x1+dim1_x, :] = val
        self.data[x2:x2+dim2_x, :] = val

    def get_local_patch_coords(self, patch_size, x_pos='random'):
        dim_x = int(self.size * patch_size)
        if x_pos == "random":
            top_left_corner_x = random.randint(0, self.size - dim_x)
        elif x_pos == "left":
            top_left_corner_x = random.randint(0, (self.size - dim_x) // 2)
        elif x_pos == "right":
            top_left_corner_x = random.randint((self.size - dim_x) // 2, self.size - dim_x)
        return dim_x, top_left_corner_x

    def get_n_local_patch_coords(self, patch_size, n, exp_seed=0):
        coords = []
        seed = random.randrange(sys.maxsize)
        for i in range(n):
            random.seed(seed+i+exp_seed)
            x_pos = "left" if i % 2 == 0 else "right"
            coords.append(self.get_local_patch_coords(patch_size, x_pos))
        return coords


    def __getitem__(self, idx):
        return self.coords, self.data
    
    def get_data_shape(self):
        return (self.size, 1)