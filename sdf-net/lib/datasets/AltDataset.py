# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os

import torch
from torch.utils.data import Dataset

import numpy as np

from lib.utils import setparam

class AltDataset(Dataset):
    """Base class for single mesh datasets."""

    def __init__(self, 
        args=None, 
        dataset_path = None,
        raw_obj_path = None,
        num_samples = None,
        validation=False
    ):
        self.args = args
        self.dataset_path = setparam(args, dataset_path, 'dataset_path')
        self.raw_obj_path = setparam(args, raw_obj_path, 'raw_obj_path')
        self.num_samples = setparam(args, num_samples, 'num_samples')

        # Possibly remove... or fix trim obj
        #if self.raw_obj_path is not None and not os.path.exists(self.dataset_path):
        #    _, _, self.mesh = trim_obj_to_file(self.raw_obj_path, self.dataset_path)
        #elif not os.path.exists(self.dataset_path):
        #    assert False and "Data does not exist and raw obj file not specified"
        #else:
        
        # Load the .npz files
        surface = np.load(os.path.join(self.dataset_path, "surface", self.raw_obj_path))
        self.surface_pos = surface['position']
        self.surface_dis = surface['distance']
        near = np.load(os.path.join(self.dataset_path, "near", self.raw_obj_path))
        self.near_pos = near['position']
        self.near_dis = near['distance']
        rand = np.load(os.path.join(self.dataset_path, "rand", self.raw_obj_path))
        self.rand_pos = rand['position']
        self.rand_dis = rand['distance']

        # Calculate the ratio of points
        print("Surface: %s\t Near: %s\t Rand: %s" % (len(self.surface_dis), len(self.near_dis), len(self.rand_dis)))
        #print("Samles per type: %s" % (n_samples_per_type))
        #assert len(self.surface_dis) % n_samples_per_type == 0

        """
        # Divide the points into batches
        self.surface_pos = np.reshape(self.surface_pos, (-1, n_samples_per_type, 3))
        self.surface_dis = np.reshape(self.surface_dis, (-1, n_samples_per_type, 1))
        self.near_pos = np.reshape(self.near_pos, (-1, n_samples_per_type, 3))
        self.near_dis = np.reshape(self.near_dis, (-1, n_samples_per_type, 1))
        self.rand_pos = np.reshape(self.rand_pos, (-1, n_samples_per_type, 3))
        self.rand_dis = np.reshape(self.rand_dis, (-1, n_samples_per_type, 1))"""

        if validation:
            self.pts = self.rand_pos
            self.d = self.rand_dis
        else:
            self.pts = np.concatenate((self.surface_pos, self.near_pos, self.rand_pos), axis=0)
            self.d = np.concatenate((self.surface_dis, self.near_dis, self.rand_dis), axis=0)

        # Batchwise scrambling of points among the three types
        rand_idx = np.random.permutation(len(self.pts))
        self.pts = self.pts[rand_idx]
        self.d = self.d[rand_idx]

        # Remove nan values
        nan_idx = np.isnan(self.pts).any(axis=1)
        self.pts = self.pts[~nan_idx][:self.num_samples]
        self.d = self.d[~nan_idx][:self.num_samples]

        # Convert to torch.tensor
        self.pts = torch.tensor(self.pts)
        self.d = torch.tensor(self.d)

        print("Datashape(x): ", self.pts.shape)
        print("Datashape(y): ", self.d.shape)

    def resample(self):
        pass

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        return self.pts[idx], self.d[idx], torch.zeros_like(self.d[idx])
            
    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.pts.size()[0]

    def num_shapes(self):
        """Return length of dataset (number of _mesh models_)."""

        return 1
