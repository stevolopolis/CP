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

import torch
import random
from torch.utils.data import DataLoader    

def compute_iou(dist_gt, dist_pr):
    """Intersection over Union.
    
    Args:
        dist_gt (torch.Tensor): Groundtruth signed distances
        dist_pr (torch.Tensor): Predicted signed distances
    """

    occ_gt = (dist_gt < 0).byte()
    occ_pr = (dist_pr < 0).byte()

    area_union = torch.sum((occ_gt | occ_pr).float())
    area_intersect = torch.sum((occ_gt & occ_pr).float())

    iou = area_intersect / area_union
    return 100. * iou


def tv_loss_fn(net, device):
    """Total variance loss of sdf"""
    # pts.shape = [1024, 1024, 1024, 3]
    random_idx = random.random() - 0.5
    pts = torch.stack(torch.meshgrid([torch.linspace(random_idx, random_idx+0.125, 64) for _ in range(3)]), dim=-1).to(device)
    # Reshape pts to (1024**3, 3)
    pts = pts.view(-1, 3)
    #data_loader = DataLoader(pts, batch_size=256**2, shuffle=False)
    # Inference
    preds = net.sdf(pts)
    # Reshape preds to [1024, 1024, 1024, 1]
    preds = preds.view(64, 64, 64, 1)
    # Calculate tv loss
    tv_x = torch.pow(preds[1:, :, :, :] - preds[:-1, :, :, :], 2).sum()
    tv_y = torch.pow(preds[:, 1:, :, :] - preds[:, :-1, :, :], 2).sum()
    tv_z = torch.pow(preds[:, :, 1:, :] - preds[:, :, :-1, :], 2).sum()
    loss = (tv_x + tv_y + tv_z) / (64**3)

    return loss

