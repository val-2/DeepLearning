# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Differentiable augmentations for PyTorch, adapted from the official StyleGAN2-ADA implementation."""

import torch
import torch.nn.functional as F
from torch import nn

#----------------------------------------------------------------------------
# Helpers for augmentation pipeline.

def rand_bool(size, prob, device):
    return torch.rand(size, device=device) < prob

def rand_translation(size, ratio, device):
    return (torch.rand(size, device=device) * 2 - 1) * ratio

def rand_scale(size, ratio, device):
    return (torch.rand(size, device=device) * 2 - 1) * ratio + 1

def rand_angle(size, device):
    return (torch.rand(size, device=device) * 2 - 1) * torch.pi

def rand_contrast(size, device):
    return (torch.rand(size, device=device) + 0.5)

def rand_brightness(size, device):
    return torch.rand(size, device=device) * 0.2

def rand_saturation(size, device):
    return (torch.rand(size, device=device) * 2) # [0, 2]

#----------------------------------------------------------------------------
# Augmentation pipeline.

class AugmentPipe(torch.nn.Module):
    def __init__(self,
        p=0.0,
        xflip=1e8, rotate=1e8, scale=1e8, translate=1e8, cutout=1e8, # geometric
        brightness=1e8, contrast=1e8, saturation=1e8, # color
    ):
        super().__init__()
        self.p = p # Overall probability of applying any augmentation.
        
        # Probabilities for individual augmentations.
        self.xflip = float(xflip)
        self.rotate = float(rotate)
        self.scale = float(scale)
        self.translate = float(translate)
        self.cutout = float(cutout)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)
        
        # Fixed parameters.
        self.translate_std = 0.125
        self.scale_std = 0.2
        self.cutout_size = 0.5

    def forward(self, images):
        if self.p == 0:
            return images
        
        B, C, H, W = images.shape
        device = images.device

        # Create a shared probability mask for the entire batch.
        # An augmentation is applied if its probability > a random value.
        aug_probs = torch.rand(B, device=device)
        
        # Apply each augmentation stochastically.
        images = self.apply_xflip(images, aug_probs)
        images = self.apply_rotate_scale_translate(images, aug_probs)
        images = self.apply_brightness(images, aug_probs)
        images = self.apply_contrast(images, aug_probs)
        images = self.apply_saturation(images, aug_probs)
        images = self.apply_cutout(images, aug_probs)

        return images
    
    def get_grid(self, B, H, W, device):
        # Create a standard coordinate grid.
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, W, device=device),
            torch.linspace(-1, 1, H, device=device),
            indexing='xy'
        )
        grid = torch.stack([grid_x, grid_y], dim=0)
        return grid.unsqueeze(0).expand(B, -1, -1, -1) # B, 2, H, W

    def apply_xflip(self, images, aug_probs):
        p = self.xflip
        if p == 0:
            return images
        
        mask = rand_bool(images.shape[0], p * self.p, device=images.device)
        return torch.where(mask.view(-1, 1, 1, 1), images.flip(3), images)

    def apply_rotate_scale_translate(self, images, aug_probs):
        p_rot = self.rotate
        p_scale = self.scale
        p_trans = self.translate
        if p_rot == 0 and p_scale == 0 and p_trans == 0:
            return images

        B, C, H, W = images.shape
        device = images.device
        grid = self.get_grid(B, H, W, device)

        # Rotation
        angle = torch.zeros(B, device=device)
        if p_rot > 0:
            angle = torch.where(rand_bool(B, p_rot * self.p, device), rand_angle(B, device), angle)
        
        # Scaling
        scale = torch.ones(B, 2, device=device)
        if p_scale > 0:
            s = torch.where(rand_bool(B, p_scale * self.p, device).unsqueeze(1), rand_scale(B, self.scale_std, device).unsqueeze(1), torch.ones(B,1,device=device))
            scale = s.expand(-1, 2)
        
        # Translation
        translation = torch.zeros(B, 2, device=device)
        if p_trans > 0:
            t = torch.where(rand_bool(B, p_trans * self.p, device).unsqueeze(1), rand_translation(B, self.translate_std, device).unsqueeze(1), torch.zeros(B,1,device=device))
            translation = t.expand(-1, 2)

        # Build affine matrix
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        
        # Matrix for rotation and scaling
        transform_matrix = torch.eye(2, 2, device=device).unsqueeze(0).repeat(B, 1, 1)
        transform_matrix[:, 0, 0] = cos_a * scale[:, 0]
        transform_matrix[:, 0, 1] = -sin_a * scale[:, 1]
        transform_matrix[:, 1, 0] = sin_a * scale[:, 0]
        transform_matrix[:, 1, 1] = cos_a * scale[:, 1]
        
        # Apply transformation to grid
        # grid is B, 2, H, W. view as B, 2, HW. permute to B, HW, 2
        # matrix is B, 2, 2. matmul with (B, 2, HW)
        new_grid = torch.bmm(transform_matrix, grid.view(B, 2, -1)).view(B, 2, H, W)
        
        # Add translation
        new_grid += translation.view(B, 2, 1, 1)

        # Resample image using the transformed grid
        return F.grid_sample(images, new_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='reflection', align_corners=False)

    def apply_brightness(self, images, aug_probs):
        p = self.brightness
        if p == 0:
            return images
        
        b = torch.where(rand_bool(images.shape[0], p * self.p, device=images.device), rand_brightness(images.shape[0], images.device), torch.zeros(images.shape[0], device=images.device))
        return torch.clamp(images + b.view(-1, 1, 1, 1), -1., 1.)

    def apply_contrast(self, images, aug_probs):
        p = self.contrast
        if p == 0:
            return images
        
        c = torch.where(rand_bool(images.shape[0], p * self.p, device=images.device), rand_contrast(images.shape[0], images.device), torch.ones(images.shape[0], device=images.device))
        return torch.clamp(images * c.view(-1, 1, 1, 1), -1., 1.)

    def apply_saturation(self, images, aug_probs):
        p = self.saturation
        if p == 0:
            return images

        s = torch.where(rand_bool(images.shape[0], p * self.p, device=images.device), rand_saturation(images.shape[0], images.device), torch.ones(images.shape[0], device=images.device))
        
        # Convert to grayscale
        grayscale = images.mean(dim=1, keepdim=True) # Simple avg grayscale
        
        # Interpolate between grayscale and color
        return torch.clamp(torch.lerp(grayscale, images, s.view(-1, 1, 1, 1)), -1., 1.)

    def apply_cutout(self, images, aug_probs):
        p = self.cutout
        if p == 0:
            return images
        
        B, C, H, W = images.shape
        device = images.device

        mask = rand_bool(B, p * self.p, device)
        if not mask.any():
            return images

        # Create cutout masks
        cutout_h = int(H * self.cutout_size)
        cutout_w = int(W * self.cutout_size)
        
        # Random top-left corners for the cutout
        y0 = torch.randint(0, H - cutout_h + 1, (B,), device=device)
        x0 = torch.randint(0, W - cutout_w + 1, (B,), device=device)
        
        # Create grid for cutout
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=device),
            torch.arange(0, W, device=device),
            indexing='ij'
        )

        # Create mask where cutout should be applied
        cutout_mask = (grid_y >= y0.view(B, 1, 1)) & (grid_y < (y0 + cutout_h).view(B, 1, 1)) & \
                      (grid_x >= x0.view(B, 1, 1)) & (grid_x < (x0 + cutout_w).view(B, 1, 1))
        
        # Apply cutout only where the initial probability mask is true
        final_mask = cutout_mask & mask.view(B, 1, 1)

        # Fill with random noise where mask is true
        noise = torch.randn_like(images)
        
        return torch.where(final_mask.unsqueeze(1), noise, images) 