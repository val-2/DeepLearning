import torch
import torch.nn as nn
import numpy as np

# A building block for the discriminator, inspired by StyleGAN2.
# It includes residual connections and downsampling.
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.downsampler = nn.AvgPool2d(2)

    def forward(self, x):
        shortcut = self.downsampler(self.shortcut(x))
        residual = self.residual(x)
        residual = self.downsampler(residual)
        
        # Add residual connection
        x = (shortcut + residual) / np.sqrt(2)
        
        return x

# Adds a feature map representing the standard deviation across the minibatch.
# This helps prevent mode collapse by encouraging variety.
class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # N, C, H, W
        B, C, H, W = x.shape
        # Calculate standard deviation over the batch dimension
        # Keep dimensions for broadcasting, group over channels and spatial dims
        std = torch.sqrt(torch.mean((x - torch.mean(x, dim=0, keepdim=True))**2, dim=0, keepdim=True))
        # Calculate the mean of the standard deviation
        mean_std = torch.mean(std)
        # Create a new feature map with this value and concatenate it
        std_feature = mean_std.expand(B, 1, H, W)
        return torch.cat([x, std_feature], dim=1)

class PikaPikaDisc(nn.Module):
    """
    Discriminator model for PikaPikaGen GAN.
    It's a conditional discriminator inspired by StyleGAN2-ADA architecture.
    It takes an image and text features, and outputs a single 'realness' score.
    """
    def __init__(self, text_embed_dim=256, image_size=256):
        super().__init__()

        # Channel configuration for each resolution block
        channels = {
            256: 32,
            128: 64,
            64: 128,
            32: 256,
            16: 512,
            8: 512,
            4: 512
        }

        # Input layer to convert RGB image to feature space
        self.from_rgb = nn.Sequential(
            nn.Conv2d(3, channels[image_size], kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Main backbone of downsampling blocks
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(channels[256], channels[128]),  # 256x256 -> 128x128
            DiscriminatorBlock(channels[128], channels[64]),   # 128x128 -> 64x64
            DiscriminatorBlock(channels[64], channels[32]),    # 64x64   -> 32x32
            DiscriminatorBlock(channels[32], channels[16]),    # 32x32   -> 16x16
        ])

        # Text conditioning is projected and injected at the 16x16 resolution stage.
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, channels[16]),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # The block after text injection needs to handle the concatenated channels
        self.conditional_block = DiscriminatorBlock(channels[16] * 2, channels[8]) # 16x16 -> 8x8

        # Block after conditional block to get to 4x4
        self.final_downsample = DiscriminatorBlock(channels[8], channels[4])

        self.final_block = nn.Sequential(
            MinibatchStdDev(),
            # Input channels are from the final downsample output + 1 (for std dev)
            nn.Conv2d(channels[4] + 1, channels[4], kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            # The input features to the linear layer depend on the final feature map size (4x4)
            nn.Linear(channels[4] * 4 * 4, 1)
        )

    def forward(self, image, text_features):
        # image: (B, 3, 256, 256)
        # text_features: (B, embed_dim) - this should be the global context vector
        
        x = self.from_rgb(image)

        for block in self.blocks:
            x = block(x) # x is now 16x16

        # Project text features and inject them
        projected_text = self.text_projection(text_features)
        # expand text features to match spatial dimensions
        projected_text = projected_text.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        
        # Concatenate along channel dimension
        x = torch.cat([x, projected_text], dim=1)
        
        # Pass through the rest of the network
        x = self.conditional_block(x) # 8x8
        x = self.final_downsample(x)  # 4x4
        x = self.final_block(x) # final score

        return x 