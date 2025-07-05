import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 pretrained on ImageNet.
    We extract features at:
      - relu1_2  (index: 3)
      - relu2_2  (index: 8)
      - relu3_2  (index: 17)
      - relu4_2  (index: 26)
    Then compute L1 distance between those feature maps.
    Input images are in [-1,1]. We convert to [0,1], then normalize with ImageNet stats.
    """
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        # We only need layers up to 26 (relu4_2)
        self.slices = nn.ModuleDict({
            "relu1_2": nn.Sequential(*vgg19[:4]),     # conv1_1, relu1_1, conv1_2, relu1_2
            "relu2_2": nn.Sequential(*vgg19[4:9]),    # pool1, conv2_1, relu2_1, conv2_2, relu2_2
            "relu3_2": nn.Sequential(*vgg19[9:18]),   # pool2, conv3_1, relu3_1, conv3_2, relu3_2, ...
            "relu4_2": nn.Sequential(*vgg19[18:27])   # pool3, conv4_1, relu4_1, conv4_2, relu4_2
        })
        for param in self.parameters():
            param.requires_grad = False

        self.l1 = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img_gen, img_ref):
        """
        img_gen, img_ref: [B,3,H,W] in range [-1,1].
        Return: sum of L1 distances between VGG feature maps at chosen layers.
        """
        # Convert to [0,1]
        gen = (img_gen + 1.0) / 2.0
        ref = (img_ref + 1.0) / 2.0
        # Normalize
        gen_norm = (gen - self.mean.to(gen.device)) / self.std.to(gen.device)
        ref_norm = (ref - self.mean.to(ref.device)) / self.std.to(ref.device)

        loss = 0.0
        x_gen = gen_norm
        x_ref = ref_norm
        for slice_mod in self.slices.values():
            x_gen = slice_mod(x_gen)
            x_ref = slice_mod(x_ref)
            loss += self.l1(x_gen, x_ref)
        return loss


def gaussian(window_size, sigma):
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g


def create_window(window_size, channel, device):
    """
    Create a 2D Gaussian window to compute SSIM.
    window_size: e.g. 11
    channel: number of channels (3 for RGB)
    """
    _1D_window = gaussian(window_size, sigma=1.5).to(device).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()  # outer product → (window_size × window_size)
    window = _2D_window.unsqueeze(0).unsqueeze(0)  # shape (1,1,ws,ws)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss. Computes (1 - mean(SSIM map)) between two images.
    Assumes inputs are in [0,1].
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3  # assume 3 channels for RGB
        self.register_buffer("C1", torch.tensor((0.01 * 255) ** 2))
        self.register_buffer("C2", torch.tensor((0.03 * 255) ** 2))
        self.register_buffer("window", create_window(self.window_size, self.channel, device="cpu"))

    def _ssim_map(self, img1, img2, window):
        """
        Compute an SSIM map between img1 and img2 using a given window.
        img1, img2: [B, C, H, W], values in [0,1]
        window: [C,1,ws,ws]
        """
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        return ssim_map

    def forward(self, img_gen, img_ref):
        """
        img_gen, img_ref: [B,3,H,W] in range [-1,1]. We'll map to [0,1].
        Returns: 1 - mean(SSIM map over all pixels & channels).
        """
        # Convert from [-1,1] to [0,1]
        gen = (img_gen + 1.0) / 2.0
        ref = (img_ref + 1.0) / 2.0

        # Ensure window is on same device
        if self.window.device != gen.device:
             self.window = self.window.to(gen.device)

        # Compute SSIM map
        if gen.shape[1] != self.channel:
            # If not 3 channels, rebuild window for correct num channels
            channel = gen.shape[1]
            window = create_window(self.window_size, channel, gen.device)
            ssim_map = self._ssim_map(gen, ref, window)
        else:
            ssim_map = self._ssim_map(gen, ref, self.window)

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            # Return a per-batch-item SSIM loss
            return 1 - ssim_map.view(ssim_map.size(0), -1).mean(1)
