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
        vgg19_features = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        # We only need layers up to 26 (relu4_2)
        self.slices = nn.ModuleDict({
            "relu1_2": nn.Sequential(*list(vgg19_features.children())[:4]),     # conv1_1, relu1_1, conv1_2, relu1_2
            "relu2_2": nn.Sequential(*list(vgg19_features.children())[4:9]),    # pool1, conv2_1, relu2_1, conv2_2, relu2_2
            "relu3_2": nn.Sequential(*list(vgg19_features.children())[9:18]),   # pool2, conv3_1, relu3_1, conv3_2, relu3_2, ...
            "relu4_2": nn.Sequential(*list(vgg19_features.children())[18:27])   # pool3, conv4_1, relu4_1, conv4_2, relu4_2
        })
        for param in self.parameters():
            param.requires_grad = False

        self.l1 = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, img_gen, img_ref):
        """
        img_gen, img_ref: [B,3,H,W] in range [-1,1].
        Return: sum of L1 distances between VGG feature maps at chosen layers.
        """
        # Convert to [0,1]
        gen = (img_gen + 1.0) / 2.0
        ref = (img_ref + 1.0) / 2.0
        # Normalize
        gen_norm = (gen - self.mean) / self.std
        ref_norm = (ref - self.mean) / self.std

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

        C1 = self.C1.to(img1.device)
        C2 = self.C2.to(img1.device)
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)) # type: ignore
        return ssim_map

    def forward(self, img_gen, img_ref):
        """
        img_gen, img_ref: [B,3,H,W] in range [-1,1]. We'll map to [0,1].
        Returns: 1 - mean(SSIM map over all pixels & channels).
        """
        # Convert from [-1,1] to [0,1]
        gen = (img_gen + 1.0) / 2.0
        ref = (img_ref + 1.0) / 2.0

        # The window buffer is assumed to have been moved to the correct device
        # by calling .to(device) on the loss module itself.
        window = self.window

        # Compute SSIM map
        if gen.shape[1] != self.channel:
            # If not 3 channels, rebuild window for correct num channels
            channel = gen.shape[1]
            # This new window is created on the correct device directly
            window = create_window(self.window_size, channel, gen.device)
            ssim_map = self._ssim_map(gen, ref, window)
        else:
            # Use the pre-computed window from the buffer
            ssim_map = self._ssim_map(gen, ref, window)

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            # Return a per-batch-item SSIM loss
            return 1 - ssim_map.view(ssim_map.size(0), -1).mean(1)


# ----------------------------------------------------------
# PatchGAN Discriminator
# ----------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator that outputs a 2D patch-wise prediction map
    instead of a single scalar. Each patch prediction corresponds to
    whether that patch in the input image is real or fake.
    """
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult, affine=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------------------
# PatchNCE (Contrastive) Loss
# ----------------------------------------------------------

class PatchNCELoss(nn.Module):
    """
    Implements the PatchNCE (InfoNCE) loss as used in CUT (Contrastive Unpaired Translation).
    This encourages the generated image to have similar patch-level features to the input image
    while being different from patches at other spatial locations.
    """
    def __init__(self, nce_T=0.07, feat_dim=256, num_patches=256):
        super(PatchNCELoss, self).__init__()
        self.nce_T = nce_T  # Temperature parameter for contrastive learning
        self.num_patches = num_patches  # Number of patches to sample per layer
        self.cross_entropy = nn.CrossEntropyLoss()

    def _patch_correlation(self, feat_q, feat_k, MLP):
        """
        Compute patch-wise contrastive loss between query and key features.

        Args:
            feat_q: Query features [B, C, H, W]
            feat_k: Key features [B, C, H, W]
            MLP: Projection head to map features to embedding space

        Returns:
            NCE loss for this feature layer
        """
        B, C, H, W = feat_q.shape
        feat_q = feat_q.view(B, C, -1)   # [B, C, L] where L = H*W
        feat_k = feat_k.view(B, C, -1)   # [B, C, L]
        L = H * W

        # Randomly sample patches to avoid memory issues with large feature maps
        idxs = torch.randperm(L, device=feat_q.device)[:self.num_patches]
        fq = feat_q[:, :, idxs]  # [B, C, N] where N = num_patches
        fk = feat_k[:, :, idxs]  # [B, C, N]

        # Reshape for MLP projection: [B, N, C] -> [B*N, C]
        fq = fq.permute(0, 2, 1).reshape(-1, C)  # [B*N, C]
        fk = fk.permute(0, 2, 1).reshape(-1, C)  # [B*N, C]

        # Project to embedding space and normalize
        q_proj = MLP(fq)  # [B*N, feat_dim]
        k_proj = MLP(fk)  # [B*N, feat_dim]
        q_norm = F.normalize(q_proj, dim=1)
        k_norm = F.normalize(k_proj, dim=1)

        # Compute similarity matrix and apply temperature
        logits = (q_norm @ k_norm.t()) / self.nce_T  # [B*N, B*N]

        # Labels: each query should match with its corresponding key (diagonal)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = self.cross_entropy(logits, labels)
        return loss

    def forward(self, feats_q, feats_k, MLPs):
        """
        Compute PatchNCE loss across multiple feature layers.

        Args:
            feats_q: Dictionary of query features {layer_name: tensor}
            feats_k: Dictionary of key features {layer_name: tensor}
            MLPs: Dictionary of projection heads {layer_name: MLP}

        Returns:
            Average PatchNCE loss across all layers
        """
        total_loss = 0.0
        for layer_name in feats_q.keys():
            total_loss += self._patch_correlation(
                feats_q[layer_name],
                feats_k[layer_name],
                MLPs[layer_name]
            )
        return total_loss / len(feats_q)


# ----------------------------------------------------------
# Sobel Loss for Edge Preservation
# ----------------------------------------------------------

class SobelLoss(nn.Module):
    """
    Computes the Sobel loss between two images, which encourages edge similarity.
    This loss operates on the grayscale versions of the input images.
    """
    def __init__(self):
        super(SobelLoss, self).__init__()
        # Sobel kernels for edge detection
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)
        self.l1 = nn.L1Loss()

        # Grayscale conversion weights (ITU-R BT.601)
        self.register_buffer("rgb_to_gray_weights", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def _get_edges(self, img):
        """
        Converts an RGB image to grayscale and applies Sobel filters.
        Args:
            img: [B, 3, H, W] image tensor in range [-1, 1].
        Returns:
            Gradient magnitude map [B, 1, H, W].
        """
        # Ensure input is 4D
        if img.dim() != 4:
            raise ValueError(f"Expected 4D input (got {img.dim()}D)")

        # Convert from [-1, 1] to [0, 1]
        img = (img + 1.0) / 2.0

        # Convert to grayscale
        # The weights need to be on the same device as the image.
        grayscale_img = F.conv2d(img, self.rgb_to_gray_weights.to(img.device)) # type: ignore

        # Apply Sobel filters. Kernels also need to be on the correct device.
        grad_x = F.conv2d(grayscale_img, self.kernel_x.to(img.device), padding=1) # type: ignore
        grad_y = F.conv2d(grayscale_img, self.kernel_y.to(img.device), padding=1) # type: ignore

        # Compute gradient magnitude
        edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6) # add epsilon for stability
        return edges

    def forward(self, img_gen, img_ref):
        """
        img_gen, img_ref: [B, 3, H, W] in range [-1, 1].
        Returns: L1 loss between the edge maps of the two images.
        """
        edges_gen = self._get_edges(img_gen)
        edges_ref = self._get_edges(img_ref)
        return self.l1(edges_gen, edges_ref)
