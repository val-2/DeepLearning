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


class SobelLoss(nn.Module):
    """
    Computes the Sobel loss between two images, which encourages edge similarity.
    This loss operates on the grayscale versions of the input images.
    """
    def __init__(self):
        super(SobelLoss, self).__init__()
        # Sobel kernels for edge detection
        self.kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.l1 = nn.L1Loss()

        # Grayscale conversion weights (ITU-R BT.601)
        self.rgb_to_gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)

    def _get_edges(self, img):
        """
        Converts an RGB image to grayscale and applies Sobel filters.
        Args:
            img: [B, 3, H, W] image tensor in range [-1, 1].
        Returns:
            Gradient magnitude map [B, 1, H, W].
        """

        # Convert from [-1, 1] to [0, 1]
        img = (img + 1.0) / 2.0

        # Convert to grayscale
        grayscale_img = F.conv2d(img, self.rgb_to_gray_weights.to(img.device))

        # Apply Sobel filters
        grad_x = F.conv2d(grayscale_img, self.kernel_x.to(img.device), padding=1)
        grad_y = F.conv2d(grayscale_img, self.kernel_y.to(img.device), padding=1)

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
