import torch.nn as nn

class PikaPikaDisc(nn.Module):
    """
    Conditional PatchGAN Discriminator.
    It uses a PatchGAN architecture to classify whether patches of an image are real or fake,
    conditioned by a context vector (textual embedding).
    """
    def __init__(self, img_channels=3):
        """
        Args:
            img_channels (int): Image channels (e.g., 3 for RGB).
        """
        super().__init__()

        self.image_path = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input image (real or generated). Shape: (B, C, H, W).


        Returns:
            Tensor: Prediction map (patch). Shape: (B, 1, H', W').
        """
        return self.image_path(x)
