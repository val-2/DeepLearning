import torch
import torch.nn as nn
from model_blocks.text_encoder import TextEncoder


class Discriminator256(nn.Module):
    def __init__(self, text_dim=256, img_channels=3):
        super(Discriminator256, self).__init__()

        self.text_encoder = TextEncoder()  # Separate text encoder for discriminators

        self.img_path = nn.Sequential(
            # 256x256 -> 128x128
            nn.Conv2d(img_channels, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128 -> 64x64
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.text_path = nn.Sequential(
            nn.Linear(text_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512)
        )

        # Unconditional classifier (real/fake without text conditioning)
        self.unconditional_classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

        # Conditional classifier (text-conditioned real/fake)
        self.conditional_classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 512, 1024),  # size: sum of flattened image and text embedding
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

    def forward(self, images, text_features=None, text_mask=None, return_both=True):
        # Encode image
        img_features = self.img_path(images)
        img_features_flat = img_features.view(img_features.size(0), -1)  # Flatten

        unconditional_output = self.unconditional_classifier(img_features_flat)

        if not return_both:
            return unconditional_output

        if text_features is None or text_mask is None:
            raise AttributeError("text_features and text_mask necessary for text conditioning")

        # Encode text (mean pooling)
        global_full_text = self.text_encoder(text_features, text_mask)
        global_text = global_full_text.mean(dim=1)
        text_features_encoded = self.text_path(global_text)

        # Combine features
        combined = torch.cat([img_features_flat, text_features_encoded], dim=1)
        conditional_output = self.conditional_classifier(combined)

        return unconditional_output, conditional_output


class Discriminator64(nn.Module):
    def __init__(self, text_dim=256, img_channels=3):
        super(Discriminator64, self).__init__()

        self.text_encoder = TextEncoder()

        self.img_path = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(img_channels, 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Text encoder for discriminator
        self.text_path = nn.Sequential(
            nn.Linear(text_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512)
        )

        # Unconditional classifier (real/fake without text conditioning)
        self.unconditional_classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

        # Conditional classifier (text-conditioned real/fake)
        self.conditional_classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4 + 512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

    def forward(self, images, text_features=None, text_mask=None, return_both=True):
        img_features = self.img_path(images)
        img_features_flat = img_features.view(img_features.size(0), -1)  # Flatten

        unconditional_output = self.unconditional_classifier(img_features_flat)

        if not return_both:
            return unconditional_output

        if text_features is None or text_mask is None:
            raise AttributeError("text_features and text_mask necessary for text conditioning")


        # Encode text (mean pooling)
        global_full_text = self.text_encoder(text_features, text_mask)
        global_text = global_full_text.mean(dim=1)
        text_features_encoded = self.text_path(global_text)

        combined = torch.cat([img_features_flat, text_features_encoded], dim=1)
        conditional_output = self.conditional_classifier(combined)

        return unconditional_output, conditional_output
