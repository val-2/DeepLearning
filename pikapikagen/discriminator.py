import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    Conditional PatchGAN Discriminator.
    It uses a PatchGAN architecture to classify whether patches of an image are real or fake,
    conditioned by a context vector (textual embedding).
    """
    def __init__(self, img_channels=3, features_d=64, n_layers=3, text_embed_dim=256):
        """
        Args:
            img_channels (int): Image channels (e.g., 3 for RGB).
            features_d (int): Number of features in the first convolutional layer.
            n_layers (int): Number of convolutional layers in the discriminator's image path.
            text_embed_dim (int): Dimension of the textual embedding.
        """
        super().__init__()

        # Part of the network that processes the image to an intermediate size.
        # E.g., 256x256 -> 32x32
        image_path_sequence = [
            self._block(img_channels, features_d, 4, 2, 1, use_norm=False), # 64x128x128
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            image_path_sequence += [
                self._block(features_d * nf_mult_prev, features_d * nf_mult, 4, 2, 1) # 128x64x64 -> 256x32x32
            ]
        self.image_path = nn.Sequential(*image_path_sequence)

        last_img_feature_channels = features_d * nf_mult

        # Project the text context vector to be compatible.
        self.text_projection = nn.Sequential(
            nn.Linear(in_features=text_embed_dim, out_features=last_img_feature_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final part that processes the combined features.
        combined_input_channels = last_img_feature_channels * 2

        self.combined_path = nn.Sequential(
             self._block(combined_input_channels, features_d * min(2**n_layers, 8), 4, 1, 1),
             nn.Conv2d(features_d * min(2**n_layers, 8), 1, kernel_size=4, stride=1, padding=1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, use_norm=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
        if use_norm:
            # Using InstanceNorm as in the original PatchGAN.
            layers.append(nn.InstanceNorm2d(out_channels, affine=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, context_vector):
        """
        Args:
            x (Tensor): Input image (real or generated). Shape: (B, C, H, W).
            context_vector (Tensor): Text context vector. Shape: (B, text_embed_dim).

        Returns:
            Tensor: Prediction map (patch). Shape: (B, 1, H', W').
        """
        # 1. Process the image to extract features.
        img_features = self.image_path(x)

        # 2. Project the text context and expand it to spatial dimensions.
        text_features = self.text_projection(context_vector)
        text_features_expanded = text_features.unsqueeze(-1).unsqueeze(-1).expand_as(img_features)

        # 3. Concatenate the image and text features.
        combined_features = torch.cat([img_features, text_features_expanded], dim=1)

        # 4. Process the combined features to get the patch map.
        output = self.combined_path(combined_features)

        return output
