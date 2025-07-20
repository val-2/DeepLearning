import torch
import torch.nn as nn
from model_blocks.image_cross_attention import ImageCrossAttention

class DecoderBlock(nn.Module):
    """
    Image decoder block
    Channel adaptation (if necessary) -> Attention (optional) -> Merge -> Residual connection
    -> Upsampling (ConvTranspose) -> Normalization -> Activation.
    """
    def __init__(self, in_channels, out_channels, use_attention=True, text_embed_dim=256, nhead=4):
        super().__init__()
        self.use_attention = use_attention

        if self.use_attention:
            # If in_channels is different from text_embed_dim, add a 1x1 conv to adapt the channel size
            if in_channels != text_embed_dim:
                self.channel_adapter = nn.Conv2d(in_channels, text_embed_dim, kernel_size=1, bias=False)
            else:
                self.channel_adapter = None

            self.cross_attention = ImageCrossAttention(embed_dim=text_embed_dim, num_heads=nhead)
            # Convolution to merge the text_embedding and the cross-attention output
            self.fusion_conv = nn.Conv2d(text_embed_dim * 2, in_channels, kernel_size=1, bias=False)

        # Upsample block as described in the instructions
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, out_channels), # Equivalent to LayerNorm for feature map (N, C, H, W)
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, encoder_output=None, attention_mask=None):
        attn_weights = None
        if self.use_attention:
            if encoder_output is None or attention_mask is None:
                raise ValueError("encoder_output and attention_mask must be provided for attention.")

            # Adapt channel size if deemed necessary
            if self.channel_adapter is not None:
                x_adapted = self.channel_adapter(x)
            else:
                x_adapted = x

            attn_output, attn_weights = self.cross_attention(
                image_features=x_adapted,
                text_features=encoder_output,
                key_padding_mask=attention_mask
            )

            # Concatenates the features with the cross-attention output,
            # then conv 1x1 and residual connection
            fused_features = torch.cat([x_adapted, attn_output], dim=1) # Shape: (B, 2*in_channels, H, W)
            skip = self.fusion_conv(fused_features) # Shape: (B, in_channels, H, W)
            x = x + skip  # Shape: (B, in_channels, H, W)


        x = self.upsample_block(x)
        return x, attn_weights
