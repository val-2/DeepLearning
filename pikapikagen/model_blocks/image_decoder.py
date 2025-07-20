import torch
import torch.nn as nn
from model_blocks.decoder_block import DecoderBlock

class ImageDecoder(nn.Module):
    """
    Decoder CNN (Generatore) che sintetizza l'immagine.
    Questa versione usa l'attenzione per-step fin dall'inizio.
    """
    def __init__(self, noise_dim, text_embed_dim, final_image_channels=3):
        super().__init__()

        # Mechanism to calculate attention scores for the initial context.
        self.initial_context_scorer = nn.Sequential(
            nn.Linear(in_features=text_embed_dim, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=1)
            # Softmax applied in forward pass to use the attention mask
        )

        # Initial linear projection to a 4x4 feature map.
        self.initial_projection = nn.Sequential(
            nn.Linear(noise_dim + text_embed_dim, 256 * 4 * 4),
            nn.GroupNorm(1, 256 * 4 * 4),
            nn.LeakyReLU(inplace=True)
        )

        # Shared blocks for both resolutions (until 64x64)
        self.blocks_64 = nn.ModuleList([
            # Input: (B, 256, 4, 4)   -> Output: (B, 256, 8, 8)
            DecoderBlock(in_channels=256, out_channels=256, use_attention=True),
            # Input: (B, 256, 8, 8)   -> Output: (B, 256, 16, 16)
            DecoderBlock(in_channels=256, out_channels=256, use_attention=True),
            # Input: (B, 256, 16, 16)  -> Output: (B, 128, 32, 32)
            DecoderBlock(in_channels=256, out_channels=128, use_attention=True),
            # Input: (B, 128, 32, 32)  -> Output: (B, 64, 64, 64)
            DecoderBlock(in_channels=128, out_channels=64, use_attention=False),
        ])

        # ModuleList is used instead of a Sequential for example because
        # of the branching based on use_attention in the forward pass

        # Blocks only for 256x256 (from 64x64 to 256x256)
        self.blocks_256 = nn.ModuleList([
            # Input: (B, 64, 64, 64)  -> Output: (B, 32, 128, 128)
            DecoderBlock(in_channels=64, out_channels=32, use_attention=True),
            # Input: (B, 32, 128, 128) -> Output: (B, 16, 256, 256)
            DecoderBlock(in_channels=32, out_channels=16, use_attention=False),
        ])

        # Last layer to get to RGB channels - 256x256
        # Input: (B, 16, 256, 256) -> Output: (B, 3, 256, 256)
        self.final_conv_256 = nn.Conv2d(16, final_image_channels, kernel_size=3, padding=1)
        self.final_activation_256 = nn.Tanh()

        # Last layer to get to RGB channels - 64x64
        # Input: (B, 64, 64, 64) -> Output: (B, 3, 64, 64)
        self.final_conv_64 = nn.Conv2d(64, final_image_channels, kernel_size=3, padding=1)
        self.final_activation_64 = nn.Tanh()

    def forward(self, noise, encoder_output_full, attention_mask):
        # noise.shape: (B, noise_dim)
        # encoder_output_full.shape: (B, seq_len, text_embed_dim)
        # attention_mask.shape: (B, seq_len)

        # 1. Compute the first attention, with the scores (logits) for each token
        attn_scores = self.initial_context_scorer(encoder_output_full)

        # Apply attention mask before Softmax.
        # Set the scores of the padding tokens to -inf.
        # The mask is (B, seq_len), the scores (B, seq_len, 1)
        # The broadcast takes care of the dimensions.
        attn_scores.masked_fill_(attention_mask.unsqueeze(-1) == 0, -1e9)

        # attention_weights.shape: (B, seq_len, 1)
        attention_weights = torch.softmax(attn_scores, dim=1)

        # Weighted average of the encoder output
        # context_vector.shape: (B, text_embed_dim)
        context_vector = torch.sum(attention_weights * encoder_output_full, dim=1)

        # 2. Merge the noise and the context vector for the initial projection
        # initial_input.shape: (B, noise_dim + text_embed_dim)
        initial_input = torch.cat([noise, context_vector], dim=1)

        # 3. Initial projection and reshape to fit the transposed convolutions
        # x.shape: (B, 256 * 4 * 4)
        x = self.initial_projection(initial_input)
        # x.shape: (B, 256, 4, 4)
        x = x.view(x.size(0), 256, 4, 4)

        # 4. Pass through the encoder blocks
        attention_maps = []

        # Shared path for both resolutions (fino a 64x64)
        for block in self.blocks_64:
            encoder_ctx = encoder_output_full if block.use_attention else None
            mask_ctx = attention_mask if block.use_attention else None
            x, attn_weights = block(x, encoder_ctx, mask_ctx)
            if attn_weights is not None:
                attention_maps.append(attn_weights)

        # Now x has size (B, 64, 64, 64)

        # 64x64-only path
        image_64 = self.final_conv_64(x)
        image_64 = self.final_activation_64(image_64)

        # 5. 256x256-only path
        for block in self.blocks_256:
            encoder_ctx = encoder_output_full if block.use_attention else None
            mask_ctx = attention_mask if block.use_attention else None
            x, attn_weights = block(x, encoder_ctx, mask_ctx)
            if attn_weights is not None:
                attention_maps.append(attn_weights)

        # Final layer for 256x256
        # x_256.shape: (B, 16, 256, 256) -> (B, 3, 256, 256)
        image_256 = self.final_conv_256(x)
        image_256 = self.final_activation_256(image_256)

        return image_256, image_64, attention_maps, attention_weights
