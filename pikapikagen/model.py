import torch
import torch.nn as nn
from model_blocks.text_encoder import TextEncoder
from model_blocks.image_decoder import ImageDecoder

class Generator(nn.Module):
    def __init__(self, text_encoder_model_name="prajjwal1/bert-mini", noise_dim=100):
        super().__init__()
        self.text_encoder = TextEncoder(
            model_name=text_encoder_model_name,
        )

        text_embed_dim = 256

        self.image_decoder = ImageDecoder(
            noise_dim=noise_dim,
            text_embed_dim=text_embed_dim
        )

        self.noise_dim = noise_dim

    def forward(self, token_ids, attention_mask, return_attentions=False):
        # token_ids.shape: (batch_size, seq_len)
        # attention_mask.shape: (batch_size, seq_len)
        # Generates random noise for the batch
        batch_size = token_ids.size(0)
        # noise.shape: (batch_size, noise_dim)
        noise = torch.randn(batch_size, self.noise_dim, device=token_ids.device)

        # 1. Encodes the text to get the vectors of each word
        # encoder_output.shape: (batch_size, seq_len, text_embed_dim)
        encoder_output = self.text_encoder(token_ids, attention_mask=attention_mask)

        # 2. Generates the image using the complete output of the encoder
        #    The decoder will internally calculate both the initial context (ATTENTION #1)
        #    and the per-step attention (ATTENTION #2)
        # generated_image_256.shape: (batch_size, 3, 256, 256)
        # generated_image_64.shape: (batch_size, 3, 64, 64)
        generated_image_256, generated_image_64, attention_maps, initial_attention_weights = self.image_decoder(noise, encoder_output, attention_mask)

        if return_attentions:
            return generated_image_256, generated_image_64, attention_maps, initial_attention_weights
        return generated_image_256, generated_image_64
