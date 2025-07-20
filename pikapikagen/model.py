import torch
import torch.nn as nn
from model_blocks.text_encoder import TextEncoder
from model_blocks.image_decoder import ImageDecoder

class Generator(nn.Module):
    """
    Modello completo che unisce Encoder e Decoder.
    """
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
        # Genera rumore casuale per il batch
        batch_size = token_ids.size(0)
        # noise.shape: (batch_size, noise_dim)
        noise = torch.randn(batch_size, self.noise_dim, device=token_ids.device)

        # 1. Codifica il testo per ottenere i vettori di ogni parola
        # encoder_output.shape: (batch_size, seq_len, text_embed_dim)
        encoder_output = self.text_encoder(token_ids, attention_mask=attention_mask)

        # 2. Genera l'immagine usando l'output completo dell'encoder
        #    Il decoder calcolerà internamente sia il contesto iniziale (ATTENZIONE #1)
        #    sia l'attenzione per-step (ATTENZIONE #2)
        # generated_image_256.shape: (batch_size, 3, 256, 256)
        # generated_image_64.shape: (batch_size, 3, 64, 64)
        generated_image_256, generated_image_64, attention_maps, initial_attention_weights = self.image_decoder(noise, encoder_output, attention_mask)

        if return_attentions:
            return generated_image_256, generated_image_64, attention_maps, initial_attention_weights
        return generated_image_256, generated_image_64
