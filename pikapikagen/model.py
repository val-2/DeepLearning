import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    Encoder per processare il testo.
    Usa gli embedding di bert-mini e li fa passare in un Transformer.
    """
    def __init__(self, model_name="prajjwal1/bert-mini", fine_tune_embeddings=True):
        super().__init__()
        # Carica il modello bert-mini pre-addestrato per estrarre gli embedding
        bert_mini_model = AutoModel.from_pretrained(model_name)

        # Estrae lo strato di embedding
        self.embedding = bert_mini_model.embeddings

        # Imposta se fare il fine-tuning degli embedding durante il training
        for param in self.embedding.parameters():
            param.requires_grad = fine_tune_embeddings

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, token_ids):
        # 1. Ottieni gli embedding dai token ID
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded_text = self.embedding(token_ids)

        # 2. Passa gli embedding attraverso il Transformer Encoder
        # Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        encoder_output = self.transformer_encoder(embedded_text)
        return encoder_output


class CrossAttention(nn.Module):
    """
    Modulo di Cross-Attention.
    Permette a una sequenza di query (dall'immagine) di "prestare attenzione"
    a una sequenza di key/value (dal testo), gestendo internamente
    il reshaping dei tensori e la maschera di padding.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, key_padding_mask=None):
        # query: (B, C, H, W) - Feature dell'immagine (spaziale)
        # key/value: (B, seq_len, embed_dim) - Output dell'encoder di testo
        # key_padding_mask: (B, seq_len) - Maschera dal tokenizer

        B, C, H, W = query.shape

        # 1. Prepara la query (feature dell'immagine)
        # Reshape da spaziale a sequenza: (B, C, H, W) -> (B, H*W, C)
        query_seq = query.view(B, C, H * W).permute(0, 2, 1)
        query_norm = self.layer_norm(query_seq)

        # 2. Prepara la maschera di padding per l'attenzione
        # La maschera di HuggingFace è 1 per i token reali, 0 per il padding.
        # MultiheadAttention si aspetta True per le posizioni da ignorare.
        if key_padding_mask is not None:
            mask = (key_padding_mask == 0)
        else:
            mask = None

        # 3. Applica l'attenzione
        attn_output, attn_weights = self.attention(
            query=query_norm,
            key=key,
            value=value,
            key_padding_mask=mask,
            need_weights=True
        )
        # attn_output: (B, H*W, C)

        # 4. Riconverti l'output nella forma spaziale originale
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        attn_output_spatial = attn_output.permute(0, 2, 1).view(B, C, H, W)

        return attn_output_spatial, attn_weights


class DecoderBlock(nn.Module):
    """
    U-Net-style decoder block: Upsampling -> Double Convolution.
    """
    def __init__(self, in_channels, out_channels, use_attention=True, text_embed_dim=256, nhead=4):
        super().__init__()
        self.use_attention = use_attention

        if self.use_attention:
            if in_channels != text_embed_dim:
                raise ValueError("in_channels must be equal to text_embedding_dim for attention.")
            self.cross_attention = CrossAttention(embed_dim=in_channels, num_heads=nhead)

        # 1. Upsampling Layer: We use a simple ConvTranspose2d to double the spatial dimensions.
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # 2. Double Convolution Block: Two standard convolutions to process features at the new scale.
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, encoder_output=None, attention_mask=None):
        attn_weights = None
        if self.use_attention:
            if encoder_output is None:
                raise ValueError("encoder_output must be provided when use_attention is True.")
            if attention_mask is None:
                raise ValueError("attention_mask must be provided when use_attention is True.")

            # La cross-attention ora gestisce internamente il reshaping e la maschera
            attn_output, attn_weights = self.cross_attention(
                query=x,
                key=encoder_output,
                value=encoder_output,
                key_padding_mask=attention_mask
            )
            x = x + attn_output # Additive (residual) connection

        # Apply the U-Net style sequence
        x = self.up_conv(x)
        x = self.conv_block(x)
        return x, attn_weights


class ImageDecoder(nn.Module):
    """
    Decoder CNN (Generatore) che sintetizza l'immagine.
    Questa versione usa l'attenzione per-step fin dall'inizio.
    """
    def __init__(self, noise_dim, text_embed_dim, final_image_channels=3):
        super().__init__()

        # Meccanismo per calcolare il contesto iniziale come media pesata
        self.initial_context_attention = nn.Sequential(
            nn.Linear(in_features=text_embed_dim, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=1),
            nn.Softmax(dim=1)
        )

        # Proiezione iniziale direttamente a 256 canali per l'attenzione.
        # Input: (B, noise_dim + text_embed_dim) -> Output: (B, 256 * 4 * 4)
        self.initial_projection = nn.Linear(noise_dim + text_embed_dim, 256 * 4 * 4)

        # Blocchi del decoder U-Net style, con attenzione fin dall'inizio.
        self.blocks = nn.ModuleList([
            # Input: (B, 256, 4, 4)   -> Output: (B, 256, 8, 8)
            DecoderBlock(in_channels=256, out_channels=256, use_attention=True),
            # Input: (B, 256, 8, 8)   -> Output: (B, 256, 16, 16)
            DecoderBlock(in_channels=256, out_channels=256, use_attention=True),
            # Input: (B, 256, 16, 16)  -> Output: (B, 256, 32, 32)
            DecoderBlock(in_channels=256, out_channels=256, use_attention=True),
            # Input: (B, 256, 32, 32)  -> Output: (B, 128, 64, 64)
            DecoderBlock(in_channels=256, out_channels=128, use_attention=True),
            # Input: (B, 128, 64, 64)  -> Output: (B, 64, 128, 128)
            DecoderBlock(in_channels=128, out_channels=64, use_attention=False),
            # Input: (B, 64, 128, 128) -> Output: (B, 32, 256, 256)
            DecoderBlock(in_channels=64, out_channels=32, use_attention=False),
        ])

        # Layer finale per portare ai canali RGB
        # Input: (B, 32, 256, 256) -> Output: (B, 3, 256, 256)
        self.final_conv = nn.Conv2d(32, final_image_channels, kernel_size=3, padding=1)
        self.final_activation = nn.Tanh()

    def forward(self, noise, encoder_output_full, attention_mask):
        # noise.shape: (B, noise_dim)
        # encoder_output_full.shape: (B, seq_len, text_embed_dim)
        # attention_mask.shape: (B, seq_len)

        # 1. Calcola il vettore di contesto iniziale con una media pesata (ATTENZIONE #1)
        # attention_weights.shape: (B, seq_len, 1)
        attention_weights = self.initial_context_attention(encoder_output_full)
        # context_vector.shape: (B, text_embed_dim)
        context_vector = torch.sum(attention_weights * encoder_output_full, dim=1)

        # 2. Prepara il vettore di input iniziale per la proiezione
        # initial_input.shape: (B, noise_dim + text_embed_dim)
        initial_input = torch.cat([noise, context_vector], dim=1)

        # 3. Proietta e rimodella
        # x.shape: (B, 256 * 4 * 4)
        x = self.initial_projection(initial_input)
        # x.shape: (B, 256, 4, 4)
        x = x.view(x.size(0), 256, 4, 4)

        # 4. Passa attraverso i blocchi del decoder
        attention_maps = []
        for block in self.blocks:
             encoder_ctx = encoder_output_full if block.use_attention else None
             mask_ctx = attention_mask if block.use_attention else None
             # La shape di x viene upsamplata in ogni blocco (es. 4x4 -> 8x8)
             x, attn_weights = block(x, encoder_ctx, mask_ctx)

             if attn_weights is not None:
                # attn_weights.shape: (B, H*W, seq_len)
                attention_maps.append(attn_weights)

        # 5. Layer finale
        # x.shape: (B, 3, 256, 256)
        x = self.final_conv(x)
        # x.shape: (B, 3, 256, 256)
        x = self.final_activation(x)
        return x, attention_maps


class PikaPikaGen(nn.Module):
    """
    Modello completo che unisce Encoder e Decoder.
    """
    def __init__(self, text_encoder_model_name="prajjwal1/bert-mini", noise_dim=100, fine_tune_embeddings=True):
        super().__init__()
        self.text_encoder = TextEncoder(
            model_name=text_encoder_model_name,
            fine_tune_embeddings=fine_tune_embeddings
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
        encoder_output = self.text_encoder(token_ids)

        # 2. Genera l'immagine usando l'output completo dell'encoder
        #    Il decoder calcolerà internamente sia il contesto iniziale (ATTENZIONE #1)
        #    sia l'attenzione per-step (ATTENZIONE #2)
        # generated_image.shape: (batch_size, 3, 256, 256)
        generated_image, attention_maps = self.image_decoder(noise, encoder_output, attention_mask)

        if return_attentions:
            return generated_image, attention_maps
        return generated_image
