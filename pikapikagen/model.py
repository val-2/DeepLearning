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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

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
    a una sequenza di key/value (dal testo).
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, query, key, value):
        # query: (batch_size, query_len, embed_dim) - Feature dell'immagine
        # key/value: (batch_size, kv_len, embed_dim) - Output dell'encoder di testo
        attn_output, attn_weights = self.attention(query, key, value, need_weights=True)
        return attn_output, attn_weights


class DecoderBlock(nn.Module):
    """
    Blocco del decoder con upsampling, attenzione e convoluzioni.
    """
    def __init__(self, in_channels, out_channels, text_embedding_dim=256, nhead=4, use_attention=True,
                 conv_kernel_size=4, conv_stride=2, conv_padding=1):
        super().__init__()
        self.use_attention = use_attention

        if self.use_attention:
            if in_channels != text_embedding_dim:
                raise ValueError("in_channels must be equal to text_embedding_dim for attention.")
            self.cross_attention = CrossAttention(embed_dim=text_embedding_dim, num_heads=nhead)
            # Layer per normalizzare le feature prima dell'attenzione
            self.layer_norm = nn.LayerNorm(in_channels)

        self.transpose_conv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, encoder_output=None):
        # x: image (Batch, Channels, Height, Width)
        # encoder_output: (Batch, Sequence Length, Text Embedding Dim)
        attn_weights = None # Inizializza a None

        if self.use_attention:
            if encoder_output is None:
                raise ValueError("encoder_output must be provided when use_attention is True.")

            B, C, H, W = x.shape

            # 1. Applica l'attenzione
            x_norm = self.layer_norm(x.reshape(B, C, H * W).permute(0, 2, 1)) # -> (B, H*W, C)
            # We need to transpose the x_norm because the MultiheadAttention expects (B, L, C)
            attn_output, attn_weights = self.cross_attention(query=x_norm, key=encoder_output, value=encoder_output)
            attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W) # -> (B, C, H, W)

            # Connessione residua
            x = x + attn_output

        # 2. Applica l'upsampling con convoluzione trasposta
        x = self.transpose_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x, attn_weights


class ImageDecoder(nn.Module):
    """
    Decoder CNN (Generatore) che sintetizza l'immagine.
    """
    def __init__(self, noise_dim, text_embed_dim, final_image_channels=3):
        super().__init__()

        # Proiezione iniziale: combina rumore e contesto testuale
        self.initial_projection = nn.Linear(noise_dim + text_embed_dim, 256 * 4 * 4)

        # Blocchi del decoder per l'upsampling.
        # La sequenza è stata approfondita per rendere l'upsampling più graduale
        # e per mantenere più canali nelle prime fasi.
        self.blocks = nn.ModuleList([
            # 4x4 -> 8x8
            DecoderBlock(in_channels=256, out_channels=256, use_attention=True),
            # 8x8 -> 16x16
            DecoderBlock(in_channels=256, out_channels=256, use_attention=True),
            # 16x16 -> 32x32
            DecoderBlock(in_channels=256, out_channels=256, use_attention=True),
            # 32x32 -> 64x64
            DecoderBlock(in_channels=256, out_channels=128, use_attention=True),
            # 64x64 -> 128x128
            DecoderBlock(in_channels=128, out_channels=64, use_attention=False),
            # 128x128 -> 256x256
            DecoderBlock(in_channels=64, out_channels=32, use_attention=False),
        ])

        # Layer finale per portare ai canali RGB
        # Da 256x256 a 256x256 (solo cambio di canali)
        self.final_conv = nn.Conv2d(32, final_image_channels, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Tanh()

    def forward(self, noise, context_vector, encoder_output_full):
        # 1. Prepara il vettore di input iniziale
        # Usa il context_vector già calcolato con la media pesata
        initial_input = torch.cat([noise, context_vector], dim=1)

        # 2. Proietta e rimodella per iniziare la sequenza di convoluzioni
        x = self.initial_projection(initial_input)
        x = x.view(x.size(0), 256, 4, 4) # -> (B, 256, 4, 4)

        # 3. Passa attraverso i blocchi del decoder
        attention_maps = []
        for block in self.blocks:
            # L'attenzione per-step usa l'output completo dell'encoder
            x, attn_weights = block(x, encoder_output_full if block.use_attention else None)
            if attn_weights is not None:
                attention_maps.append(attn_weights)

        # 4. Layer finale
        x = self.final_conv(x) # -> (B, 3, 256, 256)
        x = self.final_activation(x) # Mappa i valori in [-1, 1]

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

        # Nuovo meccanismo di attenzione per calcolare il contesto iniziale
        # come media pesata (weighted average) degli output dell'encoder.
        self.initial_context_attention = nn.Sequential(
            nn.Linear(in_features=text_embed_dim, out_features=128),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=1),
            nn.Softmax(dim=1)
        )

        self.image_decoder = ImageDecoder(
            noise_dim=noise_dim,
            text_embed_dim=text_embed_dim
        )

        self.noise_dim = noise_dim

    def forward(self, token_ids, return_attentions=False):
        # Genera rumore casuale per il batch
        batch_size = token_ids.size(0)
        noise = torch.randn(batch_size, self.noise_dim, device=token_ids.device)

        # 1. Codifica il testo per ottenere i vettori di ogni parola
        encoder_output = self.text_encoder(token_ids)

        # 2. Calcola il vettore di contesto iniziale con una media pesata (ATTENZIONE #1)
        # 2.1 Calcola i pesi (scores) per ogni parola nella sequenza
        attention_weights = self.initial_context_attention(encoder_output)

        # 2.2 Calcola il contesto come somma pesata degli output dell'encoder
        # Shapes: (B, SeqLen, 1) * (B, SeqLen, EmbDim) -> sum(dim=1) -> (B, EmbDim)
        context_vector = torch.sum(attention_weights * encoder_output, dim=1)

        # 3. Genera l'immagine usando il contesto iniziale e l'output completo dell'encoder
        #    per l'attenzione interna del decoder (ATTENZIONE #2)
        generated_image, attention_maps = self.image_decoder(noise, context_vector, encoder_output)

        if return_attentions:
            return generated_image, attention_maps
        return generated_image
