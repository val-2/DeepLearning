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


class ImageCrossAttention(nn.Module):
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

    def forward(self, image_features, text_features, key_padding_mask=None):
        # query: (B, C, H, W) - Feature dell'immagine (spaziale)
        # key/value: (B, seq_len, embed_dim) - Output dell'encoder di testo
        # key_padding_mask: (B, seq_len) - Maschera dal tokenizer

        B, C, H, W = image_features.shape

        # 1. Prepara la query (feature dell'immagine)
        # Reshape da spaziale a sequenza: (B, C, H, W) -> (B, H*W, C)
        query_seq = image_features.view(B, C, H * W).permute(0, 2, 1)
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
            key=text_features,
            value=text_features,
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
    Blocco del Generatore come da istruzioni:
    Attenzione (opzionale) -> Fusione -> Upsampling (ConvTranspose) -> Normalizzazione -> Attivazione.
    """
    def __init__(self, in_channels, out_channels, use_attention=True, text_embed_dim=256, nhead=4):
        super().__init__()
        self.use_attention = use_attention

        if self.use_attention:
            if in_channels != text_embed_dim:
                raise ValueError("in_channels must be equal to text_embedding_dim for attention.")
            self.cross_attention = ImageCrossAttention(embed_dim=in_channels, num_heads=nhead)
            # Nuova convolution per fondere le feature dell'immagine con il contesto del testo
            self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)

        # Blocco di upsampling come da istruzioni
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, out_channels), # Equivalente a LayerNorm per feature map (N, C, H, W)
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, encoder_output=None, attention_mask=None):
        attn_weights = None
        if self.use_attention:
            if encoder_output is None or attention_mask is None:
                raise ValueError("encoder_output and attention_mask must be provided for attention.")

            attn_output, attn_weights = self.cross_attention(
                image_features=x,
                text_features=encoder_output,
                key_padding_mask=attention_mask
            )

            # Concatena le feature originali (x) con il contesto (attn_output)
            # e le fonde con una convoluzione 1x1.
            fused_features = torch.cat([x, attn_output], dim=1) # Shape: (B, 2*C, H, W)
            x = self.fusion_conv(fused_features) # Shape: (B, C, H, W)

        # Apply the U-Net style sequence
        x = self.upsample_block(x)
        return x, attn_weights


class ImageDecoder(nn.Module):
    """
    Decoder CNN (Generatore) che sintetizza l'immagine.
    Questa versione usa l'attenzione per-step fin dall'inizio.
    """
    def __init__(self, noise_dim, text_embed_dim, final_image_channels=3):
        super().__init__()

        # Meccanismo per calcolare i punteggi di attenzione per il contesto iniziale.
        self.initial_context_scorer = nn.Sequential(
            nn.Linear(in_features=text_embed_dim, out_features=512),
            nn.Tanh(),
            nn.Linear(in_features=512, out_features=1)
            # Il Softmax viene applicato nel forward pass per poter usare la maschera
        )

        # Proiezione lineare iniziale a una feature map 4x4.
        self.initial_projection = nn.Sequential(
            nn.Linear(noise_dim + text_embed_dim, 256 * 4 * 4),
            nn.GroupNorm(1, 256 * 4 * 4),
            nn.LeakyReLU(inplace=True)
        )

        # Blocchi del decoder basati su GeneratorBlock
        self.blocks = nn.ModuleList([
            # Input: (B, 256, 4, 4)   -> Output: (B, 256, 8, 8)
            DecoderBlock(in_channels=256, out_channels=256, use_attention=False),
            # Input: (B, 256, 8, 8)   -> Output: (B, 256, 16, 16)
            DecoderBlock(in_channels=256, out_channels=256, use_attention=False),
            # Input: (B, 256, 16, 16)  -> Output: (B, 128, 32, 32)
            DecoderBlock(in_channels=256, out_channels=128, use_attention=False),
            # Input: (B, 128, 32, 32)  -> Output: (B, 64, 64, 64)
            DecoderBlock(in_channels=128, out_channels=64, use_attention=False),
            # Input: (B, 64, 64, 64)  -> Output: (B, 32, 128, 128)
            DecoderBlock(in_channels=64, out_channels=32, use_attention=False),
            # Input: (B, 32, 128, 128) -> Output: (B, 16, 256, 256)
            DecoderBlock(in_channels=32, out_channels=16, use_attention=False),
        ])

        # Layer finale per portare ai canali RGB
        # Input: (B, 16, 256, 256) -> Output: (B, 3, 256, 256)
        self.final_conv = nn.Conv2d(16, final_image_channels, kernel_size=3, padding=1)
        self.final_activation = nn.Tanh()

    def forward(self, noise, encoder_output_full, attention_mask):
        # noise.shape: (B, noise_dim)
        # encoder_output_full.shape: (B, seq_len, text_embed_dim)
        # attention_mask.shape: (B, seq_len)

        # 1. Calcola il vettore di contesto iniziale con una media pesata (ATTENZIONE #1)
        # Calcola i punteggi (logits) per ogni token del testo
        attn_scores = self.initial_context_scorer(encoder_output_full)

        # Applica la maschera di padding prima del softmax.
        # Imposta i punteggi dei token di padding a -infinito.
        if attention_mask is not None:
            # La maschera è (B, seq_len), i punteggi (B, seq_len, 1)
            # Il broadcast si occupa di allineare le dimensioni.
            attn_scores.masked_fill_(attention_mask.unsqueeze(-1) == 0, -1e9)

        # Ora applica il softmax per ottenere i pesi.
        # attention_weights.shape: (B, seq_len, 1)
        attention_weights = torch.softmax(attn_scores, dim=1)

        # Calcola il contesto come media pesata degli output dell'encoder.
        # context_vector.shape: (B, text_embed_dim)
        context_vector = torch.sum(attention_weights * encoder_output_full, dim=1)

        # 2. Prepara il vettore di input iniziale per la proiezione
        #    Si usa direttamente il rumore 'noise' invece del vettore di stile 'w'
        # initial_input.shape: (B, noise_dim + text_embed_dim)
        initial_input = torch.cat([noise, context_vector], dim=1)

        # 3. Proietta e rimodella
        # x.shape: (B, 256 * 4 * 4)
        x = self.initial_projection(initial_input)
        # x.shape: (B, 256, 4, 4)
        x = x.view(x.size(0), 256, 4, 4)

        # 5. Passa attraverso i blocchi del decoder
        attention_maps = []
        for block in self.blocks:
             encoder_ctx = encoder_output_full if block.use_attention else None
             mask_ctx = attention_mask if block.use_attention else None
             # La shape di x viene upsamplata in ogni blocco (es. 4x4 -> 8x8)
             x, attn_weights = block(x, encoder_ctx, mask_ctx)

             if attn_weights is not None:
                # attn_weights.shape: (B, H*W, seq_len)
                attention_maps.append(attn_weights)

        # 6. Layer finale
        # x.shape: (B, 3, 256, 256)
        x = self.final_conv(x)
        # x.shape: (B, 3, 256, 256)
        x = self.final_activation(x)
        return x, attention_maps, attention_weights


class PikaPikaGen(nn.Module):
    """
    Modello completo che unisce Encoder e Decoder.
    """
    def __init__(self, text_encoder_model_name="prajjwal1/bert-mini", noise_dim=100, fine_tune_text_encoder=True):
        super().__init__()
        self.text_encoder = TextEncoder(
            model_name=text_encoder_model_name,
            fine_tune_encoder=fine_tune_text_encoder
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
        # generated_image.shape: (batch_size, 3, 256, 256)
        generated_image, attention_maps, initial_attention_weights = self.image_decoder(noise, encoder_output, attention_mask)

        if return_attentions:
            return generated_image, attention_maps, initial_attention_weights
        return generated_image
