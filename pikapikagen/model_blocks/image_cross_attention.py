import torch.nn as nn

class ImageCrossAttention(nn.Module):
    """
    Image cross-attention module
    Allows a sequence of queries (from the image) to "pay attention"
    to a sequence of key/value (from the text), internally managing
    the reshaping of tensors and the attention mask.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, image_features, text_features, key_padding_mask=None):
        # query: (B, C, H, W) - Image features
        # key/value: (B, seq_len, embed_dim) - Text encoder output
        # key_padding_mask: (B, seq_len) - Attention mask from the tokenizer

        B, C, H, W = image_features.shape

        # Reshape from image to sequence: (B, C, H, W) -> (B, H*W, C)
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
