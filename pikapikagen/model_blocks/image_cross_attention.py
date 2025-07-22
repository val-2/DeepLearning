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

    def forward(self, image_features, text_features, attention_mask=None):
        # query: (B, C, H, W) - Image features
        # key/value: (B, seq_len, embed_dim) - Text encoder output
        # key_padding_mask: (B, seq_len) - Attention mask from the tokenizer

        B, C, H, W = image_features.shape

        # Reshape from image to sequence: (B, C, H, W) -> (B, H*W, C)
        query_seq = image_features.view(B, C, H * W).permute(0, 2, 1)
        query_norm = self.layer_norm(query_seq)

        # Prepare the padding mask from the attention mask
        # The HuggingFace mask is 1 for real tokens, 0 for padding.
        # MultiheadAttention expects True for positions to ignore.
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        attn_output, attn_weights = self.attention(
            query=query_norm,
            key=text_features,
            value=text_features,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        # attn_output: (B, H*W, C)

        # Convert output back into its original size
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        attn_output_spatial = attn_output.permute(0, 2, 1).view(B, C, H, W)

        return attn_output_spatial, attn_weights
