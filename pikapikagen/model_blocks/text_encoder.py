import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    Text encoder
    Uses bert-mini embeddings and passes them through a Transformer.
    """
    def __init__(self, model_name="prajjwal1/bert-mini", fine_tune_embeddings=True):
        super().__init__()
        # Load the pre-trained bert-mini model for embeddings
        bert_mini_model = AutoModel.from_pretrained(model_name)

        self.embedding = bert_mini_model.embeddings

        # Set whether to fine-tune the embeddings during training
        for param in self.embedding.parameters():
            param.requires_grad = fine_tune_embeddings

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, token_ids, attention_mask=None):
        # Get the embeddings from the tokens
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded_text = self.embedding(token_ids)

        # Prepare the padding mask for TransformerEncoder
        # The HuggingFace mask is 1 for real tokens, 0 for padding.
        # TransformerEncoder expects True for positions to ignore (padding).
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        # Pass the embeddings through the Transformer Encoder with the mask
        # Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        encoder_output = self.transformer_encoder(
            src=embedded_text,
            src_key_padding_mask=src_key_padding_mask
        )
        return encoder_output
