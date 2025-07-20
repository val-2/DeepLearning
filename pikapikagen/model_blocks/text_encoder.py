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

    def forward(self, token_ids, attention_mask=None):
        # 1. Ottieni gli embedding dai token ID
        # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded_text = self.embedding(token_ids)

        # 2. Prepara la maschera di padding per il TransformerEncoder
        # La maschera di HuggingFace è 1 per i token reali, 0 per il padding.
        # TransformerEncoder si aspetta True per le posizioni da ignorare (padding).
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        # 3. Passa gli embedding attraverso il Transformer Encoder con la maschera
        # Shape: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, embedding_dim)
        encoder_output = self.transformer_encoder(
            src=embedded_text,
            src_key_padding_mask=src_key_padding_mask
        )
        return encoder_output
