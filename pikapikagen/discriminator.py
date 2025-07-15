import torch
import torch.nn as nn

class PikaPikaDisc(nn.Module):
    def __init__(self, img_channels=3, features_d=64, text_embed_dim=256):
        super(PikaPikaDisc, self).__init__()

        # Parte della rete che processa l'immagine fino a una dimensione intermedia (es. 16x16)
        self.image_path = nn.Sequential(
            # Input: 3x256x256
            self._block(in_channels=img_channels, out_channels=features_d, kernel_size=4, stride=2, padding=1, bn=False), # fdx128x128
            self._block(in_channels=features_d, out_channels=features_d*2, kernel_size=4, stride=2, padding=1, bn=True), # 64x64
            self._block(in_channels=features_d*2, out_channels=features_d*4, kernel_size=4, stride=2, padding=1, bn=True), # 32x32
            self._block(in_channels=features_d*4, out_channels=features_d*8, kernel_size=4, stride=2, padding=1, bn=True), # 16x16
        )

        # Proiezione del vettore di contesto testuale per renderlo compatibile con le feature dell'immagine
        self.text_projection = nn.Sequential(
            nn.Linear(in_features=text_embed_dim, out_features=features_d*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Parte finale della rete che processa le feature combinate
        self.combined_path = nn.Sequential(
            # Input combinato: (features_d*8 + features_d*8) x 16x16
            self._block(in_channels=features_d*16, out_channels=features_d*16, kernel_size=4, stride=2, padding=1, bn=True), # 8x8
            self._block(in_channels=features_d*16, out_channels=features_d*32, kernel_size=4, stride=2, padding=1, bn=True), # 4x4
            # L'output è un punteggio grezzo, non una probabilità.
            nn.Conv2d(in_channels=features_d*32, out_channels=1, kernel_size=4, stride=1, padding=0) # 1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, bn=True):
        if bn:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                # Sostituito BatchNorm2d con GroupNorm, come raccomandato per WGAN-GP.
                # GroupNorm con 1 gruppo è essenzialmente LayerNorm.
                nn.GroupNorm(1, out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x, context_vector):
        # 1. Processa l'immagine per estrarre le feature iniziali
        # -> (B, features_d*8, 16, 16)
        img_features = self.image_path(x)

        # 2. Proietta il contesto testuale e lo espande alle dimensioni spaziali dell'immagine
        # -> (B, features_d*8)
        text_features = self.text_projection(context_vector)
        # -> (B, features_d*8, 1, 1) -> (B, features_d*8, 16, 16)
        text_features_expanded = text_features.unsqueeze(-1).unsqueeze(-1).expand_as(img_features)

        # 3. Concatena le feature dell'immagine e del testo
        # -> (B, features_d*16, 16, 16)
        combined_features = torch.cat([img_features, text_features_expanded], dim=1)

        # 4. Processa le feature combinate per ottenere il punteggio finale
        output = self.combined_path(combined_features)
        return output
