import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from transformers import CLIPModel, CLIPProcessor

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as transforms # <-- Importiamo torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as transforms # <-- Importiamo torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as transforms

class CLIPLoss(nn.Module):
    """
    CLIP Loss using Hugging Face transformers, designed to be fully differentiable for GAN training.
    This version uses the most robust method for gradient propagation.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = None,
    ):
        super().__init__()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Congeliamo il modello CLIP
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Pipeline di trasformazione differenziabile per le immagini
        vision_config = self.model.config.vision_config
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (vision_config.image_size, vision_config.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True
            ),
            transforms.Normalize(
                mean=self.processor.image_processor.image_mean,
                std=self.processor.image_processor.image_std
            ),
        ])

    def forward(
        self,
        generated_images: torch.Tensor,
        text_descriptions: List[str]
    ) -> torch.Tensor:
        # 1. Preprocessing delle immagini (differenziabile)
        if generated_images.min() < 0:
            generated_images = (generated_images + 1.0) / 2.0
        processed_images = self.image_transform(generated_images)

        # 2. Preprocessing del testo (non differenziabile, solo tokenizzazione)
        with torch.no_grad():
            text_inputs = self.processor(
                text=text_descriptions,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

        # --- LA MODIFICA CRUCIALE E FINALE ---
        # Chiamiamo il metodo `forward` principale del modello invece delle funzioni helper.
        # Questo è il modo più diretto e garantisce che il grafo sia preservato.
        outputs = self.model(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            pixel_values=processed_images,
            return_loss=False # Calcoleremo la loss manualmente dai logits
        )

        # Estraiamo i logits (similarità scalate) direttamente dall'output
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # 4. Calcolo della loss contrastiva (simmetrica)
        batch_size = generated_images.shape[0]
        labels = torch.arange(batch_size, device=self.device)

        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)

        total_loss = (loss_img + loss_txt) / 2.0

        return total_loss


if __name__ == "__main__":
    print("--- Esecuzione test per la classe SimpleCLIPLoss ---")

    # 1. Setup del test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo di test: {device}")

        # 2. Istanziazione della classe di loss
    clip_loss_fn = CLIPLoss(device=device)
    print("✅ Istanza di SimpleCLIPLoss creata con successo.")

    # 3. Creazione di dati di input fittizi (simulando un GAN)
    batch_size = 4
    # Usiamo una dimensione diversa da 224x224 per testare il resize
    image_size = 128

    # Immagini finte in range [-1, 1] con requires_grad=True per testare la backprop
    fake_images = torch.tanh(torch.randn(
        batch_size, 3, image_size, image_size, requires_grad=True
    )).to(device)

    fake_texts = [
        "una foto di un gatto rosso che dorme",
        "un disegno di una casa blu sotto la pioggia",
        "un cane verde che gioca con una palla",
        "un paesaggio montano al tramonto"
    ]
    print(f"Creati dati fittizi: immagini {fake_images.shape}, {len(fake_texts)} testi.")

    # 4. Esecuzione del forward pass
    try:
        loss = clip_loss_fn(fake_images, fake_texts)
        print("✅ Forward pass completato con successo.")
    except Exception as e:
        print(f"❌ Errore durante il forward pass: {e}")
        exit()

    # 5. Verifica dell'output e dei gradienti
    print(f"Loss calcolata: {loss.item():.4f}")

    # Controllo 1: La loss è uno scalare?
    assert loss.ndim == 0, "Errore: la loss non è uno scalare."
    print("✅ La loss è uno scalare.")

    # Controllo 2 (CRITICO): La loss ha un grafo computazionale?
    assert loss.requires_grad, "ERRORE CRITICO: La loss non ha `requires_grad=True`. I gradienti non possono fluire."
    print("✅ La loss ha `requires_grad=True`.")

    # Controllo 3 (CRITICO): La backpropagation funziona?
    try:
        loss.backward()
        print("✅ Backward pass completato con successo.")

        # Controlliamo se il tensore delle immagini ha ricevuto un gradiente
        assert fake_images.grad is not None, "ERRORE CRITICO: Nessun gradiente è stato propagato alle immagini di input."
        assert fake_images.grad.shape == fake_images.shape, "Le dimensioni del gradiente non corrispondono a quelle dell'input."
        print("✅ I gradienti sono stati propagati correttamente al tensore delle immagini.")

    except Exception as e:
        print(f"❌ Errore durante il backward pass o il controllo dei gradienti: {e}")
        exit()

    print("\n--- ✅ Tutti i test sono stati superati con successo! ---")
