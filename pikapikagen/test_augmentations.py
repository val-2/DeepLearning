import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os

from pokemon_dataset import PokemonDataset
from training import GeometricAugmentPipe
from utils import denormalize_image


def visualize_augmentations():
    """
    Carica un'immagine campione e visualizza l'effetto di AugmentPipe
    applicato più volte, salvando il risultato in una griglia.
    """
    print("Inizio visualizzazione delle augmentations...")

    # --- Setup ---
    MODEL_NAME = "prajjwal1/bert-mini"
    NUM_EXAMPLES = 8  # Numero di esempi aumentati da mostrare

    # --- Carica un'immagine campione ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        dataset = PokemonDataset(tokenizer=tokenizer)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        sample_batch = next(iter(loader))
        original_image = sample_batch["image"]
        original_desc = sample_batch["description"][0]
        print(f"Immagine campione caricata per: '{original_desc}'")
    except (FileNotFoundError, IndexError) as e:
        print(
            "Errore: Impossibile caricare il dataset. Assicurati che i file siano nella root del progetto."
        )
        print(f"Dettagli errore: {e}")
        return

    # --- Definisci la pipeline di augmentation ---
    # con p=1.0 per essere sicuri che venga sempre applicata
    augment_pipe = GeometricAugmentPipe(p=1.0)

    # --- Crea la griglia di visualizzazione ---
    num_total_images = NUM_EXAMPLES + 1  # +1 per l'immagine originale
    cols = 3
    rows = (num_total_images + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axs = axs.flatten()

    # 1. Mostra l'immagine originale
    axs[0].imshow(denormalize_image(original_image[0]).permute(1, 2, 0))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # 2. Mostra N esempi di AugmentPipe applicata
    for i in range(1, num_total_images):
        ax = axs[i]
        try:
            # Applica la trasformazione. Cloniamo l'input per sicurezza.
            augmented_image = augment_pipe(original_image.clone())
            ax.imshow(denormalize_image(augmented_image[0]).permute(1, 2, 0))
            ax.set_title(f"AugmentPipe Example #{i}", fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", fontsize=8)
            print(f"Errore durante l'applicazione di AugmentPipe: {e}")
        finally:
            ax.axis("off")

    # Pulisce gli assi non usati
    for j in range(num_total_images, len(axs)):
        axs[j].axis("off")

    fig.suptitle(
        f"Demonstration of AugmentPipe ({NUM_EXAMPLES} stochastic examples)\nPrompt: '{original_desc[:80]}...'",
        fontsize=16,
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    # Salva la figura nella stessa directory dello script
    save_path = os.path.join(
        os.path.dirname(__file__), "augmentation_test_grid.png"
    )
    plt.savefig(save_path)
    print(f"Griglia delle augmentations salvata in: {save_path}")

    # plt.show() # Decommenta per mostrare il plot in un ambiente interattivo


if __name__ == "__main__":
    visualize_augmentations()
