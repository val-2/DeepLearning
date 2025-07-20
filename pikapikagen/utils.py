import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import io
from PIL import Image

OUTPUT_DIR = "training_output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "models")


def find_sorted_checkpoints(checkpoint_dir):
    """Trova i checkpoint in una directory e li ordina dal più recente al meno recente."""
    list_of_files = glob.glob(
        os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth.tar")
    )
    if not list_of_files:
        return []

    valid_checkpoints = []
    for f in list_of_files:
        try:
            epoch = int(os.path.basename(f).split("_")[-1].split(".")[0])
            valid_checkpoints.append((epoch, f))
        except (ValueError, IndexError):
            print(
                f"Attenzione: impossibile analizzare il numero di epoca dal nome file: {f}"
            )

    # Ordina per epoca in ordine decrescente (dal più recente al più vecchio)
    valid_checkpoints.sort(key=lambda x: x[0], reverse=True)

    return [file_path for epoch, file_path in valid_checkpoints]


def save_plot_losses(losses_g, losses_d, output_dir=OUTPUT_DIR):
    """
    Genera e salva un plot delle loss del generatore e del discriminatore.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses_g, label="Generator Loss", color="blue")
    ax.plot(losses_d, label="Discriminator Loss", color="red")
    ax.set_title("Training Losses")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    save_path = os.path.join(output_dir, "training_losses.png")
    plt.savefig(save_path)
    print(f"Grafico delle loss salvato in: {save_path}")
    plt.close(fig)


def load_latest_checkpoint(checkpoint_dir, device, weights_only=False):
    """
    Carica l'ultimo checkpoint valido da una directory, provando i precedenti in caso di fallimento.
    Restituisce il dizionario del checkpoint o None se nessun checkpoint può essere caricato.
    """
    sorted_checkpoints = find_sorted_checkpoints(checkpoint_dir)
    if not sorted_checkpoints:
        return None

    for checkpoint_path in sorted_checkpoints:
        print(
            f"--- Tentativo di caricamento dal checkpoint: {os.path.basename(checkpoint_path)} ---"
        )
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
            print(f"Checkpoint caricato con successo da {os.path.basename(checkpoint_path)}")
            return checkpoint
        except Exception as e:
            print(
                f"Errore nel caricamento del checkpoint {os.path.basename(checkpoint_path)}: {e}"
            )
            print(
                "File corrotto o incompatibile. Tento con il checkpoint precedente, se disponibile."
            )

    return None



def save_checkpoint(
    epoch,
    model_G,
    optimizer_G,
    model_D,
    optimizer_D,
    best_val_loss,
    current_val_losses,
    losses_G_hist,
    losses_D_hist,
    is_best=False,
):
    """
    Salva un checkpoint del modello, degli ottimizzatori e delle loss.
    """
    state = {
        "epoch": epoch,
        "generator_state_dict": model_G.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "discriminator_state_dict": model_D.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "best_val_loss": best_val_loss,
        "current_val_losses": current_val_losses,
        "losses_G_hist": losses_G_hist,
        "losses_D_hist": losses_D_hist,
    }
    filename = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:03d}.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")
        torch.save(state, best_filename)


def denormalize_image(tensor):
    """
    Denormalizza un tensore immagine dall'intervallo [-1, 1] a [0, 1] per la visualizzazione.

    Args:
        tensor (torch.Tensor): Il tensore dell'immagine, con valori in [-1, 1].

    Returns:
        torch.Tensor: Il tensore denormalizzato con valori in [0, 1].
    """
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1)

