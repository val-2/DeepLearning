import os
import torch
from utils import load_latest_checkpoint, save_plot_losses, CHECKPOINT_DIR, OUTPUT_DIR


def analyze_and_plot():
    """
    Carica l'ultimo checkpoint valido, estrae le cronologie delle loss
    e genera un grafico.
    """
    print("--- Avvio analisi checkpoint ---")

    # Assicurati che la directory di output esista
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Usa la CPU per l'analisi, non è necessaria la GPU
    device = torch.device("cpu")

    # Carica l'ultimo checkpoint valido
    checkpoint = load_latest_checkpoint(CHECKPOINT_DIR, device, weights_only=False)

    if checkpoint:
        print("Checkpoint trovato. Estraggo e plotto le loss...")
        losses_g_hist = checkpoint.get("losses_G_hist", [])
        losses_d_hist = checkpoint.get("losses_D_hist", [])

        if losses_g_hist and losses_d_hist:
            save_plot_losses(losses_g_hist, losses_d_hist, OUTPUT_DIR)
        else:
            print(
                "Le cronologie delle loss non sono presenti o sono vuote nel checkpoint."
            )
    else:
        print("Nessun checkpoint valido trovato. Impossibile generare il grafico.")


if __name__ == "__main__":
    analyze_and_plot()
