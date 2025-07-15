import os
import glob
import re

def clean_checkpoints(n_epochs=20):
    """
    Elimina tutti i checkpoint tranne il migliore (best_model.pth.tar), l'ultimo
    e uno ogni 20 epoche
    """
    checkpoint_dir = "training_output/models"

    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} non esiste.")
        return

    # Trova tutti i checkpoint con pattern checkpoint_epoch_*.pth.tar
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth.tar'))
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth.tar')

    if not checkpoint_files:
        print("Nessun checkpoint trovato.")
        return

    # Trova l'ultimo checkpoint basandosi sul numero di epoca
    try:
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1)))
        print(f"Ultimo checkpoint: {os.path.basename(latest_checkpoint)}")
    except (ValueError, AttributeError):
        print("Impossibile determinare l'ultimo checkpoint.")
        return

    # File da preservare
    files_to_keep = {latest_checkpoint}
    if os.path.exists(best_model_path):
        files_to_keep.add(best_model_path)
        print(f"Best model trovato: {os.path.basename(best_model_path)}")

    # Aggiungi checkpoint ogni 10 epoche
    for checkpoint_file in checkpoint_files:
        match = re.search(r'epoch_(\d+)', checkpoint_file)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num % n_epochs == 0:
                files_to_keep.add(checkpoint_file)
                print(f"Conservato checkpoint ogni {n_epochs} epoche: {os.path.basename(checkpoint_file)}")

    # Elimina tutti gli altri checkpoint
    deleted_count = 0
    for checkpoint_file in checkpoint_files:
        if checkpoint_file not in files_to_keep:
            try:
                os.remove(checkpoint_file)
                print(f"Eliminato: {os.path.basename(checkpoint_file)}")
                deleted_count += 1
            except OSError as e:
                print(f"Errore nell'eliminazione di {checkpoint_file}: {e}")

    print(f"\n✅ Pulizia completata. Eliminati {deleted_count} checkpoint.")
    print(f"File conservati:")
    for kept_file in files_to_keep:
        if os.path.exists(kept_file):
            print(f"  - {os.path.basename(kept_file)}")


if __name__ == "__main__":
    clean_checkpoints(n_epochs=20)
