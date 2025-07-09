import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import glob
# Aggiungo questo per evitare problemi con la GUI di matplotlib su server senza display
import matplotlib
matplotlib.use('Agg')
from torchvision import transforms

from pokemon_dataset import PokemonDataset
from model import PikaPikaGen

# --- Parametri di Training Fissi ---
MODEL_NAME = "prajjwal1/bert-mini"
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
NOISE_DIM = 100
TRAIN_VAL_SPLIT = 0.9
NUM_WORKERS = 0
# Numero di campioni da visualizzare nelle griglie di confronto
NUM_VIZ_SAMPLES = 4
# Seed per la riproducibilità
RANDOM_SEED = 42


# --- Setup del Dispositivo ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directory per l'Output ---
OUTPUT_DIR = "training_output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


def find_latest_checkpoint(checkpoint_dir):
    """Trova l'ultimo checkpoint in una directory basandosi sul numero di epoca nel nome del file."""
    list_of_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth.tar'))
    if not list_of_files:
        return None
    try:
        latest_file = max(list_of_files, key=lambda f: int(os.path.basename(f).split('_')[-1].split('.')[0]))
        return latest_file
    except (ValueError, IndexError):
        print("Attenzione: impossibile determinare l'ultimo checkpoint a causa di nomi file non standard.")
        return None

def save_checkpoint(epoch, model, optimizer, loss, is_best=False):
    """
    Salva un checkpoint del modello, dell'ottimizzatore e della loss.

    Args:
        epoch (int): L'epoca corrente.
        model (nn.Module): Il modello da salvare.
        optimizer (torch.optim.Optimizer): L'ottimizzatore da salvare.
        loss (float): La loss di validazione corrente.
        is_best (bool, optional): Se True, salva il modello anche come "best_model.pth.tar".
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
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

def save_attention_visualization(epoch, model, tokenizer, batch, device, set_name):
    """
    Genera e salva una visualizzazione dell'attenzione con layout orizzontale.

    L'immagine mostra in alto l'immagine generata e in basso una griglia
    orizzontale dove la stessa immagine è sovrapposta con mappe di calore
    semi-trasparenti, ciascuna per uno specifico token.

    Args:
        epoch (int): L'epoca corrente.
        model (PikaPikaGen): Il modello generatore.
        tokenizer: Il tokenizer per decodificare i token.
        batch (dict): Un batch (dimensione 1) di dati di validazione/training.
        device (torch.device): Il dispositivo su cui eseguire i calcoli.
        set_name (str): 'train' o 'val', per il titolo e nome del file.
    """
    model.eval()

    with torch.no_grad():
        token_ids = batch['text'].to(device)
        # Se il batch ha più elementi, prendi solo il primo per la visualizzazione
        if token_ids.dim() > 2:
             token_ids = token_ids[0].unsqueeze(0)

        pokemon_id = batch['idx'][0]
        description = batch['description'][0]
        tokens = tokenizer.convert_ids_to_tokens(token_ids.squeeze(0))
        generated_image, attention_maps = model(token_ids, return_attentions=True)

    # Usa l'ultima mappa di attenzione, è la più raffinata
    if not attention_maps:
        print(f"Epoch {epoch}: No attention maps returned from model. Skipping visualization.")
        return
    last_attn_map = attention_maps[-1].squeeze(0)  # Shape: (H*W, seq_len)

    # Filtra i token per la visualizzazione
    display_tokens = []
    pad_token_count = 0
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
        if token == '[PAD]':
            if pad_token_count < 6:
                pad_token_count += 1
            else:
                continue
        display_tokens.append({'token': token, 'index': i})


    num_display_tokens = len(display_tokens)
    if num_display_tokens == 0:
        print(f"Epoch {epoch}: No tokens to display for attention visualization in '{description}'. Skipping.")
        return

    img_tensor_cpu = denormalize_image(generated_image.squeeze(0).cpu()).permute(1, 2, 0)

    # --- Creazione del Plot ---
    # Calcola la griglia per le attenzioni, favorendo un layout orizzontale
    cols = min(num_display_tokens, 5)
    rows = (num_display_tokens + cols - 1) // cols

    fig_height = 6 + 4 * rows  # Altezza base per immagine principale + altezza per ogni riga di attenzioni
    fig_width = max(20, 5 * cols) # Larghezza basata sul numero di colonne
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Layout principale: 2 righe. In alto l'immagine, in basso la griglia delle attenzioni
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, rows], hspace=0.4)
    fig.suptitle(f"Epoch {epoch}: Attention for Pokémon #{pokemon_id} ({set_name.capitalize()})", fontsize=24, y=0.98)

    # --- Riga Superiore: Immagine Generata ---
    ax_main_img = fig.add_subplot(gs_main[0])
    ax_main_img.imshow(img_tensor_cpu)
    ax_main_img.set_title("Generated Image", fontsize=16)
    ax_main_img.text(0.5, -0.05, f"Prompt: #{pokemon_id} - {description}", ha='center', va='top', transform=ax_main_img.transAxes, fontsize=14, wrap=True)
    ax_main_img.axis('off')

    # --- Riga Inferiore: Griglia di Sovrapposizioni di Attenzione ---
    gs_attentions = gs_main[1].subgridspec(rows, cols, wspace=0.1, hspace=0.2)

    for i, token_info in enumerate(display_tokens):
        row_idx = i // cols
        col_idx = i % cols
        ax = fig.add_subplot(gs_attentions[row_idx, col_idx])

        attn_for_token = last_attn_map[:, token_info['index']].cpu()
        map_size = int(np.sqrt(attn_for_token.shape[0]))

        if attn_for_token.shape[0] != map_size * map_size:
            ax.axis('off')
            continue

        heatmap = attn_for_token.reshape(map_size, map_size)

        ax.imshow(heatmap, cmap='jet', interpolation='nearest')
        ax.set_title(f"Focus: '{token_info['token']}'", fontsize=12)
        ax.axis('off')

    # Pulisce gli assi non usati
    for i in range(num_display_tokens, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        fig.add_subplot(gs_attentions[row_idx, col_idx]).axis('off')

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save_path = os.path.join(IMAGE_DIR, f"{epoch:03d}_{set_name}_attention_visualization.png")
    plt.savefig(save_path)
    plt.close(fig)


def save_comparison_grid(epoch, model, batch, set_name, device):
    """
    Genera e salva una griglia di confronto orizzontale (reale vs. generato).

    Args:
        epoch (int): L'epoca corrente.
        model (PikaPikaGen): Il modello generatore.
        batch (dict): Un batch di dati (training o validazione).
        set_name (str): 'train' o 'val', per il titolo e nome del file.
        device (torch.device): Il dispositivo su cui eseguire i calcoli.
    """
    model.eval()
    token_ids = batch['text'].to(device)
    real_images = batch['image']
    pokemon_ids = batch['idx']
    descriptions = batch['description']
    num_images = real_images.size(0)

    with torch.no_grad():
        generated_images = model(token_ids)

    fig, axs = plt.subplots(2, num_images, figsize=(4 * num_images, 8.5))
    fig.suptitle(f"Epoch {epoch} - {set_name.capitalize()} Comparison", fontsize=16, y=0.98)

    for i in range(num_images):
        # Riga 0: Immagini Reali
        ax_real = axs[0, i]
        ax_real.imshow(denormalize_image(real_images[i].cpu()).permute(1, 2, 0))
        ax_real.set_title(f"#{pokemon_ids[i]}: {descriptions[i][:35]}...", fontsize=10)
        ax_real.axis('off')

        # Riga 1: Immagini Generate
        ax_gen = axs[1, i]
        ax_gen.imshow(denormalize_image(generated_images[i].cpu()).permute(1, 2, 0))
        ax_gen.axis('off')

    axs[0, 0].text(-0.1, 0.5, "Real", ha="center", va="center", rotation="vertical", fontsize=14, transform=axs[0, 0].transAxes)
    axs[1, 0].text(-0.1, 0.5, "Generated", ha="center", va="center", rotation="vertical", fontsize=14, transform=axs[1, 0].transAxes)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save_path = os.path.join(IMAGE_DIR, f"{epoch:03d}_{set_name}_comparison.png")
    plt.savefig(save_path)
    plt.close(fig)


def train(continue_from_last_checkpoint: bool = True, epochs_to_run: int = NUM_EPOCHS):
    """
    Funzione principale per l'addestramento e la validazione del modello PikaPikaGen.

    Args:
        continue_from_last_checkpoint (bool): Se True, cerca di riprendere dall'ultimo
                                              checkpoint. Se non lo trova, inizia da zero.
        epochs_to_run (int): Il numero di epoche per cui eseguire l'addestramento.
    """
    print("--- Impostazioni di Training ---")
    print(f"Utilizzo del dispositivo: {DEVICE}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Divisione deterministica del Dataset ---
    full_dataset = PokemonDataset(tokenizer=tokenizer)
    train_size = int(TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Creazione deterministica dei batch per la visualizzazione ---
    # Usiamo un generatore separato per il dataloader per non interferire con lo shuffle del training
    vis_generator = torch.Generator().manual_seed(RANDOM_SEED)
    fixed_train_batch = next(iter(DataLoader(train_dataset, batch_size=NUM_VIZ_SAMPLES, shuffle=True, generator=vis_generator)))
    fixed_val_batch = next(iter(DataLoader(val_dataset, batch_size=NUM_VIZ_SAMPLES, shuffle=False))) # la validazione non ha shuffle

    vis_generator.manual_seed(RANDOM_SEED) # Reset per coerenza
    fixed_train_attention_batch = next(iter(DataLoader(train_dataset, batch_size=1, shuffle=True, generator=vis_generator)))
    fixed_val_attention_batch = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=False)))

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    model = PikaPikaGen(text_encoder_model_name=MODEL_NAME, noise_dim=NOISE_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.L1Loss()

    start_epoch = 1
    best_val_loss = float('inf')

    # --- Logica per continuare il training ---
    if continue_from_last_checkpoint:
        latest_checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
        if latest_checkpoint_path:
            print(f"--- Ripresa del training dal checkpoint: {os.path.basename(latest_checkpoint_path)} ---")
            checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('loss', float('inf'))

            print(f"Checkpoint caricato. Si riparte dall'epoca {start_epoch}.")
        else:
            print("--- Nessun checkpoint trovato. Inizio un nuovo training da zero. ---")

    final_epoch = start_epoch + epochs_to_run - 1
    print(f"--- Inizio Training (da epoca {start_epoch} a {final_epoch}) ---")

    for epoch in range(start_epoch, final_epoch + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{final_epoch} [Training]")
        for batch in pbar:
            token_ids, real_images = batch['text'].to(DEVICE), batch['image'].to(DEVICE)

            optimizer.zero_grad()
            generated_images = model(token_ids) # Non chiediamo le attention maps qui
            loss = criterion(generated_images, real_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                token_ids, real_images = batch['text'].to(DEVICE), batch['image'].to(DEVICE)
                generated_images = model(token_ids)
                loss = criterion(generated_images, real_images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{final_epoch} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Salvataggio Checkpoint ---
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        save_checkpoint(epoch, model, optimizer, avg_val_loss, is_best)

        # --- Generazione Visualizzazioni ---
        print(f"Epoch {epoch}: Generazione visualizzazioni...")
        save_attention_visualization(epoch, model, tokenizer, fixed_train_attention_batch, DEVICE, 'train')
        save_attention_visualization(epoch, model, tokenizer, fixed_val_attention_batch, DEVICE, 'val')
        save_comparison_grid(epoch, model, fixed_train_batch, 'train', DEVICE)
        save_comparison_grid(epoch, model, fixed_val_batch, 'val', DEVICE)

    print("Training completato!")


if __name__ == "__main__":
    # --- IMPOSTA QUI LA MODALITÀ DI TRAINING ---
    # continue_training=True -> Cerca l'ultimo checkpoint e riprende. Se non lo trova, parte da zero.
    # continue_training=False -> Parte sempre da zero.
    # epochs -> Numero di epoche per cui addestrare in questa sessione.
    train(
        continue_from_last_checkpoint=True,
        epochs_to_run=100
    )
