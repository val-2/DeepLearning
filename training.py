import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --- XLA Imports ---
# Prova a importare le librerie XLA. Se non ci sono, il training verrà eseguito su CPU/GPU.
try:
    import torch_xla.core.xla_model as xm
    IS_XLA_AVAILABLE = True
except ImportError:
    IS_XLA_AVAILABLE = False


from dataset import PokemonDataset
from model import PikaPikaGen

# --- Parametri Fissi ---
MODEL_NAME = "prajjwal1/bert-mini"
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
NOISE_DIM = 100
TRAIN_VAL_SPLIT = 0.9
NUM_WORKERS = 2

# Directory per l'output
OUTPUT_DIR = "training_output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


# --- Funzioni Helper ---

def save_checkpoint(epoch, model, optimizer, loss, is_best=False):
    """Salva un checkpoint del modello, gestendo l'ambiente TPU."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    filename = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth.tar")

    # Usa xm.save per ambienti TPU, che gestisce correttamente il salvataggio dal dispositivo XLA
    save_fn = xm.save if IS_XLA_AVAILABLE else torch.save
    save_fn(state, filename)

    if is_best:
        best_filename = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")
        save_fn(state, best_filename)

def denormalize_image(tensor):
    """Denormalizza un tensore immagine da [-1, 1] a [0, 255]."""
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    return tensor

def visualize_attention(epoch, generated_image, tokens, attention_maps, description):
    """
    Visualizza e salva l'immagine generata e le mappe di attenzione.
    """
    img_tensor = denormalize_image(generated_image.squeeze(0).cpu())
    num_maps = len(attention_maps)
    num_tokens = len(tokens)

    fig, axs = plt.subplots(num_tokens, num_maps + 1, figsize=(10, 2 * num_tokens))
    fig.suptitle(f"Epoch {epoch}: {description[:80]}...", fontsize=16)

    gs = axs[0, 0].get_gridspec()
    for ax in axs[:, 0]:
        ax.remove()
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img_tensor.permute(1, 2, 0))
    ax_img.set_title("Generated Image")
    ax_img.axis('off')

    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            for j in range(num_maps + 1):
                axs[i, j].axis('off')
            continue

        axs[i, 1].set_ylabel(f'"{token}"', rotation=0, size='large', ha='right', va='center')

        for j, attn_map in enumerate(attention_maps):
            attn = attn_map.squeeze(0).cpu().detach()
            token_attn = attn[:, i]

            map_size = int(np.sqrt(token_attn.shape[0]))
            heatmap = token_attn.reshape(map_size, map_size)

            axs[i, j + 1].imshow(heatmap, cmap='viridis')
            if i == 0:
                axs[i, j + 1].set_title(f"Block {j+1}")
            axs[i, j + 1].axis('off')

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_path = os.path.join(IMAGE_DIR, f"attention_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close(fig)


def train():
    """Funzione principale di training, compatibile con CPU, GPU e TPU (single-core)."""
    # --- Fase 0: Setup Ambiente e Device ---
    if IS_XLA_AVAILABLE:
        device = xm.xla_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"--- Impostazioni di Training ---")
    print(f"Utilizzo del dispositivo: {'TPU' if IS_XLA_AVAILABLE else device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # --- Fase 1: Caricamento Dati ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_dataset = PokemonDataset(tokenizer=tokenizer)

    train_size = int(TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    fixed_batch = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=True)))
    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # --- Fase 2: Inizializzazione Modello ---
    model = PikaPikaGen(text_encoder_model_name=MODEL_NAME, noise_dim=NOISE_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    best_val_loss = float('inf')

    # --- Fase 3: Ciclo di Training ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Training]")
        for batch in pbar:
            token_ids, real_images = batch['text'].to(device), batch['image'].to(device)

            optimizer.zero_grad()
            generated_images = model(token_ids)
            loss = criterion(generated_images, real_images)
            loss.backward()

            if IS_XLA_AVAILABLE:
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                token_ids, real_images = batch['text'].to(device), batch['image'].to(device)
                generated_images = model(token_ids)
                loss = criterion(generated_images, real_images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Checkpointing e Visualizzazione ---
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        save_checkpoint(epoch, model, optimizer, avg_val_loss, is_best)

        with torch.no_grad():
            fixed_token_ids = fixed_batch['text'].to(device)
            gen_img, attn_maps = model(fixed_token_ids, return_attentions=True)
            tokens = tokenizer.convert_ids_to_tokens(fixed_token_ids.squeeze(0))
            visualize_attention(epoch, gen_img, tokens, attn_maps, fixed_batch['description'][0])

    print("Training completato!")


if __name__ == "__main__":
    train()
