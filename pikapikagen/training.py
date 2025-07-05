import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# Aggiungo questo per evitare problemi con la GUI di matplotlib su server senza display
import matplotlib
matplotlib.use('Agg')

from pokemon_dataset import PokemonDataset
from model import PikaPikaGen, Discriminator

# --- Parametri di Training Fissi ---
MODEL_NAME = "prajjwal1/bert-mini"
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
NOISE_DIM = 100
TRAIN_VAL_SPLIT = 0.9
NUM_WORKERS = 0
# Peso per la L1 loss nella loss totale del generatore
LAMBDA_L1 = 100.0
# Numero di campioni da visualizzare nelle griglie di confronto
NUM_VIZ_SAMPLES = 4


# --- Setup del Dispositivo ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directory per l'Output ---
OUTPUT_DIR = "training_output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


def save_gan_checkpoint(epoch, generator, discriminator, optimizer_g, optimizer_d, loss_g, loss_d, is_best=False):
    """
    Salva un checkpoint del generatore, del discriminatore e dei loro ottimizzatori.
    """
    state = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'loss_g': loss_g,
        'loss_d': loss_d,
    }
    filename = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:03d}.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")
        torch.save(state, best_filename)

def denormalize_image(tensor):
    """
    Denormalizza un tensore immagine dall'intervallo [-1, 1] a [0, 1] per la visualizzazione.
    """
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1)

def save_attention_visualization(epoch, generator, tokenizer, batch, device, set_name):
    """
    Genera e salva una visualizzazione dell'attenzione con layout orizzontale.
    """
    generator.eval()

    with torch.no_grad():
        token_ids = batch['text'].to(device)
        pokemon_id = batch['idx'][0]
        description = batch['description'][0]
        tokens = tokenizer.convert_ids_to_tokens(token_ids.squeeze(0))
        generated_image, _, attention_maps = generator(token_ids)

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


def save_comparison_grid(epoch, generator, batch, set_name, device):
    """
    Genera e salva una griglia di confronto orizzontale (reale vs. generato).
    """
    generator.eval()
    token_ids = batch['text'].to(device)
    real_images = batch['image']
    pokemon_ids = batch['idx']
    descriptions = batch['description']
    num_images = real_images.size(0)

    with torch.no_grad():
        generated_images, _, _ = generator(token_ids)

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


def train():
    """
    Funzione principale per l'addestramento e la validazione del modello PikaPikaGen.
    """
    print("--- Impostazioni di Training ---")
    print(f"Utilizzo del dispositivo: {DEVICE}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_dataset = PokemonDataset(tokenizer=tokenizer)

    train_size = int(TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Prendi batch fissi per visualizzazioni consistenti
    fixed_train_batch = next(iter(DataLoader(train_dataset, batch_size=NUM_VIZ_SAMPLES, shuffle=False)))
    fixed_val_batch = next(iter(DataLoader(val_dataset, batch_size=NUM_VIZ_SAMPLES, shuffle=False)))
    fixed_train_attention_batch = next(iter(DataLoader(train_dataset, batch_size=1, shuffle=False)))
    fixed_val_attention_batch = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=False)))

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    generator = PikaPikaGen(text_encoder_model_name=MODEL_NAME, noise_dim=NOISE_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))

    adversarial_loss = nn.BCEWithLogitsLoss()
    reconstruction_loss = nn.L1Loss()

    best_val_g_loss = float('inf')

    print("--- Inizio Training ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Training]")
        for batch in pbar:
            token_ids, real_images = batch['text'].to(DEVICE), batch['image'].to(DEVICE)

            # --- 1. Addestra il Discriminatore ---
            optimizer_d.zero_grad()

            # Genera immagini finte
            generated_images, encoder_output, _ = generator(token_ids)

            # Loss su immagini reali
            d_real_pred = discriminator(real_images, encoder_output.detach())
            d_real_loss = adversarial_loss(d_real_pred, torch.ones_like(d_real_pred))

            # Loss su immagini finte
            d_fake_pred = discriminator(generated_images.detach(), encoder_output.detach())
            d_fake_loss = adversarial_loss(d_fake_pred, torch.zeros_like(d_fake_pred))

            # Loss totale del discriminatore
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            optimizer_d.step()

            # --- 2. Addestra il Generatore ---
            optimizer_g.zero_grad()

            # La loss avversaria del generatore
            d_fake_pred_for_g = discriminator(generated_images, encoder_output)
            g_adv_loss = adversarial_loss(d_fake_pred_for_g, torch.ones_like(d_fake_pred_for_g))

            # La loss di ricostruzione L1
            g_recon_loss = reconstruction_loss(generated_images, real_images)

            # Loss totale del generatore
            g_loss = g_adv_loss + LAMBDA_L1 * g_recon_loss
            g_loss.backward()
            optimizer_g.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            pbar.set_postfix({'G_loss': g_loss.item(), 'D_loss': d_loss.item()})

        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)

        # --- Validazione (opzionale, calcoliamo solo la loss del generatore) ---
        generator.eval()
        val_g_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                token_ids, real_images = batch['text'].to(DEVICE), batch['image'].to(DEVICE)
                generated_images, _, _ = generator(token_ids)
                g_recon_loss = reconstruction_loss(generated_images, real_images)
                val_g_loss += g_recon_loss.item() # Usiamo L1 per la validazione, è più stabile

        avg_val_g_loss = val_g_loss / len(val_loader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} -> Train G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}, Val G_L1_Loss: {avg_val_g_loss:.4f}")

        # --- Salvataggio Checkpoint ---
        is_best = avg_val_g_loss < best_val_g_loss
        if is_best:
            best_val_g_loss = avg_val_g_loss
        save_gan_checkpoint(epoch, generator, discriminator, optimizer_g, optimizer_d, avg_g_loss, avg_d_loss, is_best)

        # --- Generazione Visualizzazioni ---
        print(f"Epoch {epoch}: Generazione visualizzazioni...")
        save_attention_visualization(epoch, generator, tokenizer, fixed_train_attention_batch, DEVICE, 'train')
        save_attention_visualization(epoch, generator, tokenizer, fixed_val_attention_batch, DEVICE, 'val')
        save_comparison_grid(epoch, generator, fixed_train_batch, 'train', DEVICE)
        save_comparison_grid(epoch, generator, fixed_val_batch, 'val', DEVICE)

    print("Training completato!")


if __name__ == "__main__":
    train()
