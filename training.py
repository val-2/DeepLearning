import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, distributed
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --- XLA Imports ---
# Prova a importare le librerie XLA. Se non ci sono, il training verrà eseguito su CPU/GPU.
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    IS_XLA_AVAILABLE = True
except ImportError:
    IS_XLA_AVAILABLE = False


from dataset import PokemonDataset
from model import PikaPikaGen

# --- Parametri di Training Fissi ---
MODEL_NAME = "prajjwal1/bert-mini"
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
NOISE_DIM = 100
TRAIN_VAL_SPLIT = 0.9
NUM_WORKERS = 2

# --- Funzioni Helper ---

def save_checkpoint(epoch, model, optimizer, loss, is_best=False, rank=0):
    """Salva un checkpoint del modello, gestendo l'ambiente TPU."""
    if rank != 0:
        return # Salva solo sul processo master

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    output_dir = "training_output"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth.tar")

    # Usa xm.save per ambienti TPU, che gestisce correttamente il salvataggio dal dispositivo XLA
    save_fn = xm.save if IS_XLA_AVAILABLE else torch.save
    save_fn(state, filename)

    if is_best:
        best_filename = os.path.join(checkpoint_dir, "best_model.pth.tar")
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
    output_dir = "training_output"
    image_dir = os.path.join(output_dir, "images")

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
    save_path = os.path.join(image_dir, f"attention_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close(fig)


def train():
    """Funzione principale di training, compatibile con CPU, GPU e TPU."""
    # --- Fase 0: Setup Ambiente e Device ---
    if IS_XLA_AVAILABLE:
        device = xm.xla_device()
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    if rank == 0:
        print(f"--- Impostazioni di Training ---")
        print(f"Dispositivo: {'TPU' if IS_XLA_AVAILABLE else device}")
        print(f"Numero di processi: {world_size}")
        print(f"Batch size per processo: {BATCH_SIZE}")
        print(f"Batch size totale: {BATCH_SIZE * world_size}")
        # Crea le directory di output solo dal processo master
        output_dir = "training_output"
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    # --- Fase 1: Caricamento Dati ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_dataset = PokemonDataset(tokenizer=tokenizer)

    train_size = int(TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Sampler per la distribuzione dei dati su più core
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, shuffle=False, num_workers=NUM_WORKERS)

    if rank == 0:
        # Prendi un batch fisso per la visualizzazione (solo su master)
        fixed_batch = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=True)))
        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # --- Fase 2: Inizializzazione Modello ---
    model = PikaPikaGen(text_encoder_model_name=MODEL_NAME, noise_dim=NOISE_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()

    # Wrapper per i loader su TPU
    if IS_XLA_AVAILABLE:
        train_loader = pl.ParallelLoader(train_loader, [device])
        val_loader = pl.ParallelLoader(val_loader, [device])

    best_val_loss = float('inf')

    # --- Fase 3: Ciclo di Training ---
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        if train_sampler: train_sampler.set_epoch(epoch)

        # Ottieni il loader iterabile specifico per il dispositivo.
        # Per le TPU, questo è necessario. Per CPU/GPU, è il loader stesso.
        device_loader = train_loader.per_device_loader(device) if IS_XLA_AVAILABLE else train_loader

        pbar = tqdm(device_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Training]", disable=(rank != 0))
        for batch in pbar:
            token_ids, real_images = batch['text'].to(device), batch['image'].to(device)

            optimizer.zero_grad()
            generated_images = model(token_ids)
            loss = criterion(generated_images, real_images)
            loss.backward()

            # Usa xm.optimizer_step su TPU
            if IS_XLA_AVAILABLE:
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()

            if rank == 0: pbar.set_postfix({'loss': loss.item()})

        # --- Validation, Checkpointing e Visualizzazione (solo su master) ---
        if rank == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                # Per la validazione usiamo il loader non wrappato
                val_loader_cpu = DataLoader(val_dataset, batch_size=BATCH_SIZE)
                for batch in val_loader_cpu:
                    token_ids, real_images = batch['text'].to(device), batch['image'].to(device)
                    generated_images = model(token_ids)
                    loss = criterion(generated_images, real_images)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader_cpu)
            print(f"Epoch {epoch}/{NUM_EPOCHS} -> Val Loss: {avg_val_loss:.4f}")

            is_best = avg_val_loss < best_val_loss
            if is_best: best_val_loss = avg_val_loss
            save_checkpoint(epoch, model, optimizer, avg_val_loss, is_best, rank)

            with torch.no_grad():
                fixed_token_ids = fixed_batch['text'].to(device)
                gen_img, attn_maps = model(fixed_token_ids, return_attentions=True)
                tokens = tokenizer.convert_ids_to_tokens(fixed_token_ids.squeeze(0))
                visualize_attention(epoch, gen_img, tokens, attn_maps, fixed_batch['description'][0])

    if rank == 0: print("Training completato!")


def _mp_fn(rank):
    """Funzione di entry point per ogni processo XLA."""
    torch.set_default_tensor_type('torch.FloatTensor')
    train()

if __name__ == "__main__":
    if IS_XLA_AVAILABLE:
        xmp.spawn(_mp_fn, nprocs=None, start_method='fork')
    else:
        train()
