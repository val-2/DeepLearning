import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T

from pokemon_dataset import PokemonDataset
from model import PikaPikaGen
from discriminator import PikaPikaDisc
from losses import VGGPerceptualLoss, SSIMLoss, SobelLoss
from clip_loss import create_clip_loss
from utils import (
    save_plot_losses,
    load_latest_checkpoint,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    save_attention_visualization,
    save_comparison_grid,
    save_checkpoint,
)

# --- Parametri di Training Fissi ---
MODEL_NAME = "prajjwal1/bert-mini"
BATCH_SIZE = 32
LEARNING_RATE_G = 1e-4  # Più lento per il generatore (TTUR)
LEARNING_RATE_D = 2e-4  # Più veloce per il discriminatore (TTUR)
NOISE_DIM = 100
TRAIN_VAL_SPLIT = 0.9
NUM_WORKERS = 0
# Numero di campioni da visualizzare nelle griglie di confronto
NUM_VIZ_SAMPLES = 4
# Frequenza di salvataggio checkpoint e visualizzazioni
CHECKPOINT_EVERY_N_EPOCHS = 5
# Seed per la riproducibilità
RANDOM_SEED = 42
# Pesi per le diverse loss ausiliarie del generatore
LAMBDA_L1 = 0.1  # RE-INTRODOTTO un valore piccolo come ancora
LAMBDA_PERCEPTUAL = 0.0
LAMBDA_SSIM = 0.0
LAMBDA_SOBEL = 0.0
LAMBDA_CLIP = 0.0
LAMBDA_ADV = 1.0
# Parametri per ADA e R1 rimossi


# --- Setup del Dispositivo ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"PyTorch CUDA version: {torch.version.cuda}")  # type: ignore
    print(f"Numero di GPU disponibili: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
elif hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    print(f"PyTorch XPU version: {torch.version.xpu}")  # type: ignore
    print(f"Numero di XPU disponibili: {torch.xpu.device_count()}")
    for i in range(torch.xpu.device_count()):
        print(f"  XPU {i}: {torch.xpu.get_device_name(i)}")
else:
    DEVICE = torch.device("cpu")
print(f"Utilizzo del dispositivo primario: {DEVICE}")

# --- Directory per l'Output ---
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


class AugmentPipe(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        # Probabilità con cui l'intero set di augmentations viene applicato
        self.p = p

        # Sequenza di trasformazioni. Sono stocastiche, quindi cambiano ad ogni chiamata.
        # I valori sono scelti per essere leggeri e non distruggere troppo l'immagine.
        self.transforms = nn.Sequential(
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=1
            ),  # fill=1 per riempire con bianco (immagini in [-1, 1])
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
        )

    def forward(self, images):
        # Applica le trasformazioni solo con una certa probabilità `p`
        # Questo è il controllo principale "on/off" delle augmentations
        if self.p > 0 and torch.rand(1).item() < self.p:
            return self.transforms(images)
        # Altrimenti, ritorna le immagini non modificate
        return images


def train_discriminator(model_G, model_D, optimizer_D, batch, device, augment_pipe):
    """Esegue un passo di training per il discriminatore."""
    optimizer_D.zero_grad()

    token_ids = batch["text"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    real_images = batch["image"].to(device)

    # Il testo viene processato una sola volta con il text_encoder del generatore.
    # Non calcoliamo i gradienti rispetto al generatore in questa fase.
    with torch.no_grad():
        model_to_use_G = (
            model_G.module if isinstance(model_G, nn.DataParallel) else model_G
        )
        encoder_output = model_to_use_G.text_encoder(token_ids)
        attention_weights = model_to_use_G.image_decoder.initial_context_attention(
            encoder_output
        )
        context_vector = torch.sum(attention_weights * encoder_output, dim=1)

    # --- Predizioni su immagini reali ---
    real_images_aug = augment_pipe(real_images)
    d_real_pred = model_D(real_images_aug, context_vector)
    loss_d_real = F.softplus(-d_real_pred).mean()

    # --- Predizioni su immagini generate ---
    with torch.no_grad():
        noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
        fake_images = model_to_use_G.image_decoder(
            noise, encoder_output, attention_mask
        )[0]

    fake_images_aug = augment_pipe(fake_images)
    d_fake_pred = model_D(fake_images_aug, context_vector)
    loss_d_fake = F.softplus(d_fake_pred).mean()

    # Loss totale del discriminatore
    loss_D = loss_d_real + loss_d_fake
    loss_D.backward()
    optimizer_D.step()

    return loss_D.item()


def train_generator(model_G, model_D, optimizer_G, batch, criterions, device):
    """Esegue un passo di training per il generatore."""
    optimizer_G.zero_grad()

    token_ids = batch["text"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    real_images = batch["image"].to(device)

    # Dobbiamo ricalcolare l'output dell'encoder per far fluire i gradienti
    # attraverso il generatore e il text encoder.
    model_to_use_G = model_G.module if isinstance(model_G, nn.DataParallel) else model_G
    encoder_output_g = model_to_use_G.text_encoder(token_ids)
    attention_weights_g = model_to_use_G.image_decoder.initial_context_attention(
        encoder_output_g
    )
    context_vector_g = torch.sum(attention_weights_g * encoder_output_g, dim=1)

    noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
    generated_images, _, _ = model_to_use_G.image_decoder(
        noise, encoder_output_g, attention_mask
    )
    g_pred = model_D(generated_images, context_vector_g)

    # Loss GAN per il generatore
    loss_g_gan = F.softplus(-g_pred).mean()

    # Loss ausiliarie (pixel-wise)
    loss_l1 = (
        criterions["l1"](generated_images, real_images)
        if LAMBDA_L1 > 0
        else torch.tensor(0.0, device=device)
    )
    loss_perceptual = (
        criterions["perceptual"](generated_images, real_images)
        if LAMBDA_PERCEPTUAL > 0
        else torch.tensor(0.0, device=device)
    )
    loss_ssim = (
        criterions["ssim"](generated_images, real_images)
        if LAMBDA_SSIM > 0
        else torch.tensor(0.0, device=device)
    )
    loss_sobel = (
        criterions["sobel"](generated_images, real_images)
        if LAMBDA_SOBEL > 0
        else torch.tensor(0.0, device=device)
    )
    loss_clip = (
        criterions["clip"](generated_images, batch["description"])
        if criterions["clip"] is not None
        else torch.tensor(0.0, device=device)
    )

    # Loss totale del generatore
    loss_G = (
        LAMBDA_ADV * loss_g_gan
        + LAMBDA_L1 * loss_l1
        + LAMBDA_PERCEPTUAL * loss_perceptual
        + LAMBDA_SSIM * loss_ssim
        + LAMBDA_SOBEL * loss_sobel
        + LAMBDA_CLIP * loss_clip
    )
    loss_G.backward()

    # Clipping per la stabilità generale
    torch.nn.utils.clip_grad_norm_(model_G.parameters(), 1.0)

    optimizer_G.step()

    # Dizionario delle loss per il logging
    individual_losses = {"G_total": loss_G.item()}
    if LAMBDA_ADV > 0:
        individual_losses["G_adv"] = loss_g_gan.item()
    if LAMBDA_L1 > 0:
        individual_losses["L1"] = loss_l1.item()
    if LAMBDA_PERCEPTUAL > 0:
        individual_losses["perceptual"] = loss_perceptual.item()
    if LAMBDA_SSIM > 0:
        individual_losses["ssim"] = loss_ssim.item()
    if LAMBDA_SOBEL > 0:
        individual_losses["sobel"] = loss_sobel.item()
    if LAMBDA_CLIP > 0 and loss_clip is not None:
        individual_losses["clip"] = loss_clip.item()

    return individual_losses


def fit(
    continue_from_last_checkpoint: bool = True,
    epochs_to_run: int = 100,
    use_multi_gpu: bool = True,
    show_images_inline: bool = False,
):
    """
    Funzione principale per l'addestramento e la validazione del modello PikaPikaGen.
    """
    print("--- Impostazioni di Training ---")
    print(f"Utilizzo del dispositivo: {DEVICE}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Creazione dei Dataset ---
    # Crea un'istanza per il training e la validazione (senza aumentazioni interne)
    train_full_dataset = PokemonDataset(tokenizer=tokenizer)
    val_full_dataset = PokemonDataset(tokenizer=tokenizer)

    # --- Divisione deterministica degli indici ---
    assert len(train_full_dataset) == len(val_full_dataset)
    dataset_size = len(train_full_dataset)
    train_size = int(TRAIN_VAL_SPLIT * dataset_size)
    val_size = dataset_size - train_size

    train_indices_subset, val_indices_subset = random_split(
        TensorDataset(torch.arange(dataset_size)),  # type: ignore
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_dataset = torch.utils.data.Subset(
        train_full_dataset, train_indices_subset.indices
    )
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- Creazione deterministica dei batch per la visualizzazione ---
    vis_generator = torch.Generator().manual_seed(RANDOM_SEED)
    fixed_train_batch = next(
        iter(
            DataLoader(
                train_dataset,
                batch_size=NUM_VIZ_SAMPLES,
                shuffle=True,
                generator=vis_generator,
            )
        )
    )
    fixed_val_batch = next(
        iter(DataLoader(val_dataset, batch_size=NUM_VIZ_SAMPLES, shuffle=False))
    )  # la validazione non ha shuffle
    vis_generator.manual_seed(RANDOM_SEED)  # Reset per coerenza
    fixed_train_attention_batch = next(
        iter(
            DataLoader(
                train_dataset, batch_size=1, shuffle=True, generator=vis_generator
            )
        )
    )
    fixed_val_attention_batch = next(
        iter(DataLoader(val_dataset, batch_size=1, shuffle=False))
    )

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    # --- Inizializzazione Modelli, Ottimizzatori e Augment Pipe ---
    model_G = PikaPikaGen(text_encoder_model_name=MODEL_NAME, noise_dim=NOISE_DIM).to(
        DEVICE
    )
    model_D = PikaPikaDisc().to(DEVICE)
    # Crea l'istanza della nostra pipeline di augmentation
    # p ridotto per stabilizzare le fasi iniziali del training
    augment_pipe = AugmentPipe(p=0.2).to(DEVICE)

    # --- Supporto Multi-GPU ---
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Utilizzo di {torch.cuda.device_count()} GPU con nn.DataParallel.")
        model_G = nn.DataParallel(model_G)
        model_D = nn.DataParallel(model_D)
    elif use_multi_gpu and torch.cuda.device_count() <= 1:
        print(
            "Multi-GPU richiesto ma solo una o nessuna GPU disponibile. Proseguo con una singola GPU/CPU."
        )

    optimizer_G = optim.Adam(
        model_G.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.99)
    )
    optimizer_D = optim.Adam(
        model_D.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.99)
    )

    # Loss ausiliarie per il generatore
    criterions = {
        "l1": nn.L1Loss().to(DEVICE),
        "perceptual": VGGPerceptualLoss(device=DEVICE).to(DEVICE),
        "ssim": SSIMLoss().to(DEVICE),
        "sobel": SobelLoss().to(DEVICE),
        "clip": create_clip_loss(device=str(DEVICE)) if LAMBDA_CLIP > 0 else None,
    }

    start_epoch = 1
    best_val_loss = float("inf")

    # Liste per salvare le loss di ogni epoca
    losses_G_hist = []
    losses_D_hist = []

    # --- Logica per continuare il training ---
    if continue_from_last_checkpoint:
        checkpoint = load_latest_checkpoint(CHECKPOINT_DIR, DEVICE)
        if checkpoint:
            try:
                model_to_load_G = (
                    model_G.module if isinstance(model_G, nn.DataParallel) else model_G
                )
                model_to_load_G.load_state_dict(checkpoint["model_G_state_dict"])
                optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])

                model_to_load_D = (
                    model_D.module if isinstance(model_D, nn.DataParallel) else model_D
                )
                model_to_load_D.load_state_dict(checkpoint["model_D_state_dict"])
                optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

                start_epoch = checkpoint["epoch"] + 1
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                losses_G_hist = checkpoint.get("losses_G_hist", [])
                losses_D_hist = checkpoint.get("losses_D_hist", [])
                print(f"Checkpoint caricato. Si riparte dall'epoca {start_epoch}.")
            except Exception as e:
                print(f"Errore nell'applicare i dati del checkpoint: {e}")
                print("Inizio un nuovo training da zero.")
                start_epoch = 1
                best_val_loss = float("inf")
        else:
            print(
                "--- Nessun checkpoint valido trovato. Inizio un nuovo training da zero. ---"
            )
    else:
        print(
            "--- Inizio un nuovo training da zero (opzione 'continue_from_last_checkpoint' disabilitata). ---"
        )

    final_epoch = start_epoch + epochs_to_run - 1
    print(f"--- Inizio Training (da epoca {start_epoch} a {final_epoch}) ---")

    for epoch in range(start_epoch, final_epoch + 1):
        model_G.train()
        model_D.train()

        epoch_losses_g = []
        epoch_losses_d = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{final_epoch} [Training]")
        for i, batch in enumerate(pbar):
            # --- Fase 1: Training del Discriminatore ---
            loss_D = train_discriminator(
                model_G, model_D, optimizer_D, batch, DEVICE, augment_pipe
            )

            # --- Fase 2: Training del Generatore ---
            g_losses = train_generator(
                model_G, model_D, optimizer_G, batch, criterions, DEVICE
            )

            epoch_losses_d.append(loss_D)
            epoch_losses_g.append(g_losses["G_total"])

            postfix_dict = {"D_loss": f"{loss_D:.3f}"}
            postfix_dict.update({k: f"{v:.3f}" for k, v in g_losses.items()})
            pbar.set_postfix(postfix_dict)

        # Salva la media delle loss per l'epoca corrente
        losses_D_hist.append(np.mean(epoch_losses_d))
        losses_G_hist.append(np.mean(epoch_losses_g))

        # --- Fine Epoch: Validazione e Visualizzazione ---
        model_G.eval()
        model_D.eval()

        val_losses = {}
        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch["text"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                real_images = batch["image"].to(DEVICE)

                # Replica della logica di generazione per ottenere tutti gli output necessari
                model_to_use_G = (
                    model_G.module if isinstance(model_G, nn.DataParallel) else model_G
                )
                encoder_output_g = model_to_use_G.text_encoder(token_ids)
                attention_weights_g = (
                    model_to_use_G.image_decoder.initial_context_attention(
                        encoder_output_g
                    )
                )
                context_vector_g = torch.sum(
                    attention_weights_g * encoder_output_g, dim=1
                )
                noise = torch.randn(real_images.size(0), NOISE_DIM, device=DEVICE)
                generated_images, _, _ = model_to_use_G.image_decoder(
                    noise, encoder_output_g, attention_mask
                )

                # Calcolo di tutte le loss del generatore
                g_pred = model_D(generated_images, context_vector_g)
                loss_g_gan = F.softplus(-g_pred).mean()
                loss_l1 = criterions["l1"](generated_images, real_images)
                loss_perceptual = criterions["perceptual"](
                    generated_images, real_images
                )
                loss_ssim = criterions["ssim"](generated_images, real_images)
                loss_sobel = criterions["sobel"](generated_images, real_images)
                loss_clip = (
                    criterions["clip"](generated_images, batch["description"])
                    if criterions["clip"] is not None
                    else torch.tensor(0.0, device=DEVICE)
                )
                loss_G = (
                    LAMBDA_ADV * loss_g_gan
                    + LAMBDA_L1 * loss_l1
                    + LAMBDA_PERCEPTUAL * loss_perceptual
                    + LAMBDA_SSIM * loss_ssim
                    + LAMBDA_SOBEL * loss_sobel
                    + LAMBDA_CLIP * loss_clip
                )

                # Accumulo delle loss per il logging
                batch_losses = {"G_total": loss_G.item()}
                if LAMBDA_ADV > 0:
                    batch_losses["G_adv"] = loss_g_gan.item()
                if LAMBDA_L1 > 0:
                    batch_losses["L1"] = loss_l1.item()
                if LAMBDA_PERCEPTUAL > 0:
                    batch_losses["perceptual"] = loss_perceptual.item()
                if LAMBDA_SSIM > 0:
                    batch_losses["ssim"] = loss_ssim.item()
                if LAMBDA_SOBEL > 0:
                    batch_losses["sobel"] = loss_sobel.item()
                if LAMBDA_CLIP > 0 and loss_clip is not None:
                    batch_losses["clip"] = loss_clip.item()

                for k, v in batch_losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v

        # Calcolo della media delle loss di validazione
        avg_val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
        avg_val_g_loss = avg_val_losses.get("G_total", float("inf"))

        val_loss_str = ", ".join(
            [f"Val_{k}: {v:.4f}" for k, v in avg_val_losses.items() if v > 0]
        )
        print(f"Epoch {epoch}/{final_epoch} -> {val_loss_str}")

        # --- Generazione Visualizzazioni (ogni epoca) ---
        print(f"Epoch {epoch}: Generazione visualizzazioni...")
        # L'attenzione viene solo salvata su file, non mostrata inline
        save_attention_visualization(
            epoch,
            model_G,
            tokenizer,
            fixed_train_attention_batch,
            DEVICE,
            "train",
            IMAGE_DIR,
            show_inline=False,
        )
        save_attention_visualization(
            epoch,
            model_G,
            tokenizer,
            fixed_val_attention_batch,
            DEVICE,
            "val",
            IMAGE_DIR,
            show_inline=False,
        )

        # Mostra o salva le griglie di confronto
        save_comparison_grid(
            epoch,
            model_G,
            fixed_train_batch,
            "train",
            DEVICE,
            IMAGE_DIR,
            show_inline=show_images_inline,
        )
        save_comparison_grid(
            epoch,
            model_G,
            fixed_val_batch,
            "val",
            DEVICE,
            IMAGE_DIR,
            show_inline=show_images_inline,
        )

        # --- Salvataggio Checkpoint (ogni N epoche) ---
        if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0 or epoch == final_epoch:
            is_best = avg_val_g_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_g_loss
            print(f"Epoch {epoch}: Salvataggio checkpoint...")
            save_checkpoint(
                epoch,
                model_G,
                optimizer_G,
                model_D,
                optimizer_D,
                best_val_loss,
                avg_val_losses,
                losses_G_hist,
                losses_D_hist,
                is_best,
            )

    print("Training completato!")

    # Plotta le loss alla fine del training
    save_plot_losses(losses_G_hist, losses_D_hist)

    return losses_G_hist, losses_D_hist


if __name__ == "__main__":
    fit(
        continue_from_last_checkpoint=True,
        epochs_to_run=100,
        use_multi_gpu=True,
        show_images_inline=True,
    )
