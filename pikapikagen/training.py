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

# Import separati per chiarezza
from pokemon_dataset import PokemonDataset
from model import PikaPikaGen
from discriminator import PikaPikaDisc
from losses import VGGPerceptualLoss, SSIMLoss, SobelLoss
from clip_loss import CLIPLoss
from augment import GeometricAugmentPipe
from utils import (
    save_plot_losses,
    find_sorted_checkpoints,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    save_attention_visualization,
    save_comparison_grid,
    save_checkpoint,
)

# --- Parametri di Training Fissi ---
MODEL_NAME = "prajjwal1/bert-mini"
BATCH_SIZE = 32
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 2e-4
NOISE_DIM = 100
TRAIN_VAL_SPLIT = 0.9
NUM_WORKERS = 0
NUM_VIZ_SAMPLES = 4 # Numero di campioni da visualizzare nelle griglie di confronto
# Frequenza di salvataggio checkpoint e visualizzazioni
CHECKPOINT_EVERY_N_EPOCHS = 5
# Seed per la riproducibilità
RANDOM_SEED = 42
# Pesi per loss - simplified for basic GAN training
LAMBDA_L1 = 1.0
LAMBDA_PERCEPTUAL = 0.0
LAMBDA_SSIM = 0.0
LAMBDA_SOBEL = 0.0
LAMBDA_CLIP = 0.0
LAMBDA_ADV = 0.0


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


def _setup_directories():
    """Crea le directory di output necessarie."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    print("Directory di output create.")


def _get_dataloaders(tokenizer):
    """Prepara i dataset e i dataloader per training e validazione."""
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
    return train_loader, val_loader, train_dataset, val_dataset


def _get_visualization_batches(train_dataset, val_dataset):
    """Crea batch fissi per la visualizzazione durante il training."""
    vis_generator = torch.Generator().manual_seed(RANDOM_SEED)
    fixed_train_batch = next(
        iter(
            DataLoader(
                torch.utils.data.Subset(
                    train_dataset,
                    torch.randperm(len(train_dataset), generator=vis_generator)[
                        :NUM_VIZ_SAMPLES
                    ].tolist(),
                ),
                batch_size=NUM_VIZ_SAMPLES,
                shuffle=False,
            )
        )
    )
    fixed_val_batch = next(
        iter(
            DataLoader(
                torch.utils.data.Subset(
                    val_dataset,
                    torch.randperm(len(val_dataset), generator=vis_generator)[
                        :NUM_VIZ_SAMPLES
                    ].tolist(),
                ),
                batch_size=NUM_VIZ_SAMPLES,
                shuffle=False,
            )
        )
    )
    return fixed_train_batch, fixed_val_batch


def _initialize_models():
    """Inizializza i modelli Generatore e Discriminatore."""
    model_G = PikaPikaGen(
        text_encoder_model_name=MODEL_NAME,
        noise_dim=NOISE_DIM,
        fine_tune_embeddings=True,
    ).to(DEVICE)

    model_D = PikaPikaDisc(img_channels=3).to(DEVICE)
    return model_G, model_D


def _initialize_optimizers(model_G, model_D):
    """Inizializza gli ottimizzatori Adam per i modelli."""
    optimizer_G = optim.Adam(
        model_G.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        model_D.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999)
    )
    return optimizer_G, optimizer_D


def _initialize_criterions():
    """Inizializza le funzioni di loss solo quando necessarie."""
    criterions = {
        "gan": nn.BCEWithLogitsLoss() if LAMBDA_ADV > 0 else None,
        "l1": nn.L1Loss() if LAMBDA_L1 > 0 else None,
        "perceptual": VGGPerceptualLoss(device=DEVICE).to(DEVICE) if LAMBDA_PERCEPTUAL > 0 else None,
        "ssim": SSIMLoss(size_average=True) if LAMBDA_SSIM > 0 else None,
        "sobel": SobelLoss() if LAMBDA_SOBEL > 0 else None,
        "clip": CLIPLoss(device=str(DEVICE)) if LAMBDA_CLIP > 0 else None
    }
    return criterions


def _load_checkpoint(model_G, model_D, optimizer_G, optimizer_D):
    """Carica il checkpoint più recente se esiste."""
    start_epoch = 1
    train_losses_history = []
    val_losses_history = []
    best_val_loss = float("inf")

    sorted_checkpoints = find_sorted_checkpoints(CHECKPOINT_DIR)
    if sorted_checkpoints:
        latest_checkpoint_path = sorted_checkpoints[-1]
        print(f"Caricamento del checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, weights_only=False)

        # Caricamento degli stati per i modelli e gli ottimizzatori
        model_G.load_state_dict(checkpoint["generator_state_dict"])
        model_D.load_state_dict(checkpoint["discriminator_state_dict"])

        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

        # Caricamento dello stato del training
        start_epoch = checkpoint["epoch"] + 1
        train_losses_history = checkpoint.get("train_losses_history", [])
        val_losses_history = checkpoint.get("val_losses_history", [])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Training ripreso dall'epoca {start_epoch}")
    else:
        print("Nessun checkpoint trovato, inizio del training da zero.")
    return start_epoch, train_losses_history, val_losses_history, best_val_loss


def _train_one_epoch(epoch, model_G, model_D, train_loader, optimizer_D, optimizer_G, criterions, augment_pipe):
    """Esegue un'epoca di training."""
    model_G.train()
    if LAMBDA_ADV > 0:
        model_D.train()

    # Inizializza tutte le loss possibili
    train_epoch_losses = {
        "G_total": [], "G_adv": [], "L1": [], "perceptual": [], "ssim": [], "sobel": [], "clip": [],
        "D_loss": [], "D_real": [], "D_fake": []
    }

    progress_bar = tqdm(
        train_loader, desc=f"Training Epoca {epoch}", leave=False, unit="batch"
    )
    for batch_idx, batch in enumerate(progress_bar):

        # --- Training del Discriminatore (solo se LAMBDA_ADV > 0) ---
        if LAMBDA_ADV > 0 and criterions["gan"] is not None:
            d_losses = train_discriminator(model_G, model_D, optimizer_D, batch, DEVICE, augment_pipe, criterions["gan"])
        else:
            # Se non c'è training adversariale, traccia loss nulle
            d_losses = {
                "D_loss": 0.0,
                "D_real": 0.0,
                "D_fake": 0.0,
            }

        for key, value in d_losses.items():
            train_epoch_losses[key].append(value)

        # --- Training del Generatore (sempre) ---
        g_losses = train_generator(
            model_G, model_D, optimizer_G, batch, criterions, DEVICE
        )

        # Aggiorna le medie delle loss del generatore
        for key, value in g_losses.items():
            train_epoch_losses[key].append(value)

        # Aggiorna la barra di progresso con le loss medie (solo loss attive)
        active_losses = {}
        if LAMBDA_ADV > 0:
            active_losses["Loss_D"] = f"{np.mean(train_epoch_losses['D_loss']):.4f}"
        active_losses["Loss_G"] = f"{np.mean(train_epoch_losses['G_total']):.4f}"
        if LAMBDA_L1 > 0:
            active_losses["L1"] = f"{np.mean(train_epoch_losses['L1']):.4f}"
        if LAMBDA_PERCEPTUAL > 0:
            active_losses["Perc"] = f"{np.mean(train_epoch_losses['perceptual']):.4f}"
        if LAMBDA_SSIM > 0:
            active_losses["SSIM"] = f"{np.mean(train_epoch_losses['ssim']):.4f}"
        if LAMBDA_SOBEL > 0:
            active_losses["Sobel"] = f"{np.mean(train_epoch_losses['sobel']):.4f}"
        if LAMBDA_CLIP > 0:
            active_losses["CLIP"] = f"{np.mean(train_epoch_losses['clip']):.4f}"

        progress_bar.set_postfix(active_losses)

    # Calcolo delle loss medie per l'epoca di training
    avg_train_losses = {
        key: np.mean(values) for key, values in train_epoch_losses.items()
    }
    return avg_train_losses


def _validate_one_epoch(epoch, model_G, model_D, val_loader, criterions):
    """Esegue un'epoca di validazione."""
    model_G.eval()
    val_epoch_losses = {"G_total_val": [], "G_adv_val": []}

    progress_bar_val = tqdm(
        val_loader, desc=f"Validazione Epoca {epoch}", leave=False, unit="batch"
    )
    with torch.no_grad():
        for batch in progress_bar_val:
            token_ids = batch["text"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            real_images = batch["image"].to(DEVICE)

            # Calcolo della loss del generatore sul set di validazione
            encoder_output_g = model_G.text_encoder(token_ids)

            noise = torch.randn(real_images.size(0), NOISE_DIM, device=DEVICE)
            generated_images, _, _ = model_G.image_decoder(noise, encoder_output_g, attention_mask)

            # Loss adversariale solo se attiva
            if LAMBDA_ADV > 0 and criterions["gan"] is not None:
                d_pred_val = model_D(generated_images)
                real_labels = torch.ones_like(d_pred_val, device=DEVICE)
                loss_g_gan_val = criterions["gan"](d_pred_val, real_labels)
            else:
                loss_g_gan_val = torch.tensor(0.0, device=DEVICE)

            # Calcola tutte le loss ausiliarie attive per la validazione
            loss_l1_val = criterions["l1"](generated_images, real_images) if LAMBDA_L1 > 0 and criterions["l1"] is not None else torch.tensor(0.0, device=DEVICE)
            loss_perceptual_val = criterions["perceptual"](generated_images, real_images) if LAMBDA_PERCEPTUAL > 0 and criterions["perceptual"] is not None else torch.tensor(0.0, device=DEVICE)
            loss_ssim_val = criterions["ssim"](generated_images, real_images) if LAMBDA_SSIM > 0 and criterions["ssim"] is not None else torch.tensor(0.0, device=DEVICE)
            loss_sobel_val = criterions["sobel"](generated_images, real_images) if LAMBDA_SOBEL > 0 and criterions["sobel"] is not None else torch.tensor(0.0, device=DEVICE)
            loss_clip_val = criterions["clip"](generated_images, batch["description"]) if LAMBDA_CLIP > 0 and criterions["clip"] is not None else torch.tensor(0.0, device=DEVICE)

            # Loss totale per la validazione (stesse proporzioni del training)
            loss_G_val = loss_g_gan_val + LAMBDA_L1 * loss_l1_val + LAMBDA_PERCEPTUAL * loss_perceptual_val + LAMBDA_SSIM * loss_ssim_val + LAMBDA_SOBEL * loss_sobel_val + LAMBDA_CLIP * loss_clip_val

            val_epoch_losses["G_total_val"].append(loss_G_val.item())
            val_epoch_losses["G_adv_val"].append(loss_g_gan_val.item())

            mean_g_loss_val = np.mean(val_epoch_losses["G_total_val"])
            progress_bar_val.set_postfix({"Loss_G_val": f"{mean_g_loss_val:.4f}"})

    avg_val_losses = {
        key: np.mean(values) for key, values in val_epoch_losses.items()
    }
    return avg_val_losses


def train_discriminator(model_G, model_D, optimizer_D, batch, device, augment_pipe, criterion_gan):
    """Esegue un passo di training per il Discriminatore semplice (solo immagini)."""
    optimizer_D.zero_grad()

    token_ids = batch["text"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    real_images = batch["image"].to(device)
    batch_size = real_images.size(0)

    # --- 1. Score su immagini reali ---
    real_images_aug = augment_pipe(real_images)
    d_real_pred = model_D(real_images_aug)
    real_labels = torch.ones_like(d_real_pred, device=device)
    loss_d_real = criterion_gan(d_real_pred, real_labels)

    # --- 2. Score su immagini generate ---
    with torch.no_grad():
        encoder_output = model_G.text_encoder(token_ids)
        noise = torch.randn(batch_size, NOISE_DIM, device=device)
        fake_images = model_G.image_decoder(noise, encoder_output, attention_mask)[0]
        fake_images_aug = augment_pipe(fake_images)

    d_fake_pred = model_D(fake_images_aug)
    fake_labels = torch.zeros_like(d_fake_pred, device=device)
    loss_d_fake = criterion_gan(d_fake_pred, fake_labels)

    # --- Calcolo Loss totale del Discriminatore (semplificata) ---
    loss_D = loss_d_real + loss_d_fake
    loss_D.backward()
    optimizer_D.step()

    return {
        "D_loss": loss_D.item(),
        "D_real": loss_d_real.item(),
        "D_fake": loss_d_fake.item(),
    }


def train_generator(model_G, model_D, optimizer_G, batch, criterions, device):
    """Esegue un passo di training per il generatore con GAN semplice."""
    optimizer_G.zero_grad()

    token_ids = batch["text"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    real_images = batch["image"].to(device)

    encoder_output_g = model_G.text_encoder(token_ids)

    noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
    generated_images, _, _ = model_G.image_decoder(noise, encoder_output_g, attention_mask)

    # Loss adversariale - calcolata solo se lambda > 0
    if LAMBDA_ADV > 0 and criterions["gan"] is not None:
        d_pred_on_fake = model_D(generated_images)
        real_labels = torch.ones_like(d_pred_on_fake, device=device)
        loss_g_gan = criterions["gan"](d_pred_on_fake, real_labels)
    else:
        loss_g_gan = torch.tensor(0.0, device=device)

    # Loss ausiliarie - calcolate solo se lambda > 0
    loss_l1 = criterions["l1"](generated_images, real_images) if LAMBDA_L1 > 0 and criterions["l1"] is not None else torch.tensor(0.0, device=device)
    loss_perceptual = criterions["perceptual"](generated_images, real_images) if LAMBDA_PERCEPTUAL > 0 and criterions["perceptual"] is not None else torch.tensor(0.0, device=device)
    loss_ssim = criterions["ssim"](generated_images, real_images) if LAMBDA_SSIM > 0 and criterions["ssim"] is not None else torch.tensor(0.0, device=device)
    loss_sobel = criterions["sobel"](generated_images, real_images) if LAMBDA_SOBEL > 0 and criterions["sobel"] is not None else torch.tensor(0.0, device=device)
    loss_clip = criterions["clip"](generated_images, batch["description"]) if LAMBDA_CLIP > 0 and criterions["clip"] is not None else torch.tensor(0.0, device=device)

    # Loss totale del generatore
    loss_G = loss_g_gan + LAMBDA_L1 * loss_l1 + LAMBDA_PERCEPTUAL * loss_perceptual + LAMBDA_SSIM * loss_ssim + LAMBDA_SOBEL * loss_sobel + LAMBDA_CLIP * loss_clip
    loss_G.backward()
    optimizer_G.step()

    # Dizionario delle loss per il logging - sempre tutte le loss, anche se 0
    individual_losses = {
        "G_total": loss_G.item(),
        "G_adv": loss_g_gan.item(),
        "L1": loss_l1.item(),
        "perceptual": loss_perceptual.item(),
        "ssim": loss_ssim.item(),
        "sobel": loss_sobel.item(),
        "clip": loss_clip.item(),
    }

    return individual_losses


def fit(
    continue_from_last_checkpoint: bool = True,
    epochs_to_run: int = 100,
    show_images_inline: bool = False,
):
    """
    Funzione principale per l'addestramento e la validazione del modello PikaPikaGen.
    Training semplificato con discriminatore che accetta solo immagini.
    """
    print("--- Impostazioni di Training Semplificate ---")
    print(f"Utilizzo del dispositivo: {DEVICE}")

    _setup_directories()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader, val_loader, train_dataset, val_dataset = _get_dataloaders(tokenizer)
    fixed_train_batch, fixed_val_batch = _get_visualization_batches(train_dataset, val_dataset)

    model_G, model_D = _initialize_models()
    optimizer_G, optimizer_D = _initialize_optimizers(model_G, model_D)
    criterions = _initialize_criterions()

    augment_pipe = GeometricAugmentPipe(p=0.5)

    start_epoch = 1
    train_losses_history = []
    val_losses_history = []
    best_val_loss = float("inf")

    if continue_from_last_checkpoint:
        start_epoch, train_losses_history, val_losses_history, best_val_loss = _load_checkpoint(
            model_G, model_D, optimizer_G, optimizer_D
        )
    else:
        print("Inizio del training da zero (opzione 'continue' disabilitata).")


    # --- Loop di Training e Validazione ---
    for epoch in range(start_epoch, start_epoch + epochs_to_run):
        print(f"\n--- Epoca {epoch}/{start_epoch + epochs_to_run - 1} ---")

        # --- Fase di Training ---
        avg_train_losses = _train_one_epoch(
            epoch, model_G, model_D, train_loader, optimizer_D, optimizer_G, criterions, augment_pipe
        )
        train_losses_history.append(avg_train_losses)

        # Costruisce il messaggio di fine epoca mostrando solo le loss attive
        epoch_msg = f"Fine Training Epoca {epoch} - Loss G: {avg_train_losses.get('G_total', 0):.4f}"

        if LAMBDA_ADV > 0:
            avg_d_real = avg_train_losses.get('D_real', 0)
            avg_d_fake = avg_train_losses.get('D_fake', 0)
            epoch_msg += f", Loss D: {avg_train_losses.get('D_loss', 0):.4f} (R: {avg_d_real:.4f}, F: {avg_d_fake:.4f})"

        # Aggiungi loss ausiliarie se attive
        aux_losses = []
        if LAMBDA_L1 > 0:
            aux_losses.append(f"L1: {avg_train_losses.get('L1', 0):.4f}")
        if LAMBDA_PERCEPTUAL > 0:
            aux_losses.append(f"Perc: {avg_train_losses.get('perceptual', 0):.4f}")
        if LAMBDA_SSIM > 0:
            aux_losses.append(f"SSIM: {avg_train_losses.get('ssim', 0):.4f}")
        if LAMBDA_SOBEL > 0:
            aux_losses.append(f"Sobel: {avg_train_losses.get('sobel', 0):.4f}")
        if LAMBDA_CLIP > 0:
            aux_losses.append(f"CLIP: {avg_train_losses.get('clip', 0):.4f}")

        if aux_losses:
            epoch_msg += f", {', '.join(aux_losses)}"

        print(epoch_msg)

        # --- Fase di Validazione ---
        avg_val_losses = _validate_one_epoch(epoch, model_G, model_D, val_loader, criterions)
        val_losses_history.append(avg_val_losses)
        print(
            f"Fine Validazione Epoca {epoch} - "
            f"Loss G (Val): {avg_val_losses.get('G_total_val', 0):.4f}"
        )

        save_comparison_grid(
            epoch,
            model_G,
            fixed_train_batch,
            "train",
            DEVICE,
            IMAGE_DIR,
            show_inline=show_images_inline,
        )

        # --- Salvataggio Checkpoint e Visualizzazioni ---
        if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0:
            avg_val_g_loss = avg_val_losses.get('G_total_val', float('inf'))
            is_best = bool(avg_val_g_loss < best_val_loss)
            if is_best:
                best_val_loss = avg_val_g_loss

            save_checkpoint(
                epoch=epoch,
                model_G=model_G,
                optimizer_G=optimizer_G,
                model_D=model_D,
                optimizer_D=optimizer_D,
                best_val_loss=best_val_loss,
                current_val_losses=avg_val_losses,
                losses_G_hist=[d.get('G_total', 0) for d in train_losses_history],
                losses_D_hist=[d.get('D_loss', 0) for d in train_losses_history],
                is_best=is_best,
            )

            # Genera e salva griglie di confronto e visualizzazioni di attenzione
            save_comparison_grid(
                epoch,
                model_G,
                fixed_val_batch,
                "val",
                DEVICE,
                IMAGE_DIR,
                show_inline=show_images_inline,
            )
            # save_attention_visualization(
            #     epoch,
            #     model_G,
            #     tokenizer,
            #     fixed_val_batch,
            #     DEVICE,
            #     "val",
            #     IMAGE_DIR,
            #     show_inline=show_images_inline,
            # )
            # Salva i grafici delle loss
            save_plot_losses(
                [d.get('G_total', 0) for d in train_losses_history],
                [d.get('D_loss', 0) for d in train_losses_history],
                output_dir=OUTPUT_DIR
            )

    print("--- Addestramento completato ---")
    return model_G, model_D, train_losses_history, val_losses_history


if __name__ == "__main__":
    fit(
        continue_from_last_checkpoint=True,
        epochs_to_run=100,
        show_images_inline=True,
    )
