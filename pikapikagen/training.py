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
from discriminator import PatchGANDiscriminator
from losses import VGGPerceptualLoss, SSIMLoss, SobelLoss
from clip_loss import create_clip_loss
from augment import GeometricAugmentPipe
from utils import (
    save_plot_losses,
    find_sorted_checkpoints,  # Sostituito load_latest_checkpoint
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
# Pesi per WGAN-GP e loss ausiliarie (disattivate)
LAMBDA_L1 = 0.2
LAMBDA_PERCEPTUAL = 0.0
LAMBDA_SSIM = 0.0
LAMBDA_SOBEL = 0.0
LAMBDA_CLIP = 0.0
LAMBDA_ADV = 1.0
# Parametri per ADA e R1 rimossi
DISCRIMINATOR_ITERATIONS = 1  # Numero di iterazioni del discriminatore per ogni iterazione del generatore
LAMBDA_MISMATCH = 0.5 # Peso per la loss del testo non corrispondente


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


def train_discriminator(model_G, model_D, optimizer_D, batch, device, augment_pipe, criterion_gan):
    """Esegue un passo di training per il Discriminatore (PatchGAN)."""
    optimizer_D.zero_grad()

    token_ids = batch["text"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    real_images = batch["image"].to(device)
    batch_size = real_images.size(0)

    with torch.no_grad():
        model_to_use_G = model_G.module if isinstance(model_G, nn.DataParallel) else model_G
        encoder_output = model_to_use_G.text_encoder(token_ids)
        attention_weights = model_to_use_G.image_decoder.initial_context_attention(encoder_output)
        context_vector = torch.sum(attention_weights * encoder_output, dim=1)

    # --- 1. Score su immagini reali, testo corretto ---
    real_images_aug = augment_pipe(real_images)
    d_real_pred = model_D(real_images_aug, context_vector)
    real_labels = torch.ones_like(d_real_pred, device=device)
    loss_d_real = criterion_gan(d_real_pred, real_labels)

    # --- 2. Score su immagini reali, testo sbagliato (mismatch) ---
    # Mescola i vettori di contesto per creare coppie non corrispondenti
    wrong_context_vector = torch.roll(context_vector, shifts=1, dims=0)
    d_wrong_pred = model_D(real_images_aug.detach(), wrong_context_vector)
    fake_labels_for_wrong = torch.zeros_like(d_wrong_pred, device=device)
    loss_d_mismatch = criterion_gan(d_wrong_pred, fake_labels_for_wrong)


    # --- 3. Score su immagini generate, testo corretto ---
    with torch.no_grad():
        noise = torch.randn(batch_size, NOISE_DIM, device=device)
        fake_images = model_to_use_G.image_decoder(noise, encoder_output, attention_mask)[0]
        fake_images_aug = augment_pipe(fake_images)

    d_fake_pred = model_D(fake_images_aug, context_vector)
    fake_labels = torch.zeros_like(d_fake_pred, device=device)
    loss_d_fake = criterion_gan(d_fake_pred, fake_labels)

    # --- Calcolo Loss totale del Discriminatore ---
    loss_D = loss_d_real + loss_d_fake + LAMBDA_MISMATCH * loss_d_mismatch
    loss_D.backward()
    optimizer_D.step()

    return {
        "D_loss": loss_D.item(),
        "D_real": loss_d_real.item(),
        "D_fake": loss_d_fake.item(),
        "D_mismatch": loss_d_mismatch.item()
    }


def train_generator(model_G, model_D, optimizer_G, batch, criterions, device):
    """Esegue un passo di training per il generatore con loss GAN standard."""
    optimizer_G.zero_grad()

    token_ids = batch["text"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    real_images = batch["image"].to(device)

    model_to_use_G = model_G.module if isinstance(model_G, nn.DataParallel) else model_G
    encoder_output_g = model_to_use_G.text_encoder(token_ids)
    attention_weights_g = model_to_use_G.image_decoder.initial_context_attention(encoder_output_g)
    context_vector_g = torch.sum(attention_weights_g * encoder_output_g, dim=1)

    noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
    generated_images, _, _ = model_to_use_G.image_decoder(noise, encoder_output_g, attention_mask)

    # Score del discriminatore sulle immagini generate
    d_pred_on_fake = model_D(generated_images, context_vector_g)

    # La loss del generatore vuole che il discriminatore classifichi le immagini generate come reali (1)
    real_labels = torch.ones_like(d_pred_on_fake, device=device)
    loss_g_gan = criterions["gan"](d_pred_on_fake, real_labels)

    # Loss ausiliarie (attualmente disattivate tramite lambda)
    loss_l1 = criterions["l1"](generated_images, real_images) if LAMBDA_L1 > 0 else torch.tensor(0.0, device=device)
    loss_perceptual = criterions["perceptual"](generated_images, real_images) if LAMBDA_PERCEPTUAL > 0 else torch.tensor(0.0, device=device)
    loss_ssim = criterions["ssim"](generated_images, real_images) if LAMBDA_SSIM > 0 else torch.tensor(0.0, device=device)
    loss_sobel = criterions["sobel"](generated_images, real_images) if LAMBDA_SOBEL > 0 else torch.tensor(0.0, device=device)
    loss_clip = criterions["clip"](generated_images, batch["description"]) if criterions["clip"] is not None else torch.tensor(0.0, device=device)

    # Loss totale del generatore
    loss_G = loss_g_gan + LAMBDA_L1 * loss_l1 + LAMBDA_PERCEPTUAL * loss_perceptual + LAMBDA_SSIM * loss_ssim + LAMBDA_SOBEL * loss_sobel + LAMBDA_CLIP * loss_clip
    loss_G.backward()
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
    print("--- Impostazioni di Training con PatchGAN ---")
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

    # --- Inizializzazione dei Modelli, Ottimizzatori e Loss ---
    model_G = PikaPikaGen(
        text_encoder_model_name=MODEL_NAME,
        noise_dim=NOISE_DIM,
        fine_tune_embeddings=True,
    ).to(DEVICE)

    model_D = PatchGANDiscriminator(
        img_channels=3,
        features_d=64,
        n_layers=3,
        text_embed_dim=256,
    ).to(DEVICE)

    # --- Ottimizzatori ---
    optimizer_G = optim.Adam(
        model_G.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        model_D.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999)
    )

    # --- Funzioni di Loss ---
    criterion_gan = nn.BCEWithLogitsLoss()
    criterions = {
        "gan": criterion_gan,
        "l1": nn.L1Loss(),
        "perceptual": VGGPerceptualLoss(device=DEVICE).to(DEVICE),
        "ssim": SSIMLoss(size_average=True),
        "sobel": SobelLoss(),
        "clip": create_clip_loss(device=str(DEVICE)) if LAMBDA_CLIP > 0 else None
    }

    augment_pipe = GeometricAugmentPipe(p=0.5)

    # --- Gestione Multi-GPU ---
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Utilizzo di {torch.cuda.device_count()} GPU per il training.")
        model_G = nn.DataParallel(model_G)
        model_D = nn.DataParallel(model_D)

    # --- Caricamento del Checkpoint ---
    start_epoch = 0
    train_losses_history = []
    val_losses_history = []
    best_val_loss = float("inf")

    if continue_from_last_checkpoint:
        sorted_checkpoints = find_sorted_checkpoints(CHECKPOINT_DIR)
        if sorted_checkpoints:
            latest_checkpoint_path = sorted_checkpoints[-1]
            print(f"Caricamento del checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, weights_only=False)

            # Caricamento degli stati per i modelli e gli ottimizzatori
            model_to_load_G = model_G.module if isinstance(model_G, nn.DataParallel) else model_G
            model_to_load_G.load_state_dict(checkpoint["generator_state_dict"])

            model_to_load_D = model_D.module if isinstance(model_D, nn.DataParallel) else model_D
            model_to_load_D.load_state_dict(checkpoint["discriminator_state_dict"])

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
    else:
        print("Inizio del training da zero (opzione 'continue' disabilitata).")


    # --- Loop di Training e Validazione ---
    for epoch in range(start_epoch, start_epoch + epochs_to_run):
        print(f"\n--- Epoca {epoch}/{start_epoch + epochs_to_run - 1} ---")

        # --- Fase di Training ---
        model_G.train()
        model_D.train()
        train_epoch_losses = {"G_total": [], "D_loss": [], "D_real": [], "D_fake": [], "D_mismatch": []}

        progress_bar = tqdm(
            train_loader, desc=f"Training Epoca {epoch}", leave=False, unit="batch"
        )
        for batch_idx, batch in enumerate(progress_bar):

            # --- Training del Discriminatore ---
            d_losses = train_discriminator(model_G, model_D, optimizer_D, batch, DEVICE, augment_pipe, criterion_gan)
            for key, value in d_losses.items():
                if key not in train_epoch_losses:
                    train_epoch_losses[key] = []
                train_epoch_losses[key].append(value)

            # --- Training del Generatore ---
            # Con il GAN standard, il generatore viene addestrato ad ogni passo.
            if (batch_idx + 1) % DISCRIMINATOR_ITERATIONS == 0:
                g_losses = train_generator(
                    model_G, model_D, optimizer_G, batch, criterions, DEVICE
                )

                # Aggiorna le medie delle loss del generatore
                for key, value in g_losses.items():
                    if key not in train_epoch_losses:
                        train_epoch_losses[key] = []
                    train_epoch_losses[key].append(value)

                # Aggiorna la barra di progresso con le loss medie
                mean_d_loss = np.mean(train_epoch_losses["D_loss"])
                mean_g_loss = np.mean(train_epoch_losses["G_total"])
                progress_bar.set_postfix(
                    {"Loss_D": f"{mean_d_loss:.4f}", "Loss_G": f"{mean_g_loss:.4f}"}
                )

        # Calcolo delle loss medie per l'epoca di training
        avg_train_losses = {
            key: np.mean(values) for key, values in train_epoch_losses.items()
        }
        train_losses_history.append(avg_train_losses)
        avg_d_real = avg_train_losses.get('D_real', 0)
        avg_d_fake = avg_train_losses.get('D_fake', 0)
        avg_d_mismatch = avg_train_losses.get('D_mismatch', 0)
        print(
            f"Fine Training Epoca {epoch} - "
            f"Loss G: {avg_train_losses.get('G_total', 0):.4f}, "
            f"Loss D: {avg_train_losses.get('D_loss', 0):.4f} "
            f"(R: {avg_d_real:.4f}, F: {avg_d_fake:.4f}, M: {avg_d_mismatch:.4f})"
        )

        # --- Fase di Validazione ---
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
                model_to_use_G = model_G.module if isinstance(model_G, nn.DataParallel) else model_G
                encoder_output_g = model_to_use_G.text_encoder(token_ids)
                attention_weights_g = model_to_use_G.image_decoder.initial_context_attention(encoder_output_g)
                context_vector_g = torch.sum(attention_weights_g * encoder_output_g, dim=1)

                noise = torch.randn(real_images.size(0), NOISE_DIM, device=DEVICE)
                generated_images, _, _ = model_to_use_G.image_decoder(noise, encoder_output_g, attention_mask)

                d_pred_val = model_D(generated_images, context_vector_g)
                real_labels = torch.ones_like(d_pred_val, device=DEVICE)
                loss_g_gan_val = criterions["gan"](d_pred_val, real_labels)

                # Aggiungi altre loss di validazione se necessario (es. L1)
                loss_G_val = loss_g_gan_val

                val_epoch_losses["G_total_val"].append(loss_G_val.item())
                val_epoch_losses["G_adv_val"].append(loss_g_gan_val.item())

                mean_g_loss_val = np.mean(val_epoch_losses["G_total_val"])
                progress_bar_val.set_postfix({"Loss_G_val": f"{mean_g_loss_val:.4f}"})

        avg_val_losses = {
            key: np.mean(values) for key, values in val_epoch_losses.items()
        }
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
            # Assicurati che il modello sia in CPU e non wrappato per il salvataggio
            model_to_save_G = model_G.module if isinstance(model_G, nn.DataParallel) else model_G
            model_to_save_D = model_D.module if isinstance(model_D, nn.DataParallel) else model_D

            avg_val_g_loss = avg_val_losses.get('G_total_val', float('inf'))
            is_best = bool(avg_val_g_loss < best_val_loss)
            if is_best:
                best_val_loss = avg_val_g_loss

            save_checkpoint(
                epoch=epoch,
                model_G=model_to_save_G,
                optimizer_G=optimizer_G,
                model_D=model_to_save_D,
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
        use_multi_gpu=True,
        show_images_inline=True,
    )
