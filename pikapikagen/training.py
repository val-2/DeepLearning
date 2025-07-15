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
LEARNING_RATE_C = 2e-4  # Rinominato per chiarezza (Critico)
NOISE_DIM = 100
TRAIN_VAL_SPLIT = 0.9
NUM_WORKERS = 0
NUM_VIZ_SAMPLES = 4 # Numero di campioni da visualizzare nelle griglie di confronto
# Frequenza di salvataggio checkpoint e visualizzazioni
CHECKPOINT_EVERY_N_EPOCHS = 5
# Seed per la riproducibilità
RANDOM_SEED = 42
# Pesi per WGAN-GP e loss ausiliarie (disattivate)
LAMBDA_GP = 10  # Peso per il Gradient Penalty
LAMBDA_L1 = 0.0
LAMBDA_PERCEPTUAL = 0.0
LAMBDA_SSIM = 0.0
LAMBDA_SOBEL = 0.0
LAMBDA_CLIP = 0.0
LAMBDA_ADV = 1.0
# Parametri per ADA e R1 rimossi
CRITIC_ITERATIONS = 5  # Numero di iterazioni del critico per ogni iterazione del generatore


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


def compute_gradient_penalty(critic, real_samples, fake_samples, context_vector, device):
    """Calcola il gradient penalty per WGAN-GP."""
    # Campionamento casuale di un punto sulla linea tra un'immagine reale e una finta
    alpha = torch.randn(real_samples.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Calcolo dello score del critico per i campioni interpolati
    interpolated_scores = critic(interpolated, context_vector)

    # Calcolo dei gradienti dello score rispetto all'input interpolato
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.size(0), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty


def train_critic(model_G, model_C, optimizer_C, batch, device, augment_pipe):
    """Esegue un passo di training per il Critico (ex-Discriminatore)."""
    optimizer_C.zero_grad()

    token_ids = batch["text"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    real_images = batch["image"].to(device)

    with torch.no_grad():
        model_to_use_G = model_G.module if isinstance(model_G, nn.DataParallel) else model_G
        encoder_output = model_to_use_G.text_encoder(token_ids)
        attention_weights = model_to_use_G.image_decoder.initial_context_attention(encoder_output)
        context_vector = torch.sum(attention_weights * encoder_output, dim=1)

    # --- Score su immagini reali ---
    real_images_aug = augment_pipe(real_images)
    c_real_pred = model_C(real_images_aug, context_vector)

    # --- Score su immagini generate ---
    with torch.no_grad():
        noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
        fake_images = model_to_use_G.image_decoder(noise, encoder_output, attention_mask)[0]
        fake_images_aug = augment_pipe(fake_images)

    c_fake_pred = model_C(fake_images_aug, context_vector)

    # --- Calcolo Loss WGAN-GP ---
    # La loss del critico vuole massimizzare (real - fake)
    loss_c_adv = c_fake_pred.mean() - c_real_pred.mean()

    # Calcolo del gradient penalty
    gradient_penalty = compute_gradient_penalty(model_C, real_images, fake_images, context_vector, device)

    # Loss totale del Critico
    loss_C = loss_c_adv + LAMBDA_GP * gradient_penalty
    loss_C.backward()
    optimizer_C.step()

    return {"C_loss": loss_C.item(), "GP": gradient_penalty.item()}


def train_generator(model_G, model_C, optimizer_G, batch, criterions, device):
    """Esegue un passo di training per il generatore con loss WGAN."""
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

    # Score del critico sulle immagini generate
    g_pred = model_C(generated_images, context_vector_g)

    # Loss WGAN per il generatore: vuole massimizzare lo score del critico
    loss_g_gan = -g_pred.mean()

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
    print("--- Impostazioni di Training con WGAN-GP ---")
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
    model_C = PikaPikaDisc().to(DEVICE) # Ora è un Critico
    # Augmentation disabilitata (p=0) per isolare l'effetto di WGAN-GP
    augment_pipe = GeometricAugmentPipe(p=0.0)

    # --- Supporto Multi-GPU ---
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Utilizzo di {torch.cuda.device_count()} GPU con nn.DataParallel.")
        model_G = nn.DataParallel(model_G)
        model_C = nn.DataParallel(model_C)
    elif use_multi_gpu and torch.cuda.device_count() <= 1:
        print(
            "Multi-GPU richiesto ma solo una o nessuna GPU disponibile. Proseguo con una singola GPU/CPU."
        )

    optimizer_G = optim.Adam(
        model_G.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.9)
    )
    optimizer_C = optim.Adam(
        model_C.parameters(), lr=LEARNING_RATE_C, betas=(0.0, 0.9)
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
        sorted_checkpoints = find_sorted_checkpoints(CHECKPOINT_DIR)
        checkpoint_loaded = False
        if sorted_checkpoints:
            for checkpoint_path in sorted_checkpoints:
                try:
                    print(
                        f"--- Tentativo di caricamento dal checkpoint: {os.path.basename(checkpoint_path)} ---"
                    )
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

                    # Carica Generatore
                    model_to_load_G = (
                        model_G.module
                        if isinstance(model_G, nn.DataParallel)
                        else model_G
                    )
                    model_to_load_G.load_state_dict(checkpoint["model_G_state_dict"])
                    optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])

                    # Carica Critico
                    model_to_load_C = (
                        model_C.module
                        if isinstance(model_C, nn.DataParallel)
                        else model_C
                    )
                    model_to_load_C.load_state_dict(checkpoint["model_C_state_dict"])
                    optimizer_C.load_state_dict(checkpoint["optimizer_C_state_dict"])

                    start_epoch = checkpoint["epoch"] + 1
                    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                    losses_G_hist = checkpoint.get("losses_G_hist", [])
                    losses_D_hist = checkpoint.get("losses_D_hist", [])

                    print(
                        f"Checkpoint caricato con successo da {os.path.basename(checkpoint_path)}. Si riparte dall'epoca {start_epoch}."
                    )
                    checkpoint_loaded = True
                    break  # Esci dal ciclo se il caricamento ha successo
                except Exception as e:
                    print(
                        f"Errore nel caricare o applicare il checkpoint {os.path.basename(checkpoint_path)}: {e}"
                    )
                    print(
                        "File corrotto o incompatibile. Tento con il checkpoint precedente, se disponibile."
                    )

        if not checkpoint_loaded:
            print(
                "--- Nessun checkpoint valido trovato o applicabile. Inizio un nuovo training da zero. ---"
            )
    else:
        print(
            "--- Inizio un nuovo training da zero (opzione 'continue_from_last_checkpoint' disabilitata). ---"
        )

    final_epoch = start_epoch + epochs_to_run - 1
    print(f"--- Inizio Training (da epoca {start_epoch} a {final_epoch}) ---")

    for epoch in range(start_epoch, final_epoch + 1):
        model_G.train()
        model_C.train()

        epoch_losses_g = []
        epoch_losses_c = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{final_epoch} [Training]")
        for i, batch in enumerate(pbar):
            # --- Fase 1: Training del Critico (eseguito più volte) ---
            c_losses_list = []
            for _ in range(CRITIC_ITERATIONS):
                c_loss_item = train_critic(
                    model_G, model_C, optimizer_C, batch, DEVICE, augment_pipe
                )
                c_losses_list.append(c_loss_item)

            # --- Fase 2: Training del Generatore (eseguito una volta) ---
            g_losses = train_generator(
                model_G, model_C, optimizer_G, batch, criterions, DEVICE
            )

            # Usa la media delle loss del critico dall'ultimo ciclo per il logging
            avg_c_loss = np.mean([l["C_loss"] for l in c_losses_list])
            avg_gp = np.mean([l["GP"] for l in c_losses_list])

            epoch_losses_c.append(avg_c_loss)
            epoch_losses_g.append(g_losses["G_total"])

            postfix_dict = {
                "C_loss": f"{avg_c_loss:.3f}",
                "GP": f"{avg_gp:.3f}",
                "G_loss": f"{g_losses['G_total']:.3f}",
            }
            pbar.set_postfix(postfix_dict)

        # Salva la media delle loss per l'epoca corrente
        losses_D_hist.append(np.mean(epoch_losses_c)) # Aggiornato per salvare loss del critico
        losses_G_hist.append(np.mean(epoch_losses_g))

        # --- Fine Epoch: Validazione e Visualizzazione ---
        model_G.eval()
        model_C.eval()

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
                g_pred = model_C(generated_images, context_vector_g)
                loss_g_gan = -g_pred.mean()
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
            show_inline=False,
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
                model_C, # AGGIORNATO
                optimizer_C, # AGGIORNATO
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
