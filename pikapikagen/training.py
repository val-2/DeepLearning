import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import glob

from pokemon_dataset import PokemonDataset
from model import PikaPikaGen
from discriminator import PikaPikaDisc
from losses import VGGPerceptualLoss, SSIMLoss, SobelLoss
from clip_loss import create_clip_loss

# --- Parametri di Training Fissi ---
MODEL_NAME = "prajjwal1/bert-mini"
BATCH_SIZE = 8  # Ridotto per il maggior consumo di memoria della GAN
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
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
LAMBDA_L1 = 0.5
LAMBDA_PERCEPTUAL = 0
LAMBDA_SSIM = 0.0
LAMBDA_SOBEL = 1.0
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
OUTPUT_DIR = "training_output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "models")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")


def find_sorted_checkpoints(checkpoint_dir):
    """Trova i checkpoint in una directory e li ordina dal più recente al meno recente."""
    list_of_files = glob.glob(
        os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth.tar")
    )
    if not list_of_files:
        return []

    valid_checkpoints = []
    for f in list_of_files:
        try:
            epoch = int(os.path.basename(f).split("_")[-1].split(".")[0])
            valid_checkpoints.append((epoch, f))
        except (ValueError, IndexError):
            print(
                f"Attenzione: impossibile analizzare il numero di epoca dal nome file: {f}"
            )

    # Ordina per epoca in ordine decrescente (dal più recente al più vecchio)
    valid_checkpoints.sort(key=lambda x: x[0], reverse=True)

    return [file_path for epoch, file_path in valid_checkpoints]


def save_checkpoint(
    epoch, model_G, optimizer_G, model_D, optimizer_D, loss, is_best=False
):
    """
    Salva un checkpoint del modello, dell'ottimizzatore e della loss.
    """
    state = {
        "epoch": epoch,
        "model_G_state_dict": model_G.module.state_dict()
        if isinstance(model_G, nn.DataParallel)
        else model_G.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "model_D_state_dict": model_D.module.state_dict()
        if isinstance(model_D, nn.DataParallel)
        else model_D.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "loss": loss,
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


def save_attention_visualization(
    epoch, model, tokenizer, batch, device, set_name, show_inline=False
):
    """
    Genera e salva una visualizzazione dell'attenzione multi-livello in stile griglia.

    L'immagine mostra:
    1. In alto, l'immagine generata e il prompt.
    2. Sotto, un bar chart dell'attenzione iniziale (contesto globale).
    3. Di seguito, una serie di griglie, una per ogni strato di attenzione del decoder.
       Ciascuna griglia mostra le mappe di calore pure per ogni token rilevante.
    """
    model.eval()

    with torch.no_grad():
        token_ids = batch["text"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if token_ids.dim() > 1:  # Assicura un batch di 1
            token_ids = token_ids[0].unsqueeze(0)
            attention_mask = attention_mask[0].unsqueeze(0)

        pokemon_id = batch["idx"][0]
        description = batch["description"][0]

        model_to_use = model.module if isinstance(model, nn.DataParallel) else model
        generated_image, attention_maps, initial_context_weights = model_to_use(
            token_ids, attention_mask, return_attentions=True
        )

    decoder_attention_maps = [m for m in attention_maps if m is not None]

    if not decoder_attention_maps or initial_context_weights is None:
        print(
            f"Epoch {epoch}: Mappe di attenzione non disponibili. Salto la visualizzazione."
        )
        return

    tokens_all = tokenizer.convert_ids_to_tokens(token_ids.squeeze(0))
    display_tokens = []
    for i, token in enumerate(tokens_all):
        if (
            token not in [tokenizer.sep_token, tokenizer.pad_token]
            and attention_mask[0, i] == 1
        ):
            display_tokens.append({"token": token, "index": i})

    if not display_tokens:
        print(
            f"Epoch {epoch}: Nessun token valido da visualizzare per '{description}'. Salto."
        )
        return

    token_indices_to_display = [t["index"] for t in display_tokens]
    img_tensor_cpu = denormalize_image(generated_image.squeeze(0).cpu()).permute(
        1, 2, 0
    )
    num_decoder_layers = len(decoder_attention_maps)
    num_tokens = len(display_tokens)

    # --- Creazione del Plot ---
    # Calcola dinamicamente layout e dimensioni
    cols = min(num_tokens, 8)
    rows_per_layer = (num_tokens + cols - 1) // cols
    num_main_rows = (
        2 + num_decoder_layers
    )  # Immagine, Bar chart, e N layer di attenzione
    # Altezza per immagine, bar chart, e poi per ogni riga di ogni layer
    height_ratios = [3, 2] + [2 * rows_per_layer] * num_decoder_layers
    fig_height = sum(height_ratios)
    fig_width = max(20, 2.5 * cols)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs_main = fig.add_gridspec(
        num_main_rows, 1, height_ratios=height_ratios, hspace=1.2
    )
    fig.suptitle(
        f"Epoch {epoch}: Attention Visualization for Pokémon #{pokemon_id} ({set_name.capitalize()})",
        fontsize=24,
    )

    # --- 1. Immagine Generata e Prompt ---
    ax_main_img = fig.add_subplot(gs_main[0])
    ax_main_img.imshow(img_tensor_cpu)
    ax_main_img.set_title("Generated Image", fontsize=18)
    ax_main_img.text(
        0.5,
        -0.1,
        f"Prompt: {description}",
        ha="center",
        va="top",
        transform=ax_main_img.transAxes,
        fontsize=14,
        wrap=True,
    )
    ax_main_img.axis("off")

    # --- 2. Attenzione Iniziale per il Contesto (bar chart) ---
    ax_initial_attn = fig.add_subplot(gs_main[1])
    initial_weights_squeezed = initial_context_weights.squeeze().cpu().numpy()
    token_strings = [t["token"] for t in display_tokens]
    token_indices = [t["index"] for t in display_tokens]
    relevant_weights = initial_weights_squeezed[token_indices]
    ax_initial_attn.bar(
        np.arange(len(token_strings)), relevant_weights, color="skyblue"
    )
    ax_initial_attn.set_xticks(np.arange(len(token_strings)))
    ax_initial_attn.set_xticklabels(token_strings, rotation=45, ha="right", fontsize=10)
    ax_initial_attn.set_title("Initial Context Attention (Global)", fontsize=16)
    ax_initial_attn.set_ylabel("Weight", fontsize=12)
    ax_initial_attn.grid(axis="y", linestyle="--", alpha=0.7)

    # --- 3. Attenzione per Strato del Decoder (griglie di heatmap) ---
    for i, layer_attn_map in enumerate(decoder_attention_maps):
        map_size_flat = layer_attn_map.shape[1]
        map_side = int(np.sqrt(map_size_flat))
        layer_title = (
            f"Decoder Cross-Attention Layer {i + 1} (Size: {map_side}x{map_side})"
        )
        layer_attn_map_squeezed = layer_attn_map.squeeze(0).cpu()

        # Seleziona solo le mappe di attenzione per i token che visualizziamo
        relevant_attn_maps = layer_attn_map_squeezed[:, token_indices_to_display]

        # Trova i valori min/max per questo strato per la colorbar
        vmin = relevant_attn_maps.min()
        vmax = relevant_attn_maps.max()

        # Crea una subgrid per questo strato (con una colonna in più per la colorbar)
        gs_layer = gs_main[2 + i].subgridspec(
            rows_per_layer,
            cols + 1,
            wspace=0.2,
            hspace=0.4,
            width_ratios=[*([1] * cols), 0.1],
        )

        # Crea tutti gli assi per la griglia
        axes_in_layer = [
            fig.add_subplot(gs_layer[r, c])
            for r in range(rows_per_layer)
            for c in range(cols)
        ]

        # Usa la posizione del primo asse per il titolo
        if axes_in_layer:
            y_pos = axes_in_layer[0].get_position().y1
            fig.text(
                0.5,
                y_pos + 0.01,
                layer_title,
                ha="center",
                va="bottom",
                fontsize=16,
                weight="bold",
            )

        for j, token_info in enumerate(display_tokens):
            ax = axes_in_layer[j]
            attn_for_token = layer_attn_map_squeezed[:, token_info["index"]]
            heatmap = attn_for_token.reshape(map_side, map_side)
            im = ax.imshow(
                heatmap, cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax
            )
            ax.set_title(f"'{token_info['token']}'", fontsize=12)
            ax.axis("off")

        # Aggiungi la colorbar
        cax = fig.add_subplot(gs_layer[:, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Attention Weight", rotation=270, labelpad=15, fontsize=12)

        # Pulisce gli assi non usati nella griglia
        for j in range(num_tokens, len(axes_in_layer)):
            axes_in_layer[j].axis("off")

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_path = os.path.join(
        IMAGE_DIR, f"{epoch:03d}_{set_name}_attention_visualization.png"
    )
    plt.savefig(save_path, bbox_inches="tight")
    if show_inline:
        plt.show()
    plt.close(fig)


def save_comparison_grid(
    epoch, model, batch, set_name, device, show_inline=False, output_dir=IMAGE_DIR
):
    """
    Genera e salva/mostra una griglia di confronto orizzontale (reale vs. generato).

    Args:
        epoch (int): L'epoca corrente.
        model (PikaPikaGen): Il modello generatore.
        batch (dict): Un batch di dati (training o validazione).
        set_name (str): 'train' o 'val', per il titolo e nome del file.
        device (torch.device): Il dispositivo su cui eseguire i calcoli.
        show_inline (bool): Se True, mostra il plot direttamente (per notebook).
        output_dir (str): La directory dove salvare l'immagine.
    """
    model.eval()
    token_ids = batch["text"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    real_images = batch["image"]
    pokemon_ids = batch["idx"]
    descriptions = batch["description"]
    num_images = real_images.size(0)

    with torch.no_grad():
        generated_images = model(token_ids, attention_mask)

    fig, axs = plt.subplots(2, num_images, figsize=(4 * num_images, 8.5))
    fig.suptitle(
        f"Epoch {epoch} - {set_name.capitalize()} Comparison", fontsize=16, y=0.98
    )

    for i in range(num_images):
        # Riga 0: Immagini Reali
        ax_real = axs[0, i]
        ax_real.imshow(denormalize_image(real_images[i].cpu()).permute(1, 2, 0))
        ax_real.set_title(f"#{pokemon_ids[i]}: {descriptions[i][:35]}...", fontsize=10)
        ax_real.axis("off")

        # Riga 1: Immagini Generate
        ax_gen = axs[1, i]
        ax_gen.imshow(denormalize_image(generated_images[i].cpu()).permute(1, 2, 0))
        ax_gen.axis("off")

    axs[0, 0].text(
        -0.1,
        0.5,
        "Real",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=14,
        transform=axs[0, 0].transAxes,
    )
    axs[1, 0].text(
        -0.1,
        0.5,
        "Generated",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=14,
        transform=axs[1, 0].transAxes,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # Salva sempre l'immagine
    save_path = os.path.join(output_dir, f"{epoch:03d}_{set_name}_comparison.png")
    plt.savefig(save_path)

    if show_inline:
        plt.show()

    plt.close(fig)


def plot_losses(losses_g, losses_d):
    """
    Genera un plot delle loss del generatore e del discriminatore.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses_g, label="Generator Loss", color="blue")
    ax.plot(losses_d, label="Discriminator Loss", color="red")
    ax.set_title("Training Losses")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    plt.show()


def train_discriminator(model_G, model_D, optimizer_D, batch, device):
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
    d_real_pred = model_D(real_images, context_vector)
    loss_d_real = F.softplus(-d_real_pred).mean()

    # --- Predizioni su immagini generate ---
    with torch.no_grad():
        noise = torch.randn(real_images.size(0), NOISE_DIM, device=device)
        fake_images = model_to_use_G.image_decoder(
            noise, encoder_output, attention_mask
        )[0]

    d_fake_pred = model_D(fake_images, context_vector)
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
    # L'AugmentPipe è stata rimossa perché non usiamo più ADA

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
        model_G.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.99)
    )
    optimizer_D = optim.Adam(
        model_D.parameters(), lr=LEARNING_RATE_D, betas=(0.0, 0.99)
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
            checkpoint_path = sorted_checkpoints[0]
            print(
                f"--- Ripresa del training dal checkpoint: {os.path.basename(checkpoint_path)} ---"
            )
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

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
                best_val_loss = checkpoint.get("loss", float("inf"))
                print(f"Checkpoint caricato. Si riparte dall'epoca {start_epoch}.")
                checkpoint_loaded = True
            except Exception as e:
                print(
                    f"Errore nel caricamento del checkpoint {os.path.basename(checkpoint_path)}: {e}"
                )
                print(
                    "Il file potrebbe essere corrotto o incompatibile. Inizio un nuovo training da zero."
                )

        if not checkpoint_loaded:
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
            loss_D = train_discriminator(model_G, model_D, optimizer_D, batch, DEVICE)

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
                token_ids = batch['text'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                real_images = batch['image'].to(DEVICE)

                # Replica della logica di generazione per ottenere tutti gli output necessari
                model_to_use_G = model_G.module if isinstance(model_G, nn.DataParallel) else model_G
                encoder_output_g = model_to_use_G.text_encoder(token_ids)
                attention_weights_g = model_to_use_G.image_decoder.initial_context_attention(encoder_output_g)
                context_vector_g = torch.sum(attention_weights_g * encoder_output_g, dim=1)
                noise = torch.randn(real_images.size(0), NOISE_DIM, device=DEVICE)
                generated_images, _, _ = model_to_use_G.image_decoder(noise, encoder_output_g, attention_mask)

                # Calcolo di tutte le loss del generatore
                g_pred = model_D(generated_images, context_vector_g)
                loss_g_gan = F.softplus(-g_pred).mean()
                loss_l1 = criterions['l1'](generated_images, real_images)
                loss_perceptual = criterions['perceptual'](generated_images, real_images)
                loss_ssim = criterions['ssim'](generated_images, real_images)
                loss_sobel = criterions['sobel'](generated_images, real_images)
                loss_clip = criterions['clip'](generated_images, batch['description']) if criterions['clip'] is not None else torch.tensor(0.0, device=DEVICE)
                loss_G = (LAMBDA_ADV * loss_g_gan + LAMBDA_L1 * loss_l1 + LAMBDA_PERCEPTUAL * loss_perceptual +
                          LAMBDA_SSIM * loss_ssim + LAMBDA_SOBEL * loss_sobel + LAMBDA_CLIP * loss_clip)

                # Accumulo delle loss per il logging
                batch_losses = {'G_total': loss_G.item()}
                if LAMBDA_ADV > 0: batch_losses['G_adv'] = loss_g_gan.item()
                if LAMBDA_L1 > 0: batch_losses['L1'] = loss_l1.item()
                if LAMBDA_PERCEPTUAL > 0: batch_losses['perceptual'] = loss_perceptual.item()
                if LAMBDA_SSIM > 0: batch_losses['ssim'] = loss_ssim.item()
                if LAMBDA_SOBEL > 0: batch_losses['sobel'] = loss_sobel.item()
                if LAMBDA_CLIP > 0 and loss_clip is not None: batch_losses['clip'] = loss_clip.item()

                for k, v in batch_losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v

        # Calcolo della media delle loss di validazione
        avg_val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
        avg_val_g_loss = avg_val_losses.get('G_total', float('inf'))

        val_loss_str = ", ".join([f"Val_{k}: {v:.4f}" for k, v in avg_val_losses.items() if v > 0])
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
            show_inline=False,
        )
        save_attention_visualization(
            epoch,
            model_G,
            tokenizer,
            fixed_val_attention_batch,
            DEVICE,
            "val",
            show_inline=False,
        )

        # Mostra o salva le griglie di confronto
        save_comparison_grid(
            epoch,
            model_G,
            fixed_train_batch,
            "train",
            DEVICE,
            show_inline=show_images_inline,
        )
        save_comparison_grid(
            epoch,
            model_G,
            fixed_val_batch,
            "val",
            DEVICE,
            show_inline=show_images_inline,
        )

        # --- Salvataggio Checkpoint (ogni N epoche) ---
        if epoch % CHECKPOINT_EVERY_N_EPOCHS == 0 or epoch == final_epoch:
            is_best = avg_val_g_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_g_loss
            print(f"Epoch {epoch}: Salvataggio checkpoint...")
            save_checkpoint(epoch, model_G, optimizer_G, model_D, optimizer_D, avg_val_g_loss, is_best)


    print("Training completato!")

    # Plotta le loss alla fine del training
    plot_losses(losses_G_hist, losses_D_hist)

    return losses_G_hist, losses_D_hist


if __name__ == "__main__":
    fit(
        continue_from_last_checkpoint=False,
        epochs_to_run=200,
        use_multi_gpu=True,
        show_images_inline=True,
    )
