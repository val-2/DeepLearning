import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
from utils import denormalize_image
import torch.nn.functional as F


def save_attention_visualization(
    epoch, model, tokenizer, batch, device, set_name, output_dir, show_inline=False
):
    """
    Genera i dati di attenzione dal modello e crea la visualizzazione.

    Questa funzione funge da wrapper, chiamando prima `generate_attention_data`
    e poi `plot_attention_visualization` per un flusso di lavoro completo.
    """
    print(f"Epoch {epoch}: Generating attention visualization for {set_name} set...")

    # 1. Genera i dati necessari
    attention_data = generate_attention_data(model, tokenizer, batch, device)

    # 2. Se i dati sono stati generati con successo, crea il plot
    if attention_data:
        plot_attention_visualization(
            epoch=epoch,
            set_name=set_name,
            output_dir=output_dir,
            show_inline=show_inline,
            **attention_data,
        )
        print(f"Epoch {epoch}: Attention visualization saved for Pokémon #{attention_data['pokemon_id']}.")
    else:
        print(f"Epoch {epoch}: Skipped attention visualization due to missing data.")


def generate_attention_data(model, tokenizer, batch, device):
    """
    Esegue il modello per generare l'immagine e le mappe di attenzione.

    Questa funzione si occupa della logica di inferenza e della preparazione
    dei dati necessari per la visualizzazione.

    Args:
        model (nn.Module): Il modello da eseguire.
        tokenizer: Il tokenizer utilizzato per convertire gli ID in token.
        batch (dict): Un dizionario contenente i dati di input, che deve includere
                      'text', 'attention_mask', 'idx', e 'description'.
        device (torch.device): Il device su cui eseguire il modello (es. 'cuda').

    Returns:
        dict or None: Un dizionario contenente tutti i dati necessari per il plotting,
                      oppure None se le mappe di attenzione non sono disponibili o
                      non ci sono token validi.
                      Il dizionario contiene le seguenti chiavi:
                      - 'generated_image': (torch.Tensor) L'immagine generata dal modello.
                        Shape: (1, C, H, W).
                      - 'decoder_attention_maps': (list[torch.Tensor]) Lista delle mappe di
                        attenzione cross-attention del decoder. Ogni tensore nella lista
                        ha Shape: (1, num_patches, seq_length_totale).
                      - 'initial_context_weights': (torch.Tensor) I pesi di attenzione iniziali.
                        Shape: (1, 1, seq_length_totale).
                      - 'display_tokens': (list[dict]) Lista di dizionari, uno per ogni
                        token valido da visualizzare. Ogni dict ha {'token': str, 'index': int}.
                      - 'description': (str) La descrizione testuale (prompt).
                      - 'pokemon_id': (any) L'identificatore del campione.
    """
    model.eval()

    with torch.no_grad():
        token_ids = batch["text"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Assicura che il batch size sia 1 per la visualizzazione
        if token_ids.dim() > 1:
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
        print("Mappe di attenzione non disponibili. Salto la generazione dei dati.")
        return None

    # Estrai i token validi da visualizzare
    tokens_all = tokenizer.convert_ids_to_tokens(token_ids.squeeze(0))
    display_tokens = []
    for i, token in enumerate(tokens_all):
        if (
            token not in [tokenizer.sep_token, tokenizer.pad_token]
            and attention_mask[0, i] == 1
        ):
            display_tokens.append({"token": token, "index": i})

    if not display_tokens:
        print(f"Nessun token valido da visualizzare per '{description}'. Salto.")
        return None

    return {
        "generated_image": generated_image.cpu(),
        "decoder_attention_maps": [m.cpu() for m in decoder_attention_maps],
        "initial_context_weights": initial_context_weights.cpu(),
        "display_tokens": display_tokens,
        "description": description,
        "pokemon_id": pokemon_id,
    }


def plot_attention_visualization(
    # Argomenti per l'identificazione del plot
    epoch,
    set_name,
    output_dir,
    # Dati generati dal modello (possono essere batch completi)
    generated_images,
    decoder_attention_maps,
    initial_context_weights,
    # Input testuale originale (può essere un batch completo)
    token_ids,
    attention_mask,
    tokenizer,
    # Metadati del batch (per il campione specifico)
    description,
    pokemon_id,
    # Opzioni di controllo
    sample_idx=0,
    show_inline=False,
):
    """
    Genera e salva una visualizzazione dell'attenzione per un SINGOLO campione da un batch.

    Questa funzione è autonoma: accetta i tensori del batch completo e gestisce
    internamente la selezione del campione e la preparazione dei token.

    Args:
        epoch (int): Numero dell'epoca (per titolo/nome file).
        set_name (str): Nome del set (es. 'train', per titolo/nome file).
        output_dir (str): Cartella dove salvare l'immagine.

        generated_images (torch.Tensor): Tensore delle immagini generate.
            Shape attesa: (B, C, H, W), dove B è la dimensione del batch.
        decoder_attention_maps (list[torch.Tensor]): Lista di tensori di attenzione.
            Ogni tensore ha Shape: (B, num_patches, seq_length).
        initial_context_weights (torch.Tensor): Pesi di attenzione iniziali.
            Shape attesa: (B, 1, seq_length).

        token_ids (torch.Tensor): ID dei token di input.
            Shape attesa: (B, seq_length).
        attention_mask (torch.Tensor): Maschera di attenzione per i token.
            Shape attesa: (B, seq_length).
        tokenizer: L'oggetto tokenizer per la conversione id -> token.

        description (str): Il prompt testuale per il campione selezionato.
        pokemon_id (any): L'ID del campione selezionato.

        sample_idx (int, optional): L'indice del campione nel batch da visualizzare.
            Default a 0 (il primo campione).
        show_inline (bool, optional): Se True, mostra il plot. Default a False.
    """
    # --- 1. Estrazione e Preparazione dei Dati per il Campione Selezionato ---

    # Seleziona il campione specifico usando sample_idx e sposta su CPU
    try:
        img_tensor = generated_images[sample_idx].cpu()
        layer_maps = [m[sample_idx].cpu() for m in decoder_attention_maps if m is not None]
        initial_weights = initial_context_weights[sample_idx].cpu()
        token_ids_sample = token_ids[sample_idx].cpu()
        attention_mask_sample = attention_mask[sample_idx].cpu()
    except IndexError:
        print(f"Errore: sample_idx {sample_idx} è fuori dai limiti per un batch di size {generated_images.shape[0]}.")
        return

    # Logica di filtraggio dei token, ora DENTRO la funzione di plotting
    tokens_all = tokenizer.convert_ids_to_tokens(token_ids_sample)
    display_tokens = []
    for i, token in enumerate(tokens_all):
        if (
            token not in [tokenizer.sep_token, tokenizer.pad_token]
            and attention_mask_sample[i] == 1
        ):
            display_tokens.append({"token": token, "index": i})

    if not display_tokens or not layer_maps or initial_weights is None:
        print(f"Epoch {epoch}: Dati di attenzione o token non validi per il campione #{pokemon_id}. Salto.")
        return

    # --- 2. Creazione del Plot (logica di plotting invariata) ---
    img_tensor_cpu = denormalize_image(img_tensor).permute(1, 2, 0)
    num_decoder_layers = len(layer_maps)
    num_tokens = len(display_tokens)
    token_indices_to_display = [t["index"] for t in display_tokens]

    cols = min(num_tokens, 8)
    rows_per_layer = (num_tokens + cols - 1) // cols
    height_ratios = [3, 2] + [2 * rows_per_layer] * num_decoder_layers
    fig_height = sum(height_ratios)
    fig_width = max(20, 2.5 * cols)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs_main = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios, hspace=1.2)
    fig.suptitle(f"Epoch {epoch}: Attention for Pokémon #{pokemon_id} ({set_name.capitalize()})", fontsize=24)

    # ... (il resto del codice di plotting con plt e gs è identico a prima,
    # ma usa le variabili locali come `img_tensor_cpu`, `layer_maps`, `display_tokens`, etc.) ...

    # Immagine e Prompt
    ax_main_img = fig.add_subplot(gs_main[0])
    ax_main_img.imshow(img_tensor_cpu)
    ax_main_img.set_title("Generated Image", fontsize=18)
    ax_main_img.text(0.5, -0.1, f"Prompt: {description}", ha="center", va="top",
                      transform=ax_main_img.transAxes, fontsize=14, wrap=True)
    ax_main_img.axis("off")

    # Bar chart Attenzione Iniziale
    ax_initial_attn = fig.add_subplot(gs_main[1])
    initial_weights_squeezed = initial_weights.squeeze().numpy()
    token_strings = [t["token"] for t in display_tokens]
    relevant_weights = initial_weights_squeezed[[t["index"] for t in display_tokens]]
    ax_initial_attn.bar(np.arange(len(token_strings)), relevant_weights, color="skyblue")
    ax_initial_attn.set_xticks(np.arange(len(token_strings)))
    ax_initial_attn.set_xticklabels(token_strings, rotation=45, ha="right", fontsize=10)
    ax_initial_attn.set_title("Initial Context Attention (Global)", fontsize=16)
    ax_initial_attn.set_ylabel("Weight", fontsize=12)
    ax_initial_attn.grid(axis="y", linestyle="--", alpha=0.7)

    # Griglie di Heatmap per Strato
    for i, layer_attn_map in enumerate(layer_maps):
        # (Il codice della griglia qui è identico alla versione precedente)
        map_size_flat = layer_attn_map.shape[0] # Shape è ora (num_patches, seq_len)
        map_side = int(np.sqrt(map_size_flat))
        layer_title = f"Decoder Cross-Attention Layer {i+1} (Size: {map_side}x{map_side})"

        relevant_attn_maps = layer_attn_map[:, token_indices_to_display]
        vmin, vmax = relevant_attn_maps.min(), relevant_attn_maps.max()

        gs_layer = gs_main[2 + i].subgridspec(rows_per_layer, cols + 1, wspace=0.2, hspace=0.4, width_ratios=[*([1] * cols), 0.1])
        axes_in_layer = [fig.add_subplot(gs_layer[r, c]) for r in range(rows_per_layer) for c in range(cols)]

        if axes_in_layer:
            y_pos = axes_in_layer[0].get_position().y1
            fig.text(0.5, y_pos + 0.01, layer_title, ha="center", va="bottom", fontsize=16, weight="bold")

        im = None
        for j, token_info in enumerate(display_tokens):
            if j >= len(axes_in_layer):
                break
            ax = axes_in_layer[j]
            attn_for_token = layer_attn_map[:, token_info["index"]]
            heatmap = attn_for_token.reshape(map_side, map_side)
            im = ax.imshow(heatmap, cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(f"'{token_info['token']}'", fontsize=12)
            ax.axis("off")

        if im:
            cax = fig.add_subplot(gs_layer[:, -1])
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Attention Weight", rotation=270, labelpad=15, fontsize=12)

        for j in range(num_tokens, len(axes_in_layer)):
            axes_in_layer[j].axis("off")

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_path = os.path.join(output_dir, f"{epoch:03d}_{set_name}_attention_visualization_{pokemon_id}.png")
    plt.savefig(save_path, bbox_inches="tight")

    # Save figure to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)

    # Convert to PIL image
    attention_plot = Image.open(buf)
    plt.close(fig)  # Close the figure to free memory

    if show_inline:
        plt.show()
    plt.close(fig)

    return attention_plot


def save_plot_losses(losses_g, losses_d, output_dir="training_output", show_inline=True):
    """
    Genera e salva un plot delle loss del generatore e del discriminatore.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(losses_g, label="Generator Loss", color="blue")
    ax.plot(losses_d, label="Discriminator Loss", color="red")
    ax.set_title("Training Losses")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    save_path = os.path.join(output_dir, "training_losses.png")
    plt.savefig(save_path)
    print(f"Grafico delle loss salvato in: {save_path}")

    if show_inline:
        plt.show()
    else:
        plt.close(fig)

def save_plot_non_gan_losses(train_losses_history, val_losses_history, output_dir="training_output", show_inline=True, filter_losses=None):
    """
    Generates and saves plots of losses for non-GAN models with multiple loss components.

    Args:
        train_losses_history: List of dicts containing training losses per epoch
                             e.g., [{'l1': 0.5, 'sobel': 0.3, 'ssim': 0.2}, ...]
        val_losses_history: List of dicts containing validation losses per epoch
        output_dir: Directory to save the plot
        show_inline: Whether to display the plot inline
        filter_losses: Optional list of loss names to plot. If None, plots all losses.
                      e.g., ['l1', 'sobel'] to only plot those specific losses
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract all unique loss keys from both training and validation
    all_keys = set()
    for losses_dict in train_losses_history + val_losses_history:
        all_keys.update(losses_dict.keys())

    # Filter out non-numeric keys if any
    loss_keys = [key for key in all_keys if key not in ['epoch']]

    # Apply filter if specified
    if filter_losses is not None:
        loss_keys = [key for key in loss_keys if key in filter_losses]

    loss_keys = sorted(loss_keys)  # Sort for consistent ordering

    if not loss_keys:
        print("No valid loss keys found")
        return

    # Create subplots
    n_losses = len(loss_keys)
    cols = min(3, n_losses)  # Max 3 columns
    rows = (n_losses + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_losses == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Training and Validation Losses", fontsize=16, y=0.98)

    for i, loss_key in enumerate(loss_keys):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]

        # Extract train and validation losses for this key
        train_values = [losses.get(loss_key, 0) for losses in train_losses_history]
        val_values = [losses.get(loss_key, 0) for losses in val_losses_history]

        epochs_train = range(1, len(train_values) + 1)
        epochs_val = range(1, len(val_values) + 1)

        # Plot training and validation curves
        if train_values:
            ax.plot(epochs_train, train_values, label=f"Train {loss_key}", color="blue", linewidth=1.5)
        if val_values:
            ax.plot(epochs_val, val_values, label=f"Val {loss_key}", color="red", linewidth=1.5, linestyle='--')

        ax.set_title(f"{loss_key.capitalize()} Loss", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis to start from 0 for better visualization
        ax.set_ylim(bottom=0)

    # Hide unused subplots
    for i in range(n_losses, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(output_dir, "non_gan_training_losses.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Non-GAN training losses plot saved to: {save_path}")

    if show_inline:
        plt.show()
    else:
        plt.close(fig)


def save_comparison_grid(epoch, model, batch, set_name, device, output_dir="training_output", show_inline=True):
    """
    Genera e salva/mostra una griglia di confronto orizzontale (reale vs. generato).
    Enhanced version from utils.py - automatically handles 256x256 or 64x64 based on set_name
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    token_ids = batch["text"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    real_images = batch["image"]
    pokemon_ids = batch["idx"]
    descriptions = batch["description"]
    num_images = real_images.size(0)

    with torch.no_grad():
        generated_images = model(token_ids, attention_mask)
        # Handle the case where generator returns both 256x256 and 64x64 images
        if isinstance(generated_images, tuple):
            # Check if we want 64x64 or 256x256 based on set_name
            if "64" in set_name:
                generated_images = generated_images[1]  # Use 64x64 output
                # Resize real images to 64x64 for comparison
                real_images = F.interpolate(real_images, size=(64, 64), mode='bilinear', align_corners=False)
            else:
                generated_images = generated_images[0]  # Use 256x256 output

    fig, axs = plt.subplots(2, num_images, figsize=(4 * num_images, 8.5))
    resolution = "64x64" if "64" in set_name else "256x256"
    fig.suptitle(
        f"Epoch {epoch} - {set_name.capitalize()} Comparison ({resolution})", fontsize=16, y=0.98
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
    else:
        plt.close(fig)
