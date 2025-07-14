import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

OUTPUT_DIR = "training_output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "models")


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


def save_plot_losses(losses_g, losses_d, output_dir=OUTPUT_DIR):
    """
    Genera e salva un plot delle loss del generatore e del discriminatore.
    """
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
    plt.close(fig)


def load_latest_checkpoint(checkpoint_dir, device, weights_only=False):
    """
    Carica l'ultimo checkpoint valido da una directory, provando i precedenti in caso di fallimento.
    Restituisce il dizionario del checkpoint o None se nessun checkpoint può essere caricato.
    """
    sorted_checkpoints = find_sorted_checkpoints(checkpoint_dir)
    if not sorted_checkpoints:
        return None

    for checkpoint_path in sorted_checkpoints:
        print(
            f"--- Tentativo di caricamento dal checkpoint: {os.path.basename(checkpoint_path)} ---"
        )
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
            print(f"Checkpoint caricato con successo da {os.path.basename(checkpoint_path)}")
            return checkpoint
        except Exception as e:
            print(
                f"Errore nel caricamento del checkpoint {os.path.basename(checkpoint_path)}: {e}"
            )
            print(
                "File corrotto o incompatibile. Tento con il checkpoint precedente, se disponibile."
            )

    return None



def save_checkpoint(
    epoch,
    model_G,
    optimizer_G,
    model_D,
    optimizer_D,
    best_val_loss,
    current_val_losses,
    losses_G_hist,
    losses_D_hist,
    is_best=False,
):
    """
    Salva un checkpoint del modello, degli ottimizzatori e delle loss.
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
        "best_val_loss": best_val_loss,
        "current_val_losses": current_val_losses,
        "losses_G_hist": losses_G_hist,
        "losses_D_hist": losses_D_hist,
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
    epoch, model, tokenizer, batch, device, set_name, output_dir, show_inline=False
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
        output_dir, f"{epoch:03d}_{set_name}_attention_visualization.png"
    )
    plt.savefig(save_path, bbox_inches="tight")
    if show_inline:
        plt.show()
    plt.close(fig)


def save_comparison_grid(
    epoch, model, batch, set_name, device, output_dir, show_inline=False
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
