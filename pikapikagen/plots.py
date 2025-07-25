import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
from utils import denormalize_image
import torch.nn.functional as F
from transformers import AutoTokenizer


def save_attention_visualization(
    epoch, model, tokenizer, batch, device, set_name, output_dir, show_inline=False
):
    print(f"Epoch {epoch}: Generating attention visualization for {set_name} set...")

    attention_data = generate_attention_data(model, tokenizer, batch, device)

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
    Runs the model to generate the image and attention maps, filtering the padding tokens.
    """
    model.eval()

    with torch.no_grad():
        token_ids = batch["text"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Ensure batch size is 1 for visualization
        if token_ids.dim() > 1:
            token_ids = token_ids[0].unsqueeze(0)
            attention_mask = attention_mask[0].unsqueeze(0)

        # Get the first sample from the batch
        pokemon_id = batch["idx"][0]
        description = batch["description"][0]

        generated_image, attention_maps, initial_context_weights = model(
            token_ids, attention_mask, return_attentions=True
        )

    decoder_attention_maps = [m for m in attention_maps if m is not None]

    if not decoder_attention_maps or initial_context_weights is None:
        print("Attention maps not available. Skipping data generation.")
        return None

    # Extract valid tokens to display
    tokens_all = tokenizer.convert_ids_to_tokens(token_ids.squeeze(0))
    display_tokens = []
    for i, token in enumerate(tokens_all):
        if (
            token not in [tokenizer.sep_token, tokenizer.pad_token]
            and attention_mask[0, i] == 1
        ):
            display_tokens.append({"token": token, "index": i})

    if not display_tokens:
        print(f"No valid tokens to display for '{description}'. Skipping.")
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
    # Plot identification arguments
    epoch: int,
    set_name: str,
    output_dir: str | None,
    # Data generated by the model (can be full batches)
    generated_images: torch.Tensor,
    decoder_attention_maps: list[torch.Tensor],
    initial_context_weights: torch.Tensor,
    # Original text input (can be a full batch)
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: AutoTokenizer,
    # Batch metadata (for the specific sample)
    description: str,
    pokemon_id: int | str,
    # Control options
    sample_idx: int = 0,
    show_inline: bool = False,
):
    """
    Generates and saves an attention visualization for a single sample from a batch.

    This function is self-contained: it accepts full batch tensors and internally
    handles sample selection and token preparation.

    Args:
        epoch (int): Epoch number (for title/filename).
        set_name (str): Set name (e.g., 'train', for title/filename).
        output_dir (str, optional): Folder to save the image. If None, the plot is not saved.

        generated_images (torch.Tensor): Tensor of generated images.
            Shape: (B, C, H, W).
        decoder_attention_maps (list[torch.Tensor]): List of attention tensors.
            Each tensor shape: (B, num_patches, seq_length).
        initial_context_weights (torch.Tensor): Initial attention weights.
            Shape: (B, 1, seq_length).

        token_ids (torch.Tensor): Input token.
            Shape: (B, seq_length).
        attention_mask (torch.Tensor): Attention mask for tokens.
            Shape: (B, seq_length).
        tokenizer: The tokenizer object for id -> token conversion.

        description (str): The text prompt for the selected sample.
        pokemon_id (int or str): The ID of the selected sample.

        sample_idx (int, optional): Index of the sample in the batch to visualize.
            Defaults to 0.
        show_inline (bool, optional): If True, shows the plot. Defaults to False.
    """
    # Select the specific sample using sample_idx and move to CPU
    img_tensor = generated_images[sample_idx].cpu()
    layer_maps = [m[sample_idx].cpu() for m in decoder_attention_maps if m is not None]
    initial_weights = initial_context_weights[sample_idx].cpu()
    token_ids_sample = token_ids[sample_idx].cpu()
    attention_mask_sample = attention_mask[sample_idx].cpu()

    # Token filtering logic
    tokens_all = tokenizer.convert_ids_to_tokens(token_ids_sample)
    display_tokens = []
    for i, token in enumerate(tokens_all):
        if (
            token not in [tokenizer.sep_token, tokenizer.pad_token]
            and attention_mask_sample[i] == 1
        ):
            display_tokens.append({"token": token, "index": i})

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

    ax_main_img = fig.add_subplot(gs_main[0])
    ax_main_img.imshow(img_tensor_cpu)
    ax_main_img.set_title("Generated Image", fontsize=18)
    ax_main_img.text(0.5, -0.1, f"Prompt: {description}", ha="center", va="top",
                      transform=ax_main_img.transAxes, fontsize=14, wrap=True)
    ax_main_img.axis("off")

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

    # Iterate through each decoder layer's attention maps
    for i, layer_attn_map in enumerate(layer_maps):
        # layer_attn_map shape is now (num_patches, seq_len)
        map_size_flat = layer_attn_map.shape[0]
        map_side = int(np.sqrt(map_size_flat))
        layer_title = f"Decoder Cross-Attention Layer {i+1} (Size: {map_side}x{map_side})"

        # Extract attention weights only for tokens we want to display
        relevant_attn_maps = layer_attn_map[:, token_indices_to_display]
        vmin, vmax = relevant_attn_maps.min(), relevant_attn_maps.max()

        # Create subplot grid for this layer
        gs_layer = gs_main[2 + i].subgridspec(rows_per_layer, cols + 1, wspace=0.2, hspace=0.4, width_ratios=[*([1] * cols), 0.1])
        axes_in_layer = [fig.add_subplot(gs_layer[r, c]) for r in range(rows_per_layer) for c in range(cols)]

        # Add layer title above the token attention maps
        if axes_in_layer:
            y_pos = axes_in_layer[0].get_position().y1
            fig.text(0.5, y_pos + 0.01, layer_title, ha="center", va="bottom", fontsize=16, weight="bold")

        # Plot attention heatmap for each token
        im = None
        for j, token_info in enumerate(display_tokens):
            if j >= len(axes_in_layer):
                break
            ax = axes_in_layer[j]
            attn_for_token = layer_attn_map[:, token_info["index"]]
            # Reshape flat attention to spatial grid
            heatmap = attn_for_token.reshape(map_side, map_side)
            im = ax.imshow(heatmap, cmap="jet", interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_title(f"'{token_info['token']}'", fontsize=12)
            ax.axis("off")

        # Add colorbar for the layer
        if im:
            cax = fig.add_subplot(gs_layer[:, -1])
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Attention Weight", rotation=270, labelpad=15, fontsize=12)

        # Hide unused subplots
        for j in range(num_tokens, len(axes_in_layer)):
            axes_in_layer[j].axis("off")

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    if output_dir is not None:
        save_path = os.path.join(output_dir, f"{epoch:03d}_{set_name}_attention_visualization_{pokemon_id}.png")
        plt.savefig(save_path, bbox_inches="tight")

    # Save figure to bytes for potential further use (e.g., logging)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)

    # Convert to PIL image
    attention_plot = Image.open(buf)

    if show_inline:
        plt.show()
    plt.close(fig)

    return attention_plot


def save_plot_losses(losses_g, losses_d, output_dir="training_output", show_inline=True):
    """
    Generates and saves a plot of the generator and discriminator losses.
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
    print(f"Loss plot saved to: {save_path}")

    if show_inline:
        plt.show()
    else:
        plt.close(fig)

def save_plot_non_gan_losses(train_losses_history, val_losses_history, output_dir="training_output", show_inline=True, filter_losses=None):
    """
    Generates and saves plots of losses for non-GAN models with multiple loss components.

    Args:
        train_losses_history (list[dict]): List of dicts containing training losses per epoch.
                                           e.g., [{'l1': 0.5, 'sobel': 0.3}, ...]
        val_losses_history (list[dict]): List of dicts containing validation losses per epoch.
        output_dir (str): Directory to save the plot.
        show_inline (bool): Whether to display the plot inline.
        filter_losses (list[str], optional): List of loss names to plot.
                                             If None, plots all found losses.
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

    # Create subplots
    n_losses = len(loss_keys)
    cols = min(3, n_losses)  # Max 3 columns
    rows = (n_losses + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_losses == 1:
        axes = [axes]
    elif rows > 1:
        axes = axes.flatten()

    fig.suptitle("Training and Validation Losses", fontsize=16, y=0.98)

    for i, loss_key in enumerate(loss_keys):
        ax = axes[i]

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
        ax.set_ylim(bottom=0)

    # Hide unused subplots
    for i in range(n_losses, len(axes)):
        axes[i].set_visible(False)

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
    Generates and saves/shows a horizontal comparison grid (real vs. generated).
    Automatically handles 256x256 or 64x64 output based on set_name.
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
        # Handle tuple output from generator (e.g., 256px and 64px images)
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
        ax_real = axs[0, i]
        ax_real.imshow(denormalize_image(real_images[i].cpu()).permute(1, 2, 0))
        ax_real.set_title(f"#{pokemon_ids[i]}: {descriptions[i][:35]}...", fontsize=10)
        ax_real.axis("off")

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

    # Save the figure and optionally show it
    save_path = os.path.join(output_dir, f"{epoch:03d}_{set_name}_comparison.png")
    plt.savefig(save_path)

    if show_inline:
        plt.show()
    else:
        plt.close(fig)
