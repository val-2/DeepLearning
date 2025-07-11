import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')

from pokemon_dataset import PokemonDataset
from model import PikaPikaGen
from losses import VGGPerceptualLoss, SSIMLoss, SobelLoss
from clip_loss import create_clip_loss

# --- Parametri di Training Fissi ---
MODEL_NAME = "prajjwal1/bert-mini"
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 2e-4
NOISE_DIM = 100
TRAIN_VAL_SPLIT = 0.9
NUM_WORKERS = 0
# Numero di campioni da visualizzare nelle griglie di confronto
NUM_VIZ_SAMPLES = 4
# Frequenza di salvataggio checkpoint e visualizzazioni
SAVE_EVERY_N_EPOCHS = 5
# Seed per la riproducibilità
RANDOM_SEED = 42
# Pesi per le diverse loss
LAMBDA_L1 = 0.0
LAMBDA_PERCEPTUAL = 0.0
LAMBDA_SSIM = 0
LAMBDA_SOBEL = 0.0
LAMBDA_CLIP = 1.0

USE_GEOMETRIC_AUG = False
USE_COLOR_AUG = False
USE_BLUR_AUG = False
USE_CUTOUT_AUG = False


# --- Setup del Dispositivo ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print(f"PyTorch CUDA version: {torch.version.cuda}")  # type: ignore
    print(f"Numero di GPU disponibili: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
elif torch.xpu.is_available():
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
    list_of_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth.tar'))
    if not list_of_files:
        return []

    valid_checkpoints = []
    for f in list_of_files:
        try:
            epoch = int(os.path.basename(f).split('_')[-1].split('.')[0])
            valid_checkpoints.append((epoch, f))
        except (ValueError, IndexError):
            print(f"Attenzione: impossibile analizzare il numero di epoca dal nome file: {f}")

    # Ordina per epoca in ordine decrescente (dal più recente al più vecchio)
    valid_checkpoints.sort(key=lambda x: x[0], reverse=True)

    return [file_path for epoch, file_path in valid_checkpoints]


def save_checkpoint(epoch, model_G, optimizer_G, loss, is_best=False):
    """
    Salva un checkpoint del modello, dell'ottimizzatore e della loss.

    Args:
        epoch (int): L'epoca corrente.
        model_G (nn.Module): Il modello generatore da salvare.
        optimizer_G (torch.optim.Optimizer): L'ottimizzatore del generatore da salvare.
        loss (float): La loss di validazione corrente.
        is_best (bool, optional): Se True, salva il modello anche come "best_model.pth.tar".
    """
    state = {
        'epoch': epoch,
        'model_G_state_dict': model_G.module.state_dict() if isinstance(model_G, nn.DataParallel) else model_G.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'loss': loss,
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

def save_attention_visualization(epoch, model, tokenizer, batch, device, set_name):
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
        token_ids = batch['text'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        if token_ids.dim() > 1: # Assicura un batch di 1
             token_ids = token_ids[0].unsqueeze(0)
             attention_mask = attention_mask[0].unsqueeze(0)

        pokemon_id = batch['idx'][0]
        description = batch['description'][0]

        model_to_use = model.module if isinstance(model, nn.DataParallel) else model
        generated_image, attention_maps, initial_context_weights = model_to_use(token_ids, attention_mask, return_attentions=True)

    decoder_attention_maps = [m for m in attention_maps if m is not None]

    if not decoder_attention_maps or initial_context_weights is None:
        print(f"Epoch {epoch}: Mappe di attenzione non disponibili. Salto la visualizzazione.")
        return

    tokens_all = tokenizer.convert_ids_to_tokens(token_ids.squeeze(0))
    display_tokens = []
    for i, token in enumerate(tokens_all):
        if token not in [tokenizer.sep_token, tokenizer.pad_token] and attention_mask[0, i] == 1:
            display_tokens.append({'token': token, 'index': i})

    if not display_tokens:
        print(f"Epoch {epoch}: Nessun token valido da visualizzare per '{description}'. Salto.")
        return

    token_indices_to_display = [t['index'] for t in display_tokens]
    img_tensor_cpu = denormalize_image(generated_image.squeeze(0).cpu()).permute(1, 2, 0)
    num_decoder_layers = len(decoder_attention_maps)
    num_tokens = len(display_tokens)

    # --- Creazione del Plot ---
    # Calcola dinamicamente layout e dimensioni
    cols = min(num_tokens, 8)
    rows_per_layer = (num_tokens + cols - 1) // cols
    num_main_rows = 2 + num_decoder_layers # Immagine, Bar chart, e N layer di attenzione
    # Altezza per immagine, bar chart, e poi per ogni riga di ogni layer
    height_ratios = [3, 2] + [2 * rows_per_layer] * num_decoder_layers
    fig_height = sum(height_ratios)
    fig_width = max(20, 2.5 * cols)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs_main = fig.add_gridspec(num_main_rows, 1, height_ratios=height_ratios, hspace=1.2)
    fig.suptitle(f"Epoch {epoch}: Attention Visualization for Pokémon #{pokemon_id} ({set_name.capitalize()})", fontsize=24)

    # --- 1. Immagine Generata e Prompt ---
    ax_main_img = fig.add_subplot(gs_main[0])
    ax_main_img.imshow(img_tensor_cpu)
    ax_main_img.set_title("Generated Image", fontsize=18)
    ax_main_img.text(0.5, -0.1, f"Prompt: {description}", ha='center', va='top', transform=ax_main_img.transAxes, fontsize=14, wrap=True)
    ax_main_img.axis('off')

    # --- 2. Attenzione Iniziale per il Contesto (bar chart) ---
    ax_initial_attn = fig.add_subplot(gs_main[1])
    initial_weights_squeezed = initial_context_weights.squeeze().cpu().numpy()
    token_strings = [t['token'] for t in display_tokens]
    token_indices = [t['index'] for t in display_tokens]
    relevant_weights = initial_weights_squeezed[token_indices]
    ax_initial_attn.bar(np.arange(len(token_strings)), relevant_weights, color='skyblue')
    ax_initial_attn.set_xticks(np.arange(len(token_strings)))
    ax_initial_attn.set_xticklabels(token_strings, rotation=45, ha="right", fontsize=10)
    ax_initial_attn.set_title("Initial Context Attention (Global)", fontsize=16)
    ax_initial_attn.set_ylabel("Weight", fontsize=12)
    ax_initial_attn.grid(axis='y', linestyle='--', alpha=0.7)

    # --- 3. Attenzione per Strato del Decoder (griglie di heatmap) ---
    for i, layer_attn_map in enumerate(decoder_attention_maps):
        map_size_flat = layer_attn_map.shape[1]
        map_side = int(np.sqrt(map_size_flat))
        layer_title = f"Decoder Cross-Attention Layer {i+1} (Size: {map_side}x{map_side})"
        layer_attn_map_squeezed = layer_attn_map.squeeze(0).cpu()

        # Seleziona solo le mappe di attenzione per i token che visualizziamo
        relevant_attn_maps = layer_attn_map_squeezed[:, token_indices_to_display]

        # Trova i valori min/max per questo strato per la colorbar
        vmin = relevant_attn_maps.min()
        vmax = relevant_attn_maps.max()

        # Crea una subgrid per questo strato (con una colonna in più per la colorbar)
        gs_layer = gs_main[2 + i].subgridspec(rows_per_layer, cols + 1, wspace=0.2, hspace=0.4, width_ratios=[*([1] * cols), 0.1])


        # Crea tutti gli assi per la griglia
        axes_in_layer = [fig.add_subplot(gs_layer[r, c]) for r in range(rows_per_layer) for c in range(cols)]

        # Usa la posizione del primo asse per il titolo
        if axes_in_layer:
            y_pos = axes_in_layer[0].get_position().y1
            fig.text(0.5, y_pos + 0.01, layer_title, ha='center', va='bottom', fontsize=16, weight='bold')


        for j, token_info in enumerate(display_tokens):
            ax = axes_in_layer[j]
            attn_for_token = layer_attn_map_squeezed[:, token_info['index']]
            heatmap = attn_for_token.reshape(map_side, map_side)
            im = ax.imshow(heatmap, cmap='jet', interpolation='nearest', vmin=vmin, vmax=vmax)
            ax.set_title(f"'{token_info['token']}'", fontsize=12)
            ax.axis('off')

        # Aggiungi la colorbar
        cax = fig.add_subplot(gs_layer[:, -1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15, fontsize=12)

        # Pulisce gli assi non usati nella griglia
        for j in range(num_tokens, len(axes_in_layer)):
            axes_in_layer[j].axis('off')


    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_path = os.path.join(IMAGE_DIR, f"{epoch:03d}_{set_name}_attention_visualization.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def save_comparison_grid(epoch, model, batch, set_name, device):
    """
    Genera e salva una griglia di confronto orizzontale (reale vs. generato).

    Args:
        epoch (int): L'epoca corrente.
        model (PikaPikaGen): Il modello generatore.
        batch (dict): Un batch di dati (training o validazione).
        set_name (str): 'train' o 'val', per il titolo e nome del file.
        device (torch.device): Il dispositivo su cui eseguire i calcoli.
    """
    model.eval()
    token_ids = batch['text'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    real_images = batch['image']
    pokemon_ids = batch['idx']
    descriptions = batch['description']
    num_images = real_images.size(0)

    with torch.no_grad():
        generated_images = model(token_ids, attention_mask)

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


def train(continue_from_last_checkpoint: bool = True, epochs_to_run: int = NUM_EPOCHS, use_multi_gpu: bool = True, use_geometric_aug: bool = USE_GEOMETRIC_AUG, use_color_aug: bool = USE_COLOR_AUG, use_blur_aug: bool = USE_BLUR_AUG, use_cutout_aug: bool = USE_CUTOUT_AUG, save_every_n_epochs: int = SAVE_EVERY_N_EPOCHS):
    """
    Funzione principale per l'addestramento e la validazione del modello PikaPikaGen.

    Args:
        continue_from_last_checkpoint (bool): Se True, cerca di riprendere dall'ultimo
                                              checkpoint. Se non lo trova, inizia da zero.
        epochs_to_run (int): Il numero di epoche per cui eseguire l'addestramento.
        use_multi_gpu (bool): Se True, abilita l'uso di DataParallel se sono disponibili più GPU.
        use_geometric_aug (bool): Abilita le aumentazioni geometriche.
        use_color_aug (bool): Abilita le aumentazioni di colore.
        use_blur_aug (bool): Abilita l'aumentazione di sfocatura.
        use_cutout_aug (bool): Abilita l'aumentazione di cutout.
        save_every_n_epochs (int): La frequenza (in epoche) con cui salvare i checkpoint
                                   e generare le visualizzazioni.
    """
    if all(loss <= 0 for loss in [LAMBDA_L1, LAMBDA_PERCEPTUAL, LAMBDA_SSIM, LAMBDA_SOBEL, LAMBDA_CLIP]):
        raise ValueError(
            "Tutti i pesi (lambda) delle loss sono a zero o negativi. "
            "Almeno una loss deve essere abilitata con un peso positivo per l'addestramento."
        )

    print("--- Impostazioni di Training ---")
    print(f"Utilizzo del dispositivo: {DEVICE}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Creazione dei Dataset ---
    # Crea un'istanza per il training con le aumentazioni attive
    train_full_dataset = PokemonDataset(
        tokenizer=tokenizer,
        use_geometric_aug=use_geometric_aug,
        use_color_aug=use_color_aug,
        use_blur_aug=use_blur_aug,
        use_cutout_aug=use_cutout_aug
    )
    # Crea un'istanza per la validazione SENZA aumentazioni per una valutazione stabile
    val_full_dataset = PokemonDataset(
        tokenizer=tokenizer,
        use_geometric_aug=False,
        use_color_aug=False,
        use_blur_aug=False,
        use_cutout_aug=False
    )

    # --- Divisione deterministica degli indici ---
    # Assicurati che entrambi i dataset abbiano la stessa lunghezza
    assert len(train_full_dataset) == len(val_full_dataset)
    dataset_size = len(train_full_dataset)
    train_size = int(TRAIN_VAL_SPLIT * dataset_size)
    val_size = dataset_size - train_size

    # Usa random_split su un range di indici per ottenere Subset di indici
    # in modo deterministico e disgiunto.
    train_indices_subset, val_indices_subset = random_split(
        TensorDataset(torch.arange(dataset_size)),  # type: ignore
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # Crea i dataset finali usando i Subset degli indici sui rispettivi dataset
    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices_subset.indices)
    val_dataset = torch.utils.data.Subset(val_full_dataset, val_indices_subset.indices)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Creazione deterministica dei batch per la visualizzazione ---
    # Usiamo un generatore separato per il dataloader per non interferire con lo shuffle del training
    vis_generator = torch.Generator().manual_seed(RANDOM_SEED)
    fixed_train_batch = next(iter(DataLoader(train_dataset, batch_size=NUM_VIZ_SAMPLES, shuffle=True, generator=vis_generator)))
    fixed_val_batch = next(iter(DataLoader(val_dataset, batch_size=NUM_VIZ_SAMPLES, shuffle=False))) # la validazione non ha shuffle

    vis_generator.manual_seed(RANDOM_SEED) # Reset per coerenza
    fixed_train_attention_batch = next(iter(DataLoader(train_dataset, batch_size=1, shuffle=True, generator=vis_generator)))
    fixed_val_attention_batch = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=False)))

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    model_G = PikaPikaGen(text_encoder_model_name=MODEL_NAME, noise_dim=NOISE_DIM).to(DEVICE)

    # --- Supporto Multi-GPU ---
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Utilizzo di {torch.cuda.device_count()} GPU con nn.DataParallel.")
        model_G = nn.DataParallel(model_G)
    elif use_multi_gpu and torch.cuda.device_count() <= 1:
        print("Multi-GPU richiesto ma solo una o nessuna GPU disponibile. Proseguo con una singola GPU/CPU.")

    optimizer_G = optim.Adam(model_G.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    criterion_l1 = nn.L1Loss().to(DEVICE)
    criterion_perceptual = VGGPerceptualLoss(device=DEVICE).to(DEVICE)
    criterion_ssim = SSIMLoss().to(DEVICE)
    criterion_sobel = SobelLoss().to(DEVICE)
    criterion_clip = create_clip_loss(device=str(DEVICE)) if LAMBDA_CLIP > 0 else None

    start_epoch = 1
    best_val_loss = float('inf')

    # --- Logica per continuare il training ---
    if continue_from_last_checkpoint:
        sorted_checkpoints = find_sorted_checkpoints(CHECKPOINT_DIR)
        checkpoint_loaded = False
        if sorted_checkpoints:
            for checkpoint_path in sorted_checkpoints:
                try:
                    print(f"--- Ripresa del training dal checkpoint: {os.path.basename(checkpoint_path)} ---")
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

                    # Il nome della chiave per il modello G potrebbe essere il vecchio 'model_state_dict'
                    if 'model_G_state_dict' in checkpoint:
                        model_G.load_state_dict(checkpoint['model_G_state_dict'])
                        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                    elif 'model_state_dict' in checkpoint: # Compatibilità con vecchi checkpoint
                         model_G.load_state_dict(checkpoint['model_state_dict'])
                         optimizer_G.load_state_dict(checkpoint['optimizer_state_dict'])


                    start_epoch = checkpoint['epoch'] + 1
                    best_val_loss = checkpoint.get('loss', float('inf'))
                    if best_val_loss is None:
                        best_val_loss = float('inf')

                    print(f"Checkpoint caricato. Si riparte dall'epoca {start_epoch}.")
                    checkpoint_loaded = True
                    break # Usciamo dal loop se il caricamento ha successo
                except Exception as e:
                    print(f"Errore nel caricamento del checkpoint {os.path.basename(checkpoint_path)}: {e}")
                    print("Il file potrebbe essere corrotto o incompatibile. Tento con il checkpoint precedente, se disponibile.")

        if not checkpoint_loaded:
            print("--- Nessun checkpoint valido trovato. Inizio un nuovo training da zero. ---")
    else:
        print("--- Inizio un nuovo training da zero (opzione 'continue_from_last_checkpoint' disabilitata). ---")


    final_epoch = start_epoch + epochs_to_run - 1
    print(f"--- Inizio Training (da epoca {start_epoch} a {final_epoch}) ---")

    for epoch in range(start_epoch, final_epoch + 1):
        model_G.train()

        train_loss_g, train_loss_l1, train_loss_perceptual, train_loss_ssim, train_loss_sobel, train_loss_clip = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{final_epoch} [Training]")
        for batch in pbar:
            token_ids = batch['text'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            real_images = batch['image'].to(DEVICE)
            descriptions = batch['description'] if LAMBDA_CLIP > 0 else None

            # ---------------------
            #  Training del Generatore
            # ---------------------
            optimizer_G.zero_grad()

            generated_images = model_G(token_ids, attention_mask)

            # Calcola le loss solo se il loro peso è maggiore di zero
            loss_l1 = criterion_l1(generated_images, real_images) if LAMBDA_L1 > 0 else torch.tensor(0.0, device=DEVICE)
            loss_perceptual = criterion_perceptual(generated_images, real_images) if LAMBDA_PERCEPTUAL > 0 else torch.tensor(0.0, device=DEVICE)
            loss_ssim = criterion_ssim(generated_images, real_images) if LAMBDA_SSIM > 0 else torch.tensor(0.0, device=DEVICE)
            loss_sobel = criterion_sobel(generated_images, real_images) if LAMBDA_SOBEL > 0 else torch.tensor(0.0, device=DEVICE)
            loss_clip = criterion_clip(generated_images, descriptions) if LAMBDA_CLIP > 0 and criterion_clip is not None else torch.tensor(0.0, device=DEVICE)


            # Loss totale del generatore
            loss_G = LAMBDA_L1 * loss_l1 + LAMBDA_PERCEPTUAL * loss_perceptual + LAMBDA_SSIM * loss_ssim + LAMBDA_SOBEL * loss_sobel + LAMBDA_CLIP * loss_clip
            loss_G.backward()
            optimizer_G.step()

            # Aggiorna le loss per il logging
            train_loss_g += loss_G.item()
            train_loss_l1 += loss_l1.item()
            train_loss_perceptual += loss_perceptual.item()
            train_loss_ssim += loss_ssim.item()
            train_loss_sobel += loss_sobel.item()
            train_loss_clip += loss_clip.item()

            pbar.set_postfix({
                'G_loss': f"{loss_G.item():.4f}",
                'L1': f"{loss_l1.item():.4f}",
                'Perceptual': f"{loss_perceptual.item():.4f}",
                'SSIM': f"{loss_ssim.item():.4f}",
                'Sobel': f"{loss_sobel.item():.4f}",
                'CLIP': f"{loss_clip.item():.4f}"
            })

        avg_train_loss_g = train_loss_g / len(train_loader)
        avg_train_loss_l1 = train_loss_l1 / len(train_loader)
        avg_train_loss_perceptual = train_loss_perceptual / len(train_loader)
        avg_train_loss_ssim = train_loss_ssim / len(train_loader)
        avg_train_loss_sobel = train_loss_sobel / len(train_loader)
        avg_train_loss_clip = train_loss_clip / len(train_loader)


        # Esegui la validazione, il salvataggio e la visualizzazione solo ogni N epoche o all'ultima epoca
        if epoch % save_every_n_epochs == 0 or epoch == final_epoch:
            model_G.eval()
            val_loss_l1, val_loss_perceptual, val_loss_ssim, val_loss_sobel, val_loss_clip = 0.0, 0.0, 0.0, 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    token_ids = batch['text'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    real_images = batch['image'].to(DEVICE)
                    descriptions = batch['description'] if LAMBDA_CLIP > 0 else None
                    generated_images = model_G(token_ids, attention_mask)

                    if LAMBDA_L1 > 0:
                        val_loss_l1 += criterion_l1(generated_images, real_images).item()
                    if LAMBDA_PERCEPTUAL > 0:
                        val_loss_perceptual += criterion_perceptual(generated_images, real_images).item()
                    if LAMBDA_SSIM > 0:
                        val_loss_ssim += criterion_ssim(generated_images, real_images).item()
                    if LAMBDA_SOBEL > 0:
                        val_loss_sobel += criterion_sobel(generated_images, real_images).item()
                    if LAMBDA_CLIP > 0 and criterion_clip is not None:
                        val_loss_clip += criterion_clip(generated_images, descriptions).item()

            avg_val_loss_l1 = val_loss_l1 / len(val_loader) if val_loss_l1 > 0 else 0.0
            avg_val_loss_perceptual = val_loss_perceptual / len(val_loader) if val_loss_perceptual > 0 else 0.0
            avg_val_loss_ssim = val_loss_ssim / len(val_loader) if val_loss_ssim > 0 else 0.0
            avg_val_loss_sobel = val_loss_sobel / len(val_loader) if val_loss_sobel > 0 else 0.0
            avg_val_loss_clip = val_loss_clip / len(val_loader) if val_loss_clip > 0 else 0.0

            # Calcola la loss di validazione totale come somma pesata per il checkpointing
            avg_val_loss = (LAMBDA_L1 * avg_val_loss_l1 +
                            LAMBDA_PERCEPTUAL * avg_val_loss_perceptual +
                            LAMBDA_SSIM * avg_val_loss_ssim +
                            LAMBDA_SOBEL * avg_val_loss_sobel +
                            LAMBDA_CLIP * avg_val_loss_clip)

            print(
                f"Epoch {epoch}/{final_epoch} -> "
                f"Train G_Loss: {avg_train_loss_g:.4f} (L1: {avg_train_loss_l1:.4f}, Perceptual: {avg_train_loss_perceptual:.4f}, SSIM: {avg_train_loss_ssim:.4f}, Sobel: {avg_train_loss_sobel:.4f}, CLIP: {avg_train_loss_clip:.4f}), "
                f"Val Loss (L1: {avg_val_loss_l1:.4f}, Perceptual: {avg_val_loss_perceptual:.4f}, SSIM: {avg_val_loss_ssim:.4f}, Sobel: {avg_val_loss_sobel:.4f}, CLIP: {avg_val_loss_clip:.4f})"
            )

            # --- Salvataggio Checkpoint ---
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
            save_checkpoint(epoch, model_G, optimizer_G, avg_val_loss, is_best)

            # --- Generazione Visualizzazioni ---
            print(f"Epoch {epoch}: Generazione visualizzazioni...")
            save_attention_visualization(epoch, model_G, tokenizer, fixed_train_attention_batch, DEVICE, 'train')
            save_attention_visualization(epoch, model_G, tokenizer, fixed_val_attention_batch, DEVICE, 'val')
            save_comparison_grid(epoch, model_G, fixed_train_batch, 'train', DEVICE)
            save_comparison_grid(epoch, model_G, fixed_val_batch, 'val', DEVICE)
        else:
            # Stampa solo la loss di training per le epoche intermedie
            print(
                f"Epoch {epoch}/{final_epoch} -> "
                f"Train G_Loss: {avg_train_loss_g:.4f} (L1: {avg_train_loss_l1:.4f}, Perceptual: {avg_train_loss_perceptual:.4f}, SSIM: {avg_train_loss_ssim:.4f}, Sobel: {avg_train_loss_sobel:.4f}, CLIP: {avg_train_loss_clip:.4f})"
            )

    print("Training completato!")


if __name__ == "__main__":
    # --- IMPOSTA QUI LA MODALITÀ DI TRAINING ---
    # continue_training=True -> Cerca l'ultimo checkpoint e riprende. Se non lo trova, parte da zero.
    # continue_training=False -> Parte sempre da zero.
    # epochs -> Numero di epoche per cui addestrare in questa sessione.
    # use_multi_gpu -> Se True, utilizza DataParallel se sono disponibili più GPU.
    train(
        continue_from_last_checkpoint=True,
        epochs_to_run=100,
        use_multi_gpu=True,
    )
