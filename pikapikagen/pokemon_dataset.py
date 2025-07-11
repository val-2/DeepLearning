import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
from download_dataset import download_dataset_if_not_exists
from pathlib import Path

class PokemonDataset(Dataset):
    def __init__(self, tokenizer, csv_path="dataset/pokedex-main/data/pokemon.csv",
                 image_dir="dataset/pokedex-main/images/small_images",
                 max_length=128,
                 use_geometric_aug: bool = True,
                 use_color_aug: bool = True,
                 use_blur_aug: bool = True,
                 use_cutout_aug: bool = True):
        """
        Dataset per Pokemon: testo (descrizione) -> immagine (sprite)

        Args:
            csv_path: Percorso al file CSV con i dati
            image_dir: Directory contenente le immagini dei Pokemon
            tokenizer: Tokenizer per il preprocessing del testo (es. BERT)
            max_length: Lunghezza massima delle sequenze tokenizzate
            use_geometric_aug: Abilita le aumentazioni geometriche (flip, rotazione, etc.).
            use_color_aug: Abilita le aumentazioni di colore (jitter).
            use_blur_aug: Abilita l'aumentazione di sfocatura (blur).
            use_cutout_aug: Abilita l'aumentazione di cutout (random erasing).
        """
        download_dataset_if_not_exists()

        self.df = pd.read_csv(csv_path, encoding='utf-16 LE', delimiter='\t')
        self.image_dir = Path(image_dir)
        print(f"Dataset caricato: {len(self.df)} Pokemon con descrizioni e immagini")

        self.tokenizer = tokenizer
        self.max_length = max_length

        # --- Impostazioni per l'aumentazione ---
        self.use_geometric_aug = use_geometric_aug
        self.use_color_aug = use_color_aug
        self.use_blur_aug = use_blur_aug
        self.use_cutout_aug = use_cutout_aug

        # --- Pipeline di trasformazione ---
        # 1. Aumentazioni di colore (su immagini PIL)
        self.color_augment = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        # 2. Aumentazione di sfocatura (su immagini PIL)
        self.blur_augment = transforms.GaussianBlur(kernel_size=3)

        # 3. Trasformazioni finali (su tensori)
        final_transforms = [
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizza a [-1, 1]
        ]
        if self.use_cutout_aug:
            # RandomErasing si aspetta un tensore normalizzato.
            # Il valore '1' in un tensore normalizzato in [-1, 1] corrisponde al bianco.
            final_transforms.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=1))

        self.final_transform = transforms.Compose(final_transforms)

    def __len__(self):
        """Restituisce il numero totale di campioni"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Restituisce un singolo campione: (testo_tokenizzato, immagine_tensor)

        Args:
            idx: Indice del campione

        Returns:
            dict: {
                'text': tensor dei token IDs,
                'attention_mask': maschera di attenzione per il testo,
                'image': tensor dell'immagine [C, H, W],
                'description': stringa originale (opzionale, per debug)
            }
        """
        # Ottieni la riga corrispondente
        row = self.df.iloc[idx]

        # === PREPROCESSING DEL TESTO ===
        description = str(row['description'])

        # Tokenizza il testo
        encoded = self.tokenizer(
            description,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Estrai token_ids e attention_mask
        text_ids = encoded['input_ids'].squeeze(0)  # Rimuovi la dimensione batch
        attention_mask = encoded['attention_mask'].squeeze(0)

        # === CARICAMENTO E PREPROCESSING DELL'IMMAGINE ===
        # Costruisce il percorso dell'immagine
        image_filename = f"{row['national_number']:03d}.png"
        image_path = self.image_dir/image_filename

        # Carica l'immagine
        image_rgba = Image.open(image_path).convert('RGBA')

        # Separa i canali RGB e Alpha per l'aumentazione selettiva
        alpha_mask = image_rgba.split()[-1]
        image_rgb = image_rgba.convert('RGB')

        # --- APPLICAZIONE DELLE AUMENTAZIONI ---

        # 1. Aumentazione di colore (solo su RGB)
        if self.use_color_aug:
            image_rgb = self.color_augment(image_rgb)

        # 2. Aumentazioni geometriche (su RGB e maschera alpha)
        if self.use_geometric_aug:
            # Flip orizzontale
            if torch.rand(1) < 0.5:
                image_rgb = TF.hflip(image_rgb)  # type: ignore[arg-type]
                alpha_mask = TF.hflip(alpha_mask)  # type: ignore[arg-type]

            # Trasformazione Affine (rotazione, scala, traslazione)
            affine_params = transforms.RandomAffine.get_params(
                degrees=[-15, 15],
                translate=[0.1, 0.1],
                scale_ranges=[0.9, 1.1],
                shears=None,
                img_size=list(image_rgb.size)  # type: ignore[arg-type]
            )
            angle, translations, scale, shear = affine_params
            image_rgb = TF.affine(
                image_rgb,  # type: ignore[arg-type]
                angle=angle,
                translate=list(translations),
                scale=scale,
                shear=list(shear),
                interpolation=TF.InterpolationMode.BILINEAR
            )
            alpha_mask = TF.affine(
                alpha_mask,  # type: ignore[arg-type]
                angle=angle,
                translate=list(translations),
                scale=scale,
                shear=list(shear),
                interpolation=TF.InterpolationMode.NEAREST
            )

        # 3. Aumentazione di sfocatura (solo su RGB, con probabilità)
        if self.use_blur_aug and torch.rand(1) < 0.5:
            image_rgb = self.blur_augment(image_rgb)

        # Gestisce la trasparenza: ricombina l'immagine aumentata con uno sfondo bianco
        background = Image.new('RGB', image_rgb.size, (255, 255, 255))  # type: ignore[arg-type]
        background.paste(image_rgb, mask=alpha_mask)  # type: ignore[arg-type]

        # Applica le trasformazioni finali (ToTensor, Resize, Normalize, Cutout)
        image_tensor = self.final_transform(background)

        # Costruisce il risultato
        sample = {
            'text': text_ids,
            'image': image_tensor,
            'description': description,  # Per debug o visualizzazione
            'pokemon_name': row['english_name'],
            'idx': idx,
            'attention_mask': attention_mask,
        }

        return sample

# === FUNZIONI DI UTILITÀ ===

def get_dataloader(dataset, batch_size=16, shuffle=True, num_workers=0):
    """
    Crea un DataLoader per il dataset

    Args:
        dataset: Istanza di PokemonDataset
        batch_size: Dimensione del batch
        shuffle: Se mescolare i dati
        num_workers: Numero di worker per il caricamento parallelo

    Returns:
        DataLoader di PyTorch
    """

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Migliora le prestazioni con GPU
    )


if __name__ == "__main__":
    # Test del dataset
    from transformers import AutoTokenizer

    # Carica il tokenizer
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")

    # Crea il dataset
    dataset = PokemonDataset(tokenizer=tokenizer)

    # Prova a caricare alcuni campioni
    print(f"Dataset size: {len(dataset)}")

    # Carica il primo campione
    sample = dataset[0]
    print("Primo campione:")
    print(f"  - Testo shape: {sample['text'].shape}")
    print(f"  - Immagine shape: {sample['image'].shape}")
    print(f"  - Descrizione: {sample['description'][:100]}...")
    print(f"  - Pokemon: {sample['pokemon_name']}")
    print(f"  - Index: {sample['idx']}")
    print(f"  - Attention mask: {sample['attention_mask']}")

    # Crea un dataloader
    dataloader = get_dataloader(dataset, batch_size=4)

    # Testa un batch
    batch = next(iter(dataloader))
    print("\nPrimo batch:")
    print(f"  - Testo batch shape: {batch['text'].shape}")
    print(f"  - Immagine batch shape: {batch['image'].shape}")
