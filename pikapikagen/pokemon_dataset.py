import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
import os
from download_dataset import download_dataset_if_not_exists
from pathlib import Path

class PokemonDataset(Dataset):
    def __init__(self, tokenizer, csv_path="dataset/pokedex-main/data/pokemon.csv",
                 image_dir="dataset/pokedex-main/images/small_images",
                 max_length=128):
        """
        Dataset per Pokemon: testo (descrizione) -> immagine (sprite)

        Args:
            csv_path: Percorso al file CSV con i dati
            image_dir: Directory contenente le immagini dei Pokemon
            tokenizer: Tokenizer per il preprocessing del testo (es. BERT)
            max_length: Lunghezza massima delle sequenze tokenizzate
        """
        download_dataset_if_not_exists()

        self.df = pd.read_csv(csv_path, encoding='utf-16 LE', delimiter='\t')
        self.image_dir = Path(image_dir)
        print(f"Dataset caricato: {len(self.df)} Pokemon con descrizioni e immagini")

        self.tokenizer = tokenizer
        self.max_length = max_length

        # --- Pipeline di trasformazione ---
        # Trasformazioni finali (su tensori)
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizza a [-1, 1]
        ])

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

        # Gestisce la trasparenza: ricombina l'immagine con uno sfondo bianco
        background = Image.new('RGB', image_rgba.size, (255, 255, 255))
        background.paste(image_rgba, mask=image_rgba.split()[-1])

        # Applica le trasformazioni finali (ToTensor, Resize, Normalize)
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

def get_dataloader(dataset, batch_size=16, shuffle=True, num_workers=0, RANDOM_SEED = 42):
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


def create_training_setup(tokenizer, train_val_split=0.8, batch_size=16, num_workers=0,
                         num_viz_samples=4, random_seed=42):
    """
    Crea un setup completo per il training con dataset, dataloader e batch fissi per visualizzazione.

    Args:
        tokenizer: Tokenizer per il preprocessing del testo
        train_val_split: Frazione del dataset da usare per il training (default: 0.8)
        batch_size: Dimensione del batch per i dataloader
        num_workers: Numero di worker per il caricamento parallelo
        num_viz_samples: Numero di campioni per i batch di visualizzazione
        random_seed: Seed per la riproducibilità

    Returns:
        dict: Contiene train_loader, val_loader, fixed_train_batch, fixed_val_batch,
              fixed_train_attention_batch, fixed_val_attention_batch
    """

    # --- Creazione dei Dataset ---
    # Crea un'istanza per il training e la validazione (senza aumentazioni interne)
    train_full_dataset = PokemonDataset(tokenizer=tokenizer)
    val_full_dataset = PokemonDataset(tokenizer=tokenizer)

    # --- Divisione deterministica degli indici ---
    assert len(train_full_dataset) == len(val_full_dataset)
    dataset_size = len(train_full_dataset)
    train_size = int(train_val_split * dataset_size)
    val_size = dataset_size - train_size

    train_indices_subset, val_indices_subset = random_split(
        TensorDataset(torch.arange(dataset_size)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    train_dataset = Subset(train_full_dataset, train_indices_subset.indices)
    val_dataset = Subset(val_full_dataset, val_indices_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Creazione deterministica dei batch per la visualizzazione ---
    vis_generator = torch.Generator().manual_seed(random_seed)
    fixed_train_batch = next(
        iter(
            DataLoader(
                train_dataset,
                batch_size=num_viz_samples,
                shuffle=True,
                generator=vis_generator,
            )
        )
    )
    fixed_val_batch = next(
        iter(DataLoader(val_dataset, batch_size=num_viz_samples, shuffle=False))
    )  # la validazione non ha shuffle

    vis_generator.manual_seed(random_seed)  # Reset per coerenza
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

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'fixed_train_batch': fixed_train_batch,
        'fixed_val_batch': fixed_val_batch,
        'fixed_train_attention_batch': fixed_train_attention_batch,
        'fixed_val_attention_batch': fixed_val_attention_batch,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
    }


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
