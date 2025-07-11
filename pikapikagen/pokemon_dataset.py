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
