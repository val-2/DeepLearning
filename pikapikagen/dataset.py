import os
import urllib.request
import zipfile
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image
from typing import TypedDict
import torch


class PokemonSample(TypedDict):
    text: torch.Tensor  # Text already tokenized
    image: torch.Tensor
    description: str  # Text before tokenization
    pokemon_name: str
    idx: int
    attention_mask: torch.Tensor


def reporthook(block_num, block_size, total_size):
    if block_num % 16384 == 0:
        print(f"Downloading... {block_num * block_size / (1024 * 1024):.2f} MB")


def download_dataset_if_not_exists():
    dataset_dir = "dataset"
    pokedex_main_dir = os.path.join(dataset_dir, "pokedex-main")
    zip_url = "https://github.com/cristobalmitchell/pokedex/archive/refs/heads/main.zip"
    zip_path = "pokedex_main.zip"

    if os.path.exists(pokedex_main_dir):
        print(f"{pokedex_main_dir} already exists. Skipping download.")
        return

    os.makedirs(dataset_dir, exist_ok=True)

    print("Downloading dataset...")
    urllib.request.urlretrieve(zip_url, zip_path, reporthook)
    print("Download complete.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Extraction complete.")

    os.remove(zip_path)


class PokemonDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        csv_path="dataset/pokedex-main/data/pokemon.csv",
        image_dir="dataset/pokedex-main/images/small_images",
        max_length=128,
        augmentation_transforms=None,
    ):
        self.df = pd.read_csv(csv_path, encoding="utf-16 LE", delimiter="\t")
        self.image_dir = Path(image_dir)
        print(f"Dataset caricato: {len(self.df)} Pokemon con descrizioni e immagini")

        self.tokenizer = tokenizer
        self.max_length = max_length

        if augmentation_transforms is not None:
            self.final_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((256, 256), antialias=True),
                    augmentation_transforms,
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),  # Normalizza a [-1, 1]
                ]
            )
        else:
            self.final_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((256, 256), antialias=True),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ),  # Normalizza a [-1, 1]
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> PokemonSample:
        # Ottieni la riga corrispondente
        row = self.df.iloc[idx]

        # === PREPROCESSING DEL TESTO ===
        description = str(row["description"])

        # Tokenizza il testo
        encoded = self.tokenizer(
            description,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Estrai token_ids e attention_mask
        text_ids = encoded["input_ids"].squeeze(0)  # Rimuovi la dimensione batch
        attention_mask = encoded["attention_mask"].squeeze(0)

        # === CARICAMENTO E PREPROCESSING DELL'IMMAGINE ===
        # Costruisce il percorso dell'immagine
        image_filename = f"{row['national_number']:03d}.png"
        image_path = self.image_dir / image_filename

        # Carica l'immagine
        image_rgba = Image.open(image_path).convert("RGBA")

        # Gestisce la trasparenza: ricombina l'immagine con uno sfondo bianco
        background = Image.new("RGB", image_rgba.size, (255, 255, 255))
        background.paste(image_rgba, mask=image_rgba.split()[-1])

        # Applica le trasformazioni finali (ToTensor, Resize, Normalize)
        image_tensor = self.final_transform(background)

        # Costruisce il risultato (matches pokemon_dataset.py structure)
        sample = {
            "text": text_ids,
            "image": image_tensor,
            "description": description,
            "pokemon_name": row["english_name"],
            "idx": idx,
            "attention_mask": attention_mask,
        }

        return sample


download_dataset_if_not_exists()
print("Dataset ready!")
