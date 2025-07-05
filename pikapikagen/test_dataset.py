import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pokemon_dataset import PokemonDataset

def test_dataset_integrity():
    """
    Verifica che tutte le immagini nel PokemonDataset abbiano la dimensione corretta.
    """
    print("Avvio del test di integrità del dataset...")

    # 1. Inizializza il Tokenizer e il Dataset
    try:
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
        dataset = PokemonDataset(tokenizer=tokenizer)
        print(f"Dataset caricato con {len(dataset)} campioni.")
    except Exception as e:
        print(f"Errore durante l'inizializzazione del dataset: {e}")
        return

    mismatched_images = []
    expected_shape = (3, 215, 215)

    # 2. Itera su tutti i campioni del dataset
    print(f"Verifica della dimensione di ogni immagine. Dimensione attesa: {expected_shape}")
    for i in tqdm(range(len(dataset)), desc="Controllo campioni"):
        try:
            sample = dataset[i]
            image_tensor = sample['image']

            # 3. Controlla la dimensione del tensore dell'immagine
            if image_tensor.shape != expected_shape:
                mismatched_images.append({
                    'index': i,
                    'pokemon_name': sample.get('pokemon_name', 'N/A'),
                    'shape': image_tensor.shape
                })
        except Exception as e:
            mismatched_images.append({
                'index': i,
                'pokemon_name': 'Errore nel caricamento',
                'shape': f"Errore: {e}"
            })

    # 4. Stampa il risultato del test
    print("\n--- Risultato del Test di Integrità ---")
    if not mismatched_images:
        print(f"✅ Successo! Tutte le {len(dataset)} immagini hanno la dimensione corretta {expected_shape}.")
    else:
        print(f"❌ Fallito! Trovate {len(mismatched_images)} immagini con dimensioni non corrette:")
        for item in mismatched_images:
            print(f"  - Indice: {item['index']}, Pokemon: {item['pokemon_name']}, Dimensione Trovata: {item['shape']}")
    print("------------------------------------")

if __name__ == "__main__":
    test_dataset_integrity()
