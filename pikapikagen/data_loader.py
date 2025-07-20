from torch.utils.data import DataLoader, Subset
import torch
from dataset import PokemonDataset
from transformers import AutoTokenizer
import math

def create_training_setup(
    tokenizer,
    test_set_size,
    val_set_size,
    batch_size,
    num_workers=0,
    num_viz_samples=4,
    random_seed=42,
    train_augmentation_pipeline=None,
):
    """
    Create a complete setup for training with dataset, dataloaders and fixed batches for visualization.
    """
    assert 0 < test_set_size < 1.0, "test_set_size must be a float between 0 and 1"
    assert 0 < val_set_size < 1.0, "val_set_size must be a float between 0 and 1"
    assert (test_set_size + val_set_size) < 1.0, "The sum of test and validation sizes must be less than 1"

    train_full_dataset = PokemonDataset(tokenizer=tokenizer, augmentation_transforms=train_augmentation_pipeline)
    # Don't use augmentation for test and validation
    test_val_full_dataset = PokemonDataset(tokenizer=tokenizer)

    dataset_size = len(train_full_dataset)

    # --- Deterministic index split ---
    # 1. Create a random reproducible permutation
    generator = torch.Generator().manual_seed(random_seed)
    shuffled_indices = torch.randperm(dataset_size, generator=generator)

    val_count = math.ceil(val_set_size * dataset_size)
    test_count = math.ceil(test_set_size * dataset_size)
    train_count = dataset_size - val_count - test_count

    # 3. Partition based on the computed splits
    train_indices = shuffled_indices[:train_count].tolist()
    test_indices = shuffled_indices[train_count : train_count + test_count].tolist()
    val_indices = shuffled_indices[train_count + test_count :].tolist()

    # Create the subsets based on the indices
    train_dataset = Subset(train_full_dataset, train_indices)
    test_dataset = Subset(test_val_full_dataset, test_indices)
    val_dataset = Subset(test_val_full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    # Batch for visualization
    vis_generator = torch.Generator().manual_seed(random_seed)

    fixed_train_batch = next(
        iter(DataLoader(train_dataset, batch_size=num_viz_samples, shuffle=True, generator=vis_generator))
    )
    # Since no shuffle, a generator is not needed
    fixed_test_batch = next(iter(DataLoader(test_dataset, batch_size=num_viz_samples, shuffle=False)))
    fixed_val_batch = next(iter(DataLoader(val_dataset, batch_size=num_viz_samples, shuffle=False)))

    # Batch (dimensione 1) for attention map visualization
    vis_generator.manual_seed(random_seed)
    fixed_train_attention_batch = next(
        iter(DataLoader(train_dataset, batch_size=1, shuffle=True, generator=vis_generator))
    )
    fixed_test_attention_batch = next(iter(DataLoader(test_dataset, batch_size=1, shuffle=False)))
    fixed_val_attention_batch = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=False)))

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'fixed_train_batch': fixed_train_batch,
        'fixed_val_batch': fixed_val_batch,
        'fixed_test_batch': fixed_test_batch,
        'fixed_train_attention_batch': fixed_train_attention_batch,
        'fixed_val_attention_batch': fixed_val_attention_batch,
        'fixed_test_attention_batch': fixed_test_attention_batch,
    }

# --- Ispezione degli Indici nei Set ---
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")

# Questo codice va eseguito DOPO aver chiamato create_training_setup
training_setup = create_training_setup(
    tokenizer=tokenizer,
    test_set_size=0.15,
    val_set_size=0.1,
    batch_size=16,
    num_workers=0,
    num_viz_samples=4,
    random_seed=42,
    train_augmentation_pipeline=None
)


print("\n--- Ispezione degli Indici nei Set ---")

# 1. Estrarre gli elenchi di indici dai dataset di tipo Subset
train_indices = training_setup['train_dataset'].indices
val_indices = training_setup['val_dataset'].indices

# Gli indici sono semplici liste/tensori di numeri interi.
# Nota: Poiché abbiamo mischiato gli indici, non saranno sequenziali (es. 0, 1, 2...).
# Saranno una permutazione casuale (es. 45, 892, 11, ...).

# 2. Stampare un campione degli indici per ogni set
print(f"\nNumero di indici nel Training Set: {len(train_indices)}")
print(f"Primi 10 indici del Training Set: {train_indices[:10]}")

print(f"\nNumero di indici nel Validation Set: {len(val_indices)}")
print(f"Primi 10 indici del Validation Set: {val_indices[:10]}")

# 3. VERIFICA DI COERENZA: Assicurarsi che i set siano disgiunti
# Questa è una verifica cruciale per assicurarsi che non ci sia "data leakage"
# tra i set. Non dovrebbero esserci indici in comune.

print("\n--- Verifica della Sovrapposizione degli Indici (Data Leakage Check) ---")

# Convertiamo gli indici in set per operazioni di intersezione efficienti
set_train = set(train_indices)
set_val = set(val_indices)

# Calcoliamo le intersezioni
train_val_overlap = set_train.intersection(set_val)

print(f"Sovrapposizione Train <-> Validation: {len(train_val_overlap)} indici comuni. Risultato: {'OK!' if not train_val_overlap else 'ERRORE!'}")

if not train_val_overlap:
    print("\n[SUCCESS] I set sono correttamente disgiunti. Nessun data leakage.")
else:
    print("\n[FAIL] Attenzione! C'è una sovrapposizione tra i set. La suddivisione non è corretta.")

# 4. VERIFICA FINALE: Assicurarsi che la somma degli indici corrisponda alla dimensione totale
total_unique_indices = len(set_train) + len(set_val)
original_dataset_size = len(PokemonDataset(tokenizer=tokenizer)) # Ricreiamo il dataset fittizio per avere la dimensione originale

print("\n--- Verifica Conteggio Totale ---")
print(f"Numero totale di indici unici nei tre set: {total_unique_indices}")
print(f"Dimensione del dataset originale:             {original_dataset_size}")
print(f"I totali corrispondono? {'Sì!' if total_unique_indices == original_dataset_size else 'No, qualcosa è andato storto.'}")
