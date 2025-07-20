from torch.utils.data import DataLoader
import torch
from dataset import PokemonDataset

def create_training_setup(tokenizer, train_val_split, batch_size, num_workers=0,
                         num_viz_samples=4, random_seed=42, train_augmentation_pipeline=None):
    """
    Crea un setup completo per il training con dataset, dataloader e batch fissi per visualizzazione.
    Enhanced with modular augmentation pipeline support
    """
    from torch.utils.data import random_split, TensorDataset, Subset

    # --- Creazione dei Dataset ---
    # Crea un'istanza per il training (con augmentazione) e la validazione (senza augmentazione)
    train_full_dataset = PokemonDataset(tokenizer=tokenizer, augmentation_transforms=train_augmentation_pipeline)
    val_full_dataset = PokemonDataset(tokenizer=tokenizer)  # No augmentation for validation

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
