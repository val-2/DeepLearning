from torch.utils.data import DataLoader, Subset
import torch
from dataset import PokemonDataset
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
    assert 0 <= test_set_size < 1.0, "test_set_size must be a float between 0 and 1"
    assert 0 <= val_set_size < 1.0, "val_set_size must be a float between 0 and 1"
    assert (test_set_size + val_set_size) < 1.0, "The sum of test and validation sizes must be less than 1"

    train_full_dataset = PokemonDataset(tokenizer=tokenizer, augmentation_transforms=train_augmentation_pipeline)
    # Don't use augmentation for test and validation
    test_val_full_dataset = PokemonDataset(tokenizer=tokenizer)

    dataset_size = len(train_full_dataset)

    # Create a random reproducible permutation
    generator = torch.Generator().manual_seed(random_seed)
    shuffled_indices = torch.randperm(dataset_size, generator=generator)

    val_count = math.ceil(val_set_size * dataset_size)
    test_count = math.ceil(test_set_size * dataset_size)
    train_count = dataset_size - val_count - test_count

    # Partition based on the computed splits
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
