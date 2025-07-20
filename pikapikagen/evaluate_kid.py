#!/usr/bin/env python3
"""
KID Evaluation Script for PikaPikaGen
Loads the trained model and computes Kernel Inception Distance on the validation set.
Uses torch-fidelity for KID computation.
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from model import Generator as PikaPikaGen
from data_loader import create_training_setup
from utils import denormalize_image
from torch_fidelity import calculate_metrics
from tqdm import tqdm
import argparse
import os
import tempfile
from PIL import Image

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "pikapikagen/model_checkpoints/checkpoint_epoch_150.pth"
TOKENIZER_NAME = "prajjwal1/bert-mini"

class PokemonKIDEvaluator:
    """Evaluator class for computing KID metrics on PikaPikaGen"""

    def __init__(self, checkpoint_path=CHECKPOINT_PATH, device=DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.checkpoint_path = checkpoint_path

        print(f"🎮 Initializing PikaPikaGen KID Evaluator on {device}")
        self._load_model()

    def _load_model(self):
        """Load the trained PikaPikaGen model (similar to gradio_demo.py)"""
        # Initialize model
        self.generator = PikaPikaGen().to(self.device)

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)

        # Load saved weights into model
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"✅ Generator loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")

        # Set to evaluation mode
        self.generator.eval()


    def _tensor_to_pil(self, tensor):
        """Convert normalized tensor [-1, 1] to PIL Image for torch-fidelity"""
        # Denormalize from [-1, 1] to [0, 1]
        denormalized = denormalize_image(tensor)
        # Convert to [0, 255] uint8 format
        uint8_tensor = (denormalized * 255).clamp(0, 255).to(torch.uint8)
        # Convert to numpy and then PIL
        img_np = uint8_tensor.permute(1, 2, 0).numpy()
        return Image.fromarray(img_np)

    def _save_images_to_temp_dir(self, images_tensor, prefix="images"):
        """Save tensor images to temporary directory for torch-fidelity"""
        temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_")

        for i, img_tensor in enumerate(images_tensor):
            pil_img = self._tensor_to_pil(img_tensor)
            img_path = os.path.join(temp_dir, f"{i:06d}.png")
            pil_img.save(img_path)

        return temp_dir

    def evaluate_kid(self, val_loader, max_batches=None, resolution="256x256",
                     subset_size=100, num_subsets=10):
        """
        Compute KID score between real validation images and generated images using torch-fidelity

        Args:
            val_loader: Validation data loader
            max_batches: Maximum number of batches to evaluate (None for all)
            resolution: Which resolution to evaluate ("256x256" or "64x64")
            subset_size: Number of samples in each subset for KID computation (used in torch-fidelity)
            num_subsets: Number of subsets for KID computation (used in torch-fidelity)

        Returns:
            dict: torch-fidelity metrics including KID
        """
        print(f"🧮 Computing KID score for {resolution} resolution using torch-fidelity...")

        all_real_images = []
        all_generated_images = []

        with torch.no_grad():
            batch_count = 0

            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing validation batches")):
                if max_batches and batch_idx >= max_batches:
                    break

                # Extract data from batch
                text_ids = batch["text"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                real_images = batch["image"]  # Shape: (B, 3, 256, 256)

                batch_size = text_ids.size(0)

                # Generate images using the model
                generated_256, generated_64 = self.generator(text_ids, attention_mask)

                # Select the appropriate resolution
                if resolution == "256x256":
                    generated_images = generated_256
                    # Real images are already 256x256
                    processed_real_images = real_images
                elif resolution == "64x64":
                    generated_images = generated_64
                    # Resize real images to 64x64 for fair comparison
                    processed_real_images = torch.nn.functional.interpolate(
                        real_images, size=(64, 64), mode='bilinear', align_corners=False
                    )
                else:
                    raise ValueError(f"Unsupported resolution: {resolution}")

                # Collect images for later processing
                all_real_images.append(processed_real_images.cpu())
                all_generated_images.append(generated_images.cpu())

                batch_count += 1

                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    total_images = (batch_idx + 1) * batch_size
                    print(f"  Processed {batch_idx + 1} batches, {total_images} images collected")

        # Concatenate all images
        all_real_images = torch.cat(all_real_images, dim=0)
        all_generated_images = torch.cat(all_generated_images, dim=0)

        total_images = all_real_images.size(0)
        print(f"📊 Collected {total_images} image pairs from {batch_count} batches")

        # Save images to temporary directories for torch-fidelity
        print("💾 Saving images to temporary directories...")
        real_temp_dir = self._save_images_to_temp_dir(all_real_images, "real")
        generated_temp_dir = self._save_images_to_temp_dir(all_generated_images, "generated")

        print(f"   Real images saved to: {real_temp_dir}")
        print(f"   Generated images saved to: {generated_temp_dir}")

        try:
            # Compute metrics using torch-fidelity
            print("🔄 Computing KID using torch-fidelity...")
            metrics = calculate_metrics(
                input1=generated_temp_dir,  # Generated (fake) images
                input2=real_temp_dir,       # Real images
                kid=True,                   # Calculate KID
                kid_subset_size=subset_size,
                kid_subsets=num_subsets,
                batch_size=64,              # Batch size for feature extraction
                device=self.device,
                verbose=True
            )

            return metrics

        finally:
            # Clean up temporary directories
            print("🧹 Cleaning up temporary directories...")
            import shutil
            shutil.rmtree(real_temp_dir)
            shutil.rmtree(generated_temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Evaluate KID score for PikaPikaGen')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH,
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to evaluate (None for all)')
    parser.add_argument('--resolution', type=str, choices=['256x256', '64x64'],
                        default='256x256', help='Resolution to evaluate')
    parser.add_argument('--subset-size', type=int, default=50,
                        help='Number of samples in each KID subset')
    parser.add_argument('--num-subsets', type=int, default=10,
                        help='Number of subsets for KID computation')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    # Initialize tokenizer for dataset loading
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Create training setup to get validation loader (same as in the notebook)
    print("📦 Setting up validation data loader...")
    training_setup = create_training_setup(
        tokenizer=tokenizer,
        train_val_split=0.9,
        batch_size=args.batch_size,
        num_workers=0,
        num_viz_samples=4,
        random_seed=42,
        train_augmentation_pipeline=None
    )

    val_loader = training_setup['val_loader']
    print(f"✅ Validation loader ready with {len(val_loader)} batches")

    # Initialize evaluator
    evaluator = PokemonKIDEvaluator(checkpoint_path=args.checkpoint)

    # Compute KID
    try:
        kid_mean, kid_std = evaluator.evaluate_kid(
            val_loader=val_loader,
            max_batches=args.max_batches,
            resolution=args.resolution,
            subset_size=args.subset_size,
            num_subsets=args.num_subsets
        )

        print("\n" + "="*50)
        print("🏆 KID EVALUATION RESULTS")
        print("="*50)
        print(f"Resolution: {args.resolution}")
        print(f"KID Mean: {kid_mean:.6f}")
        print(f"KID Std:  {kid_std:.6f}")
        print(f"Subset size: {args.subset_size}")
        print(f"Number of subsets: {args.num_subsets}")

        if args.max_batches:
            print(f"Evaluated on: {args.max_batches} batches")
        else:
            print("Evaluated on: Full validation set")

        print("="*50)

        # Save results to file
        results_file = f"kid_results_{args.resolution.replace('x', '_')}_{kid_mean:.6f}.txt"
        with open(results_file, 'w') as f:
            f.write(f"KID Evaluation Results\n")
            f.write(f"=====================\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Resolution: {args.resolution}\n")
            f.write(f"KID Mean: {kid_mean:.6f}\n")
            f.write(f"KID Std: {kid_std:.6f}\n")
            f.write(f"Subset size: {args.subset_size}\n")
            f.write(f"Number of subsets: {args.num_subsets}\n")
            if args.max_batches:
                f.write(f"Batches evaluated: {args.max_batches}\n")
            else:
                f.write(f"Batches evaluated: Full validation set\n")

        print(f"📝 Results saved to: {results_file}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
