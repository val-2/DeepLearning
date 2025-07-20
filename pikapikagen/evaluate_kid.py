#!/usr/bin/env python3
"""
KID Evaluation Script for PikaPikaGen (Refactored)

This script loads a trained PikaPikaGen model and computes the Kernel Inception
Distance (KID) on the validation set for both 64x64 and 256x256 resolutions.

It is self-contained and does not require command-line arguments. All
configuration is done via the constants defined below.

Dependencies:
- torch, transformers, numpy, Pillow
- torch-fidelity
- Project files: model.py, data_loader.py, utils.py
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from model import Generator as PikaPikaGen
from data_loader import create_training_setup
from utils import denormalize_image
from torch_fidelity import calculate_metrics
from tqdm import tqdm
import os
import tempfile
from PIL import Image
import shutil
import time

# --- SCRIPT CONFIGURATION ---
# Path to the model checkpoint to evaluate
CHECKPOINT_PATH = "pikapikagen/model_checkpoint/checkpoint_epoch_150.pth"

# Model and tokenizer configuration
TOKENIZER_NAME = "prajjwal1/bert-mini"

# Data loader configuration
BATCH_SIZE = 16  # Batch size for generating images
NUM_WORKERS = 2  # Number of workers for the data loader

# KID evaluation parameters.
# With a validation set of 89 images, we sample 50 images 20 times (with replacement)
# to get a stable estimate of the KID score.
KID_SUBSET_SIZE = 50
KID_NUM_SUBSETS = 20

# --- DO NOT EDIT BELOW THIS LINE ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PokemonKIDEvaluator:
    """Evaluator class for computing KID metrics on PikaPikaGen."""

    def __init__(self, checkpoint_path, device=DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.checkpoint_path = checkpoint_path

        print(f"🎮 Initializing PikaPikaGen KID Evaluator on {device}")
        self._load_model()

    def _load_model(self):
        """Load the trained PikaPikaGen model from a checkpoint."""
        self.generator = PikaPikaGen().to(self.device)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"❌ Checkpoint file not found at: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        print(f"✅ Generator loaded from checkpoint (epoch {checkpoint.get('epoch', 'N/A')})")

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert a single normalized [-1, 1] image tensor to a PIL Image."""
        denormalized = denormalize_image(tensor)
        uint8_tensor = (denormalized * 255).clamp(0, 255).to(torch.uint8)
        img_np = uint8_tensor.cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(img_np)

    def _save_images_to_temp_dir(self, images_tensor: torch.Tensor, prefix: str) -> str:
        """Save a batch of image tensors to a new temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix=f"pikakid_{prefix}_")
        for i, img_tensor in enumerate(tqdm(images_tensor, desc=f"💾 Saving {prefix} images")):
            pil_img = self._tensor_to_pil(img_tensor)
            img_path = os.path.join(temp_dir, f"{i:06d}.png")
            pil_img.save(img_path)
        return temp_dir

    def evaluate_kid(self, val_loader, resolution="256x256"):
        """
        Compute KID score between real validation images and generated images.

        Args:
            val_loader: The validation data loader.
            resolution (str): The resolution to evaluate ('64x64' or '256x256').

        Returns:
            A tuple containing (kid_mean, kid_std).
        """
        print(f"\n🧮 Starting KID evaluation for {resolution} resolution...")

        all_real_images = []
        all_generated_images = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"🎨 Generating {resolution} images"):
                text_ids = batch["text"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                real_images_256 = batch["image"]  # Always (B, 3, 256, 256)

                # Generate images from text prompts
                generated_256, generated_64 = self.generator(text_ids, attention_mask)

                # Select the correct resolution for both real and generated images
                if resolution == "256x256":
                    generated_images = generated_256
                    processed_real_images = real_images_256
                elif resolution == "64x64":
                    generated_images = generated_64
                    processed_real_images = torch.nn.functional.interpolate(
                        real_images_256, size=(64, 64), mode='bilinear', align_corners=False
                    )
                else:
                    raise ValueError(f"Unsupported resolution: {resolution}")

                all_real_images.append(processed_real_images.cpu())
                all_generated_images.append(generated_images.cpu())

        # Combine all batches into single tensors
        all_real_images = torch.cat(all_real_images, dim=0)
        all_generated_images = torch.cat(all_generated_images, dim=0)

        total_images = len(all_real_images)
        print(f"📊 Collected {total_images} real and {len(all_generated_images)} generated images.")

        # Save images to temporary directories for torch-fidelity
        real_temp_dir = self._save_images_to_temp_dir(all_real_images, "real")
        generated_temp_dir = self._save_images_to_temp_dir(all_generated_images, "generated")
        print(f"   - Real images saved to: {real_temp_dir}")
        print(f"   - Generated images saved to: {generated_temp_dir}")

        # Compute KID using torch-fidelity
        print("🔄 Computing KID score with torch-fidelity. This may take a while...")
        metrics = calculate_metrics(
            input1=generated_temp_dir,      # Path to generated (fake) images
            input2=real_temp_dir,           # Path to real images
            kid=True,
            kid_subset_size=KID_SUBSET_SIZE,
            kid_subsets=KID_NUM_SUBSETS,
            batch_size=BATCH_SIZE,
            device=self.device,
            verbose=False                   # Set to True for detailed torch-fidelity logs
        )

        kid_mean = metrics['kernel_inception_distance_mean']
        kid_std = metrics['kernel_inception_distance_std']

        # IMPORTANT: Clean up the temporary directories to free up disk space
        print("🧹 Cleaning up temporary image directories...")
        shutil.rmtree(real_temp_dir)
        shutil.rmtree(generated_temp_dir)

        return kid_mean, kid_std

def main():
    """Main function to run the evaluation."""
    start_time = time.time()

    # 1. Set up validation data loader
    print("📦 Setting up validation data loader...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    training_setup = create_training_setup(
        tokenizer=tokenizer,
        train_val_split=0.9,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        random_seed=42, # Use a fixed seed for a reproducible validation set
    )
    val_loader = training_setup['val_loader']
    val_set_size = len(val_loader.dataset)
    print(f"✅ Validation loader ready with {val_set_size} samples across {len(val_loader)} batches.")

    # 2. Initialize the evaluator with the model
    evaluator = PokemonKIDEvaluator(checkpoint_path=CHECKPOINT_PATH)

    # 3. Run evaluation for specified resolutions
    results = {}
    resolutions_to_test = ['64x64', '256x256']

    for res in resolutions_to_test:
        kid_mean, kid_std = evaluator.evaluate_kid(val_loader, resolution=res)
        results[res] = (kid_mean, kid_std)
        print(f"⭐ Result for {res}: KID = {kid_mean:.4f} ± {kid_std:.4f}")

    # 4. Print and save final summary
    print("\n" + "="*50)
    print("🏆 FINAL KID EVALUATION RESULTS 🏆")
    print("="*50)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Validation Samples: {val_set_size}")
    print(f"KID Subset Size: {KID_SUBSET_SIZE}")
    print(f"KID Subsets: {KID_NUM_SUBSETS}")
    print("-" * 50)
    for res, (mean, std) in results.items():
        print(f"Resolution {res}:\t KID = {mean:.6f} ± {std:.6f}")
    print("="*50)

    # Save results to a file
    results_file = "kid_evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("KID Evaluation Results\n")
        f.write("======================\n")
        f.write(f"Timestamp: {time.ctime(start_time)}\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Validation Samples: {val_set_size}\n\n")
        f.write(f"KID Parameters:\n")
        f.write(f"  - Subset Size: {KID_SUBSET_SIZE}\n")
        f.write(f"  - Number of Subsets: {KID_NUM_SUBSETS}\n\n")
        f.write("Results:\n")
        for res, (mean, std) in results.items():
            f.write(f"  - Resolution {res}: KID = {mean:.6f} ± {std:.6f}\n")

    total_time = time.time() - start_time
    print(f"📝 Results saved to: {results_file}")
    print(f"🕒 Total evaluation time: {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
