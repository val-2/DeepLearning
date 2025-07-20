import torch
from transformers import AutoTokenizer
from model import Generator as PikaPikaGen
from data_loader import create_training_setup
from utils import denormalize_image
from torch_fidelity import calculate_metrics
import os
import tempfile
from PIL import Image
import shutil

CHECKPOINT_PATH = "pikapikagen/model_checkpoint/checkpoint_epoch_150.pth"

TOKENIZER_NAME = "prajjwal1/bert-mini"

BATCH_SIZE = 16  # Batch size for generating images
NUM_WORKERS = 2  # Number of workers for the data loader

KID_SUBSET_SIZE = 50
KID_NUM_SUBSETS = 20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PokemonKIDEvaluator:
    """Evaluator class for computing KID metrics on PikaPikaGen."""

    def __init__(self, checkpoint_path, device=DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.checkpoint_path = checkpoint_path

        self._load_model()  # As in gradio demo

    def _load_model(self):
        self.generator = PikaPikaGen().to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        denormalized = denormalize_image(tensor)
        uint8_tensor = (denormalized * 255).clamp(0, 255).to(torch.uint8)
        img_np = uint8_tensor.cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(img_np)

    def _save_images_to_temp_dir(self, images_tensor: torch.Tensor, prefix: str) -> str:
        """Save a batch of image tensors to a new temporary directory."""
        temp_dir = tempfile.mkdtemp(prefix=f"pikakid_{prefix}_")
        for i, img_tensor in enumerate(images_tensor):
            pil_img = self._tensor_to_pil(img_tensor)
            img_path = os.path.join(temp_dir, f"{i:06d}.png")
            pil_img.save(img_path)
        return temp_dir

    def evaluate_kid(self, test_loader, resolution="256x256"):

        all_real_images = []
        all_generated_images = []

        with torch.no_grad():
            for batch in test_loader:
                text_ids = batch["text"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                real_images_256 = batch["image"]  # (B, 3, 256, 256)

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

        # Save images to temporary directories for torch-fidelity
        real_temp_dir = self._save_images_to_temp_dir(all_real_images, "real")
        generated_temp_dir = self._save_images_to_temp_dir(all_generated_images, "generated")

        metrics = calculate_metrics(
            input1=generated_temp_dir,      # Path to generated (fake) images
            input2=real_temp_dir,           # Path to real images
            kid=True,
            kid_subset_size=KID_SUBSET_SIZE,
            kid_subsets=KID_NUM_SUBSETS,
            batch_size=BATCH_SIZE,
            device=self.device
        )

        kid_mean = metrics['kernel_inception_distance_mean']
        kid_std = metrics['kernel_inception_distance_std']

        # Clean up the temporary directories
        shutil.rmtree(real_temp_dir)
        shutil.rmtree(generated_temp_dir)

        return kid_mean, kid_std

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    training_setup = create_training_setup(
        tokenizer=tokenizer,
        test_set_size=0.2,
        val_set_size=0.1,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        random_seed=42, # Use a fixed seed for a reproducible split
    )
    test_loader = training_setup['test_loader']
    test_set_size = len(test_loader.dataset)

    evaluator = PokemonKIDEvaluator(checkpoint_path=CHECKPOINT_PATH)

    resolutions_to_test = ['64x64', '256x256']

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test samples: {test_set_size}")
    print(f"KID Subset Size: {KID_SUBSET_SIZE}")
    print(f"KID Subsets: {KID_NUM_SUBSETS}")

    for res in resolutions_to_test:
        kid_mean, kid_std = evaluator.evaluate_kid(test_loader, resolution=res)
        print(f"Resolution {res}:\t KID = {kid_mean:.6f} ± {kid_std:.6f}")


if __name__ == "__main__":
    main()
