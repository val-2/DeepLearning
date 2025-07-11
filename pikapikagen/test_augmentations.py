import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from augment import AugmentPipe

def denormalize_image(tensor):
    """
    Denormalizes an image tensor from the [-1, 1] range to [0, 1] for visualization.
    """
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1)

def main():
    """
    Generates a grid of all available augmentations and saves it to a file.
    """
    print("--- Testing Augmentation Pipeline ---")

    # --- Setup ---
    # Use a sample image from the dataset
    # This requires the dataset to be downloaded.
    # If you run this script from the root of the project, the path should be correct.
    try:
        image_path = Path("dataset/pokedex-main/images/small_images/025.png") # Pikachu!
        image_rgba = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Sample image not found at {image_path}.")
        print("Please ensure the dataset is downloaded and you are running this script from the project root.")
        return

    # Convert to RGB with a white background and resize, similar to the dataset loader
    background = Image.new('RGB', image_rgba.size, (255, 255, 255))
    background.paste(image_rgba, mask=image_rgba.split()[-1])
    
    # Basic transforms to get it into the right tensor format [-1, 1]
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Unsqueeze to create a batch of 1
    base_image_tensor = transform(background).unsqueeze(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_image_tensor = base_image_tensor.to(device)

    # --- Define Augmentations to Test ---
    # We create a new pipe for each to isolate its effect
    augmentation_configs = {
        'xflip':      {'xflip': 1, 'rotate': 0, 'scale': 0, 'translate': 0, 'cutout': 0, 'brightness': 0, 'contrast': 0, 'saturation': 0},
        'rotate':     {'xflip': 0, 'rotate': 1, 'scale': 0, 'translate': 0, 'cutout': 0, 'brightness': 0, 'contrast': 0, 'saturation': 0},
        'scale':      {'xflip': 0, 'rotate': 0, 'scale': 1, 'translate': 0, 'cutout': 0, 'brightness': 0, 'contrast': 0, 'saturation': 0},
        'translate':  {'xflip': 0, 'rotate': 0, 'scale': 0, 'translate': 1, 'cutout': 0, 'brightness': 0, 'contrast': 0, 'saturation': 0},
        'cutout':     {'xflip': 0, 'rotate': 0, 'scale': 0, 'translate': 0, 'cutout': 1, 'brightness': 0, 'contrast': 0, 'saturation': 0},
        'brightness': {'xflip': 0, 'rotate': 0, 'scale': 0, 'translate': 0, 'cutout': 0, 'brightness': 1, 'contrast': 0, 'saturation': 0},
        'contrast':   {'xflip': 0, 'rotate': 0, 'scale': 0, 'translate': 0, 'cutout': 0, 'brightness': 0, 'contrast': 1, 'saturation': 0},
        'saturation': {'xflip': 0, 'rotate': 0, 'scale': 0, 'translate': 0, 'cutout': 0, 'brightness': 0, 'contrast': 0, 'saturation': 1},
    }

    augmented_images = {}
    for name, kwargs in augmentation_configs.items():
        print(f"Applying: {name}")
        # p=1.0 ensures the augmentation pipeline is always applied
        pipe = AugmentPipe(p=1.0, **kwargs).to(device)
        with torch.no_grad():
            # Run it a few times to see different random outcomes
            augmented_images[name] = [pipe(base_image_tensor) for _ in range(3)]

    # --- Visualization ---
    num_augs = len(augmented_images)
    num_cols = 4 # Original + 3 examples
    
    fig, axs = plt.subplots(num_augs, num_cols, figsize=(num_cols * 3, num_augs * 3))
    fig.suptitle("AugmentPipe Visualization", fontsize=24)

    for i, (name, img_list) in enumerate(augmented_images.items()):
        # Set title for the row
        axs[i, 0].set_ylabel(name.capitalize(), fontsize=14, rotation=90, labelpad=20)
        
        # Show original image in the first column
        ax_orig = axs[i, 0]
        orig_img_display = denormalize_image(base_image_tensor.squeeze(0).cpu()).permute(1, 2, 0)
        ax_orig.imshow(orig_img_display)
        ax_orig.set_title("Original")
        ax_orig.axis('off')

        # Show augmented examples
        for j, aug_tensor in enumerate(img_list):
            ax_aug = axs[i, j + 1]
            aug_img_display = denormalize_image(aug_tensor.squeeze(0).cpu()).permute(1, 2, 0)
            ax_aug.imshow(aug_img_display)
            ax_aug.set_title(f"Example {j+1}")
            ax_aug.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = "augmentation_test_grid.png"
    plt.savefig(save_path)
    print(f"\nSaved visualization to: {save_path}")

if __name__ == "__main__":
    main() 