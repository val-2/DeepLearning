from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class CLIPLoss(nn.Module):
    """
    CLIP (Contrastive Language-Image Pre-training) Loss implementation using SentenceTransformer.

    This loss function computes the contrastive loss between image and text embeddings,
    encouraging the model to generate images that are semantically aligned with their
    corresponding text descriptions.

    The loss is computed as the sum of:
    1. Image-to-text contrastive loss (how well generated images match their text descriptions)
    2. Text-to-image contrastive loss (symmetric loss for better alignment)
    """

    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        temperature: float = 0.07,
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize CLIP loss.

        Args:
            model_name: Name of the SentenceTransformer model to use for encoding
            temperature: Temperature parameter for scaling the logits
            device: Device to run the model on (auto-detected if None)
            normalize_embeddings: Whether to L2 normalize embeddings before computing similarity
        """
        super(CLIPLoss, self).__init__()

        # Initialize the sentence transformer model
        self.model = SentenceTransformer(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.device = device

        # Temperature parameter for contrastive learning
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings

        # Image preprocessing for the CLIP model
        # Images should be in range [0, 1] and normalized with ImageNet stats
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # CLIP typically uses 224x224 images
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for CLIP encoding.

        Args:
            images: Tensor of shape [B, C, H, W] in range [-1, 1] or [0, 1]

        Returns:
            Preprocessed images ready for CLIP encoding
        """
        # Convert from [-1, 1] to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        # Apply CLIP preprocessing
        batch_size = images.shape[0]
        processed_images = []

        for i in range(batch_size):
            img = images[i]
            img = self.image_transform(img)
            processed_images.append(img)

        return torch.stack(processed_images, dim=0)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using the CLIP model.

        Args:
            images: Tensor of shape [B, C, H, W]

        Returns:
            Image embeddings of shape [B, embedding_dim]
        """
        # Preprocess images
        processed_images = self.preprocess_images(images)

        # Convert to PIL Images for SentenceTransformer
        pil_images = []
        for img in processed_images:
            # Denormalize for PIL conversion
            img_denorm = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img.device) + \
                        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img.device)
            img_denorm = torch.clamp(img_denorm, 0, 1)

            # Convert to numpy and then PIL
            img_np = img_denorm.cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)

        # Encode images
        with torch.no_grad():
            image_embeddings = self.model.encode(
                pil_images,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )

        return image_embeddings

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text descriptions using the CLIP model.

        Args:
            texts: List of text descriptions

        Returns:
            Text embeddings of shape [B, embedding_dim]
        """
        with torch.no_grad():
            text_embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )

        return text_embeddings

    def compute_contrastive_loss(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional contrastive loss between image and text embeddings.

        Args:
            image_embeddings: Tensor of shape [B, embedding_dim]
            text_embeddings: Tensor of shape [B, embedding_dim]

        Returns:
            Tuple of (image_to_text_loss, text_to_image_loss)
        """
        batch_size = image_embeddings.shape[0]

        # Normalize embeddings if specified
        if self.normalize_embeddings:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        # Compute similarity matrix
        # Shape: [B, B] where entry (i,j) is similarity between image i and text j
        similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature

        # Create labels for contrastive loss (diagonal should be positive pairs)
        labels = torch.arange(batch_size, device=self.device)

        # Image-to-text loss: for each image, text at same index should have highest similarity
        image_to_text_loss = F.cross_entropy(similarity_matrix, labels)

        # Text-to-image loss: for each text, image at same index should have highest similarity
        text_to_image_loss = F.cross_entropy(similarity_matrix.t(), labels)

        return image_to_text_loss, text_to_image_loss

    def forward(
        self,
        generated_images: torch.Tensor,
        text_descriptions: Union[List[str], torch.Tensor],
        return_similarities: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute CLIP loss between generated images and text descriptions.

        Args:
            generated_images: Generated images tensor of shape [B, C, H, W]
            text_descriptions: List of text descriptions or tokenized text tensor
            return_similarities: Whether to return similarity scores along with loss

        Returns:
            CLIP loss (scalar tensor) or (loss, similarity_matrix) if return_similarities=True
        """
        # Handle text input
        if isinstance(text_descriptions, torch.Tensor):
            # If tokenized text is provided, you'll need to decode it first
            # This is a placeholder - you'd need to implement text decoding based on your tokenizer
            raise NotImplementedError("Tokenized text input not yet supported. Please provide text as List[str]")

        # Encode images and texts
        image_embeddings = self.encode_images(generated_images)
        text_embeddings = self.encode_texts(text_descriptions)

        # Compute contrastive losses
        img_to_text_loss, text_to_img_loss = self.compute_contrastive_loss(
            image_embeddings, text_embeddings
        )

        # Total CLIP loss is the sum of both directions
        total_loss = (img_to_text_loss + text_to_img_loss) / 2.0

        if return_similarities:
            # Compute similarity matrix for analysis
            if self.normalize_embeddings:
                image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
                text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
            return total_loss, similarity_matrix

        return total_loss


class CLIPLossWithFeatures(CLIPLoss):
    """
    Extended CLIP loss that can also work with intermediate feature representations
    instead of just final images. Useful for training at multiple scales or with
    feature-level supervision.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add a feature projection layer to map arbitrary feature dimensions
        # to the CLIP embedding dimension (typically 512 for ViT-B/32)
        self.feature_projector = None

    def set_feature_projector(self, input_dim: int, output_dim: int = 512):
        """
        Set up a feature projector to map feature representations to CLIP embedding space.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of CLIP embeddings (usually 512)
        """
        self.feature_projector = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        ).to(self.device)

    def forward_features(
        self,
        image_features: torch.Tensor,
        text_descriptions: List[str],
        return_similarities: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute CLIP loss using image features instead of raw images.

        Args:
            image_features: Image feature representations [B, feature_dim]
            text_descriptions: List of text descriptions
            return_similarities: Whether to return similarity scores

        Returns:
            CLIP loss or (loss, similarities)
        """
        if self.feature_projector is None:
            raise ValueError("Feature projector not set. Call set_feature_projector() first.")

        # Project features to CLIP embedding space
        projected_features = self.feature_projector(image_features)

        # Encode texts
        text_embeddings = self.encode_texts(text_descriptions)

        # Compute contrastive losses
        img_to_text_loss, text_to_img_loss = self.compute_contrastive_loss(
            projected_features, text_embeddings
        )

        total_loss = (img_to_text_loss + text_to_img_loss) / 2.0

        if return_similarities:
            if self.normalize_embeddings:
                projected_features = F.normalize(projected_features, p=2, dim=1)
                text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            similarity_matrix = torch.matmul(projected_features, text_embeddings.t())
            return total_loss, similarity_matrix

        return total_loss


# Example usage and utility functions
def create_clip_loss(
    model_name: str = "clip-ViT-B-32",
    temperature: float = 0.07,
    device: Optional[str] = None
) -> CLIPLoss:
    """
    Factory function to create a CLIP loss instance with common defaults.

    Args:
        model_name: SentenceTransformer model name
        temperature: Temperature for contrastive learning
        device: Device to use

    Returns:
        Configured CLIPLoss instance
    """
    return CLIPLoss(
        model_name=model_name,
        temperature=temperature,
        device=device,
        normalize_embeddings=True
    )


def test_clip_loss():
    """
    Simple test function to verify CLIP loss functionality.
    """
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256)  # Random images
    texts = [
        "a red pokemon with fire abilities",
        "a blue water type pokemon",
        "a green grass pokemon in a forest",
        "a yellow electric pokemon with lightning"
    ]

    # Create CLIP loss
    clip_loss = create_clip_loss()

    # Compute loss
    try:
        loss = clip_loss(images, texts)
        print(f"CLIP Loss computed successfully: {loss.item():.4f}")

        # Test with similarity return
        loss, similarities = clip_loss(images, texts, return_similarities=True)
        print(f"Similarity matrix shape: {similarities.shape}")
        print(f"Diagonal similarities (should be highest): {torch.diag(similarities)}")

    except Exception as e:
        print(f"Error computing CLIP loss: {e}")


if __name__ == "__main__":
    test_clip_loss()
