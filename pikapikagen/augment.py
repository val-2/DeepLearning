import torch
import torchvision.transforms as T

class GeometricAugmentPipe:
    """
    Una pipeline di augmentation "gentile" che applica solo
    trasformazioni geometriche. Non eredita da nn.Module per evitare
    ogni possibile problema con il grafo computazionale.
    """
    def __init__(self, p=0.5):
        self.p = p
        self.transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=1),
        ])

    def __call__(self, images):
        if self.p > 0 and torch.rand(1).item() < self.p:
            return self.transforms(images)
        return images
