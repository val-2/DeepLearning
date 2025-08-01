def denormalize_image(tensor):
    """
    Denormalizes an image tensor from the range [-1, 1] to [0, 1] for visualization.

    Args:
        tensor (torch.Tensor): The image tensor, with values in [-1, 1].

    Returns:
        torch.Tensor: The denormalized tensor with values in [0, 1].
    """
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1)
