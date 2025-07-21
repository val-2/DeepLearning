def denormalize_image(tensor):
    """
    Denormalizza un tensore immagine dall'intervallo [-1, 1] a [0, 1] per la visualizzazione.

    Args:
        tensor (torch.Tensor): Il tensore dell'immagine, con valori in [-1, 1].

    Returns:
        torch.Tensor: Il tensore denormalizzato con valori in [0, 1].
    """
    tensor = (tensor + 1) / 2
    return tensor.clamp(0, 1)
