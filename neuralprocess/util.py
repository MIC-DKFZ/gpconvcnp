import torch



def tensor_to_loc_scale(tensor, distribution, logvar_transform=True, axis=1):
    """
    Split tensor into two and construct loc-scale distribution from it.

    Args:
        tensor (torch.tensor): Shape (B, 2*C, ...).
        distribution (type): A subclass of torch.distributions.Distribution.
        logvar_transform (bool): Apply x -> exp(0.5*x) to scale.
        axis (int): Split along this axis.

    Returns:
        torch.distributions.Distribution: A loc-scale distribution.

    """

    if tensor.shape[axis] % 2 != 0:
        raise IndexError("Axis {} of 'tensor' must be divisible by 2.".format(axis))

    loc, scale = torch.split(tensor, tensor.shape[1]//2, axis)
    if logvar_transform:
        scale = torch.exp(0.5 * scale)

    return distribution(loc, scale)