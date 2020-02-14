import torch



def tensor_to_loc_scale(tensor, distribution, logvar_transform=True, axis=1):
    """
    Split tensor into two and construct loc-scale distribution from it.

    Args:
        tensor (torch.tensor): Shape (..., 2*C, ...).
        distribution (type): A subclass of torch.distributions.Distribution.
        logvar_transform (bool): Apply x -> exp(0.5*x) to scale.
        axis (int): Split along this axis.

    Returns:
        torch.distributions.Distribution: A loc-scale distribution.

    """

    if tensor.shape[axis] % 2 != 0:
        raise IndexError("Axis {} of 'tensor' must be divisible by 2.".format(axis))

    loc, scale = torch.split(tensor, tensor.shape[axis]//2, axis)
    if logvar_transform:
        scale = torch.exp(0.5 * scale)

    return distribution(loc, scale)



def stack_batch(tensor):
    """Stacks first axis along second axis."""

    return tensor.reshape(tensor.shape[0]*tensor.shape[1], *tensor.shape[2:])



def unstack_batch(tensor, N):
    """Reverses stack_batch."""

    B = tensor.shape[0] // N
    return tensor.reshape(N, B, *tensor.shape[1:])



def make_grid(x, points_per_unit, padding=0.1, grid_divisible_by=None):
    """
    Make a grid for an input. The input can have multiple channels,
    but we use the same grid for all channels and just broadcast it to
    all channels. This means all input channels should have roughly the
    same range.

    Args:
        x (torch.tensor): Input values, shape (N, B, Cin). Can alternatively
            be a list or tuple of tensors, then the min/max will be taken
            over all tensors.
        points_per_unit (int): The grid resolution.
        padding (float): Pad the grid range on both sides by this value.
        grid_divisible_by (int): Increase grid size until it's divisible
            by this number.

    Returns:
        torch.tensor: The grid

    """

    if torch.is_tensor(x):
        min_ = x.min().item()
        max_ = x.max().item()
    else:
        min_ = 1e9
        max_ = -1e9
        for t in x:
            min_ = min(min_, t.min().item())
            max_ = max(max_, t.max().item())
    min_ -= padding
    max_ += padding

    if not torch.is_tensor(x):
        x = x[0]

    num_points = int(points_per_unit * (max_ - min_))
    if grid_divisible_by not in (None, 0):
        while num_points % grid_divisible_by != 0:
            num_points += 1
    grid = torch.linspace(min_, max_, num_points).reshape(-1, 1, 1)
    grid = grid.repeat(1, *x.shape[1:])
    grid = grid.to(dtype=x.dtype, device=x.device)

    return grid