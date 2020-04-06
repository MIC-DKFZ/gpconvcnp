import torch
import numpy as np



def set_seeds(seed, cuda=True):
    """
    Set all seeds in numpy and torch.

    Args:
        seed (int): Set this seed. Can also be a list or tuple of 3 ints.
        cuda (bool): Also set CUDA seed in torch.

    """

    if not hasattr(seed, "__iter__"):
        seed = (seed, seed, seed)
    np.random.seed(seed[0])
    torch.manual_seed(seed[1])
    if cuda: torch.cuda.manual_seed_all(seed[2])



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



def unstack_batch(tensor, B):
    """Reverses stack_batch."""

    N = tensor.shape[0] // B
    return tensor.reshape(B, N, *tensor.shape[1:])



def make_grid(x, points_per_unit, padding=0.1, grid_divisible_by=None):
    """
    Make a grid for an input. The input can have multiple channels,
    but we use the same grid for all channels and just broadcast it to
    all channels. This means all input channels should have roughly the
    same range.

    Args:
        x (torch.tensor): Input values, shape (B, N, Cin). Can alternatively
            be a list or tuple of tensors, then the min/max will be taken
            over all tensors.
        points_per_unit (int): The grid resolution.
        padding (float): Pad the grid range on both sides by this value.
        grid_divisible_by (int): Increase grid size until it's divisible
            by this number.

    Returns:
        torch.tensor: The grid, shape (B, G, Cin)

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
        num_points += grid_divisible_by - num_points % grid_divisible_by
    grid = torch.linspace(min_, max_, num_points).reshape(1, -1, 1)
    grid = grid.repeat(x.shape[0], 1, x.shape[2])
    grid = grid.to(dtype=x.dtype, device=x.device)

    return grid



def match_shapes(*args, ignore_axes=None):
    """
    Expand multiple tensors to have matching shapes.
    If a tensor has fewer dimensions than others, new axes will be added at the end.

    Args:
        *args (torch.tensor): Any number of tensors.
        ignore_axes (iterable): Can be None, int or collection of ints.
            These axes won't be modified.

    Returns:
        tuple: Modified input tensors.

    """

    args = list(filter(lambda x: x is not None, args))

    dims = [a.ndim for a in args]
    target_dim = max(dims)
    for a, arr in enumerate(args):
        while arr.ndim < target_dim:
            arr = arr.unsqueeze(-1)
        args[a] = arr

    shapes = np.array([np.array(a.shape) for a in args])
    target_shape = np.max(shapes, axis=0)

    for a, arr in enumerate(args):
        target_shape_a = target_shape.copy()
        if ignore_axes is not None:
            if isinstance(ignore_axes, int):
                target_shape_a[ignore_axes] = arr.shape[ignore_axes]
            elif len(ignore_axes) > 0:
                for ax in ignore_axes:
                    target_shape_a[ax] = arr.shape[ax]
        args[a] = arr.expand(*target_shape_a)

    return tuple(args)