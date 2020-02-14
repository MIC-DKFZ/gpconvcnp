import numpy as np
import torch
import gpytorch

from neuralprocess.data.base import FunctionGenerator



class WeaklyPeriodicKernel(gpytorch.kernels.Kernel):
    """Weakly periodic GP kernel as defined in the ConvCNP paper."""
    
    has_lengthscale = False

    def forward(self, x1, x2=None, **kwargs):
        """
        Compute a covariance matrix.

        Args:
            x1 (torch.tensor): First input, shape (B, N, C).
            x2 (torch.tensor): Second input, shape (B, M, C).

        Returns:
            torch.tensor: Covariance matrix, shape (B, N, M)
        
        """
        
        if x2 == None:
            x2 = x1
        
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(2)
        
        result = torch.pow(torch.cos(8*np.pi*x1) - torch.cos(8*np.pi*x2), 2).sum(-1)
        result += torch.pow(torch.sin(8*np.pi*x1) - torch.sin(8*np.pi*x2), 2).sum(-1)
        result = torch.exp(-0.5 * result)
        result *= torch.exp(-1/8. * torch.pow(x1 - x2, 2).sum(-1))
        
        return result



class GaussianProcessGenerator(FunctionGenerator):
    """
    Generate samples from a Gaussian Process prior.

    Args:
        batch_size (int): Batch size.
        kernel_type (type): A class that can construct a covariance matrix
            when an instance of it is called. Examples for this are
            GPyTorch kernels.
        kernel_kwargs (dict): Kernel arguments.

    """

    def __init__(self,
                 batch_size,
                 kernel_type,
                 kernel_kwargs,
                 *args, **kwargs):

        super().__init__(batch_size, *args, **kwargs)

        if kernel_kwargs is None:
            kernel_kwargs = dict()
        self.kernel = kernel_type(**kernel_kwargs)

    def apply(self, x):
        """
        Generate y values for input.

        Args:
            x (torch.tensor): x values, shape (B, N, 1)

        Returns:
            torch.tensor: y values, shape (B, N, 1)

        """

        covar = self.kernel(x, x)
        mean = torch.zeros(*covar.shape[:2], dtype=covar.dtype)
        dist = gpytorch.distributions.MultivariateNormal(mean, covar)
        return dist.sample().unsqueeze(-1)