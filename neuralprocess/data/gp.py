import numpy as np
import torch
from sklearn.gaussian_process.kernels import (
    Hyperparameter,
    Kernel as KernelSklearn,
    StationaryKernelMixin,
    NormalizedKernelMixin,
)

from neuralprocess.data.base import FunctionGenerator


class Kernel:
    """
    A simple base class for kernels that exposes construction arguments
    as attributes.
    """

    def __init__(self, **kwargs):

        self._params = dict()
        self.params = kwargs

    @property
    def params(self):

        return self._params.copy()

    @params.setter
    def params(self, kwargs):

        for key, val in kwargs.items():
            setattr(self, key, val)
        self._params.update(kwargs)

    def __setattr__(self, key, val):

        super().__setattr__(key, val)
        if key not in ("params", "_params"):
            self._params.update({key: val})

    def __call__(self, a, b):

        raise NotImplementedError


class GaussianKernel(Kernel):
    def __init__(self, lengthscale=0.5, amplitude=1.0):

        super().__init__(lengthscale=lengthscale, amplitude=amplitude)

    def __call__(self, a, b):
        """a, b assumed to be of shape (N,1)"""

        sqdist = (a - b.T) ** 2
        l = self.lengthscale ** 2
        return self.amplitude * np.exp(-0.5 / l * sqdist)


class PeriodicKernel(Kernel):
    def __init__(self, lengthscale=0.5, amplitude=1.0, periodicity=1.0):

        super().__init__(
            lengthscale=lengthscale, amplitude=amplitude, periodicity=periodicity
        )

    def __call__(self, a, b):
        """a, b assumed to be of shape (N,1)"""

        dist = np.abs(a - b.T)
        result = np.power(np.sin(np.pi * dist / self.periodicity), 2)
        l = self.lengthscale ** 2
        result = self.amplitude * np.exp(-2 / l * result)
        return result


class WeaklyPeriodicKernel(Kernel):
    """The weakly periodic kernel that was given in the ConvCNP paper."""

    def __init__(self, *args, **kwargs):

        super().__init__()

    def __call__(self, a, b):
        """a, b assumed to be of shape (N,1)"""

        d1 = np.power(a - b.T, 2)
        d2 = np.power(np.cos(8 * np.pi * a) - np.cos(8 * np.pi * b.T), 2)
        d3 = np.power(np.sin(8 * np.pi * a) - np.sin(8 * np.pi * b.T), 2)
        return np.exp(-1 / 8.0 * d1) * np.exp(-0.5 * d2 - 0.5 * d3)


class WeaklyPeriodicKernelSklearn(
    StationaryKernelMixin, NormalizedKernelMixin, KernelSklearn
):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        """Returns the length scale"""
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):

        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)

        K1 = np.power(X - Y.T, 2)
        K2 = np.power(np.cos(8 * np.pi * X) - np.cos(8 * np.pi * Y.T), 2)
        K3 = np.power(np.sin(8 * np.pi * X) - np.sin(8 * np.pi * Y.T), 2)

        return np.exp(-1 / 8.0 * K1 / self.length_scale) * np.exp(
            -0.5 * K2 / self.length_scale - 0.5 * K3 / self.length_scale
        )

    def __repr__(self):
        return "{0}(length_scale={1:.3g})".format(
            self.__class__.__name__, self.length_scale
        )


class Matern52Kernel(Kernel):
    def __init__(self, lengthscale=0.5, amplitude=1.0):

        super().__init__(lengthscale=lengthscale, amplitude=amplitude)

    def __call__(self, a, b):
        """a, b assumed to be of shape (N,1)"""

        dist = np.abs(a - b.T)
        l = self.lengthscale
        result = np.exp(-np.sqrt(5) * dist / l)
        result *= 1 + np.sqrt(5) * dist / l + 5 / 3 * np.power(dist, 2) / l / l
        result *= self.amplitude
        return result


class GaussianProcessGenerator(FunctionGenerator):
    """
    Generate samples from a Gaussian Process prior.

    Args:
        batch_size (int): Batch size.
        kernel_type (type): A class that can construct a covariance matrix
            when an instance of it is called.
        kernel_kwargs (dict): Kernel arguments.

    """

    def __init__(
        self, batch_size, kernel_type, kernel_kwargs, noise=1e-6, *args, **kwargs
    ):

        super().__init__(batch_size, *args, **kwargs)

        if kernel_kwargs is None:
            kernel_kwargs = dict()
        self.kernel = kernel_type(**kernel_kwargs)

        self.noise = noise

    def apply(self, x):
        """
        Generate y values for input.

        Args:
            x (np.ndarray): x values, shape (1, N, 1)

        Returns:
            np.ndarray: y values, shape (B, N, 1)

        """

        # This could be replaced by a more professional implementation,
        # but it's fast
        K = self.kernel(x[0], x[0])
        L = np.linalg.cholesky(K + self.noise * np.eye(x.shape[1]))
        samples = np.random.normal(size=(x.shape[1], self.batch_size))
        samples = np.dot(L, samples)
        samples = samples.astype(x.dtype)

        return samples.T[:, :, None]
