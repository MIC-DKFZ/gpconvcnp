import numpy as np
import torch

from neuralprocess.data.base import FunctionGenerator


class FourierSeriesGenerator(FunctionGenerator):
    """
    Generate random Fourier series. We use the amplitude-phase representation,
    meaning the instances are of the form
        
        y(x) = bias + SUM_k(a_k * cos((k*x - phase_k) / frequency_scale))

    Args:
        batch_size (int): Batch size.
        series_length (int): Number of Fourier components, not counting
            bias term. Can be an iterable to draw randomly from.
        amplitude (float): a_k in definition above.
            Can be an iterable to draw randomly from.
        phase (float): phase_k in definition above.
            Can be an iterable to draw randomly from.
        bias (float): bias in definition above.
            Can be an iterable to draw randomly from.
        frequency_scale (float): Scale for x axis, see definition above.
            Can be an iterable to draw randomly from.

    """

    def __init__(
        self,
        batch_size,
        series_length=[1, 10],
        amplitude=[-1, 1],
        phase=[-1, 1],
        bias=[-1, 1],
        frequency_scale=2.0,
        **kwargs
    ):

        super().__init__(batch_size, **kwargs)

        self.series_length = series_length
        self.amplitude = amplitude
        self.phase = phase
        self.bias = bias
        self.frequency_scale = frequency_scale

    def apply(self, x):
        """
        Generate y values for input.

        Args:
            x (np.ndarray): x values, shape (1, N, 1)

        Returns:
            np.ndarray: y values, shape (B, N, 1)

        """

        if hasattr(self.series_length, "__iter__"):
            series_length = np.random.randint(*self.series_length)
        else:
            series_length = self.series_length

        if hasattr(self.bias, "__iter__"):
            y = np.random.uniform(*self.bias, size=(self.batch_size, 1, 1))
        else:
            y = np.ones((self.batch_size, 1, 1)) * self.bias

        if hasattr(self.frequency_scale, "__iter__"):
            frequency_scale = np.random.uniform(
                *self.frequency_scale, size=(self.batch_size, 1, 1)
            )
        else:
            frequency_scale = np.ones((self.batch_size, 1, 1)) * self.frequency_scale

        for i in range(series_length):

            if hasattr(self.amplitude, "__iter__"):
                amplitude = np.random.uniform(
                    *self.amplitude, size=(self.batch_size, 1, 1)
                )
            else:
                amplitude = np.ones((self.batch_size, 1, 1)) * self.amplitude

            if hasattr(self.phase, "__iter__"):
                phase = np.random.uniform(*self.phase, size=(self.batch_size, 1, 1))
            else:
                phase = np.ones((self.batch_size, 1, 1)) * self.phase

            y = y + amplitude * np.cos(((i + 1) * x - phase) / frequency_scale)

        return y.astype(np.float32)
