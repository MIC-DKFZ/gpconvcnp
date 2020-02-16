import numpy as np
import torch

from neuralprocess.data.base import FunctionGenerator



class StepFunctionGenerator(FunctionGenerator):
    """
    Generate step functions.

    Args:
        batch_size (int): Batch size.
        y_range (tuple): Allowed y range.
        number_of_steps (int): Number of steps in the interval. If list
            or tuple, will be drawn randomly.
        min_step_width (float): Will have at least this much space
            between steps.
        min_step_height (float): Steps will have a y delta of at least
            this much.

    """

    def __init__(self,
                 batch_size,
                 y_range=[-3, 3],
                 number_of_steps=[3, 10],
                 min_step_width=0.1,
                 min_step_height=0.1,
                 *args, **kwargs):

        super().__init__(batch_size, *args, **kwargs)

        self.y_range = y_range
        self.number_of_steps = number_of_steps
        self.min_step_width = min_step_width
        self.min_step_height = min_step_height

    def apply(self, x):
        """
        Generate y values for input.

        Args:
            x (np.ndarray): x values, shape (1, N, 1)

        Returns:
            np.ndarray: y values, shape (B, N, 1)

        """

        if hasattr(self.number_of_steps, "__iter__"):
            number_of_steps = np.random.randint(*self.number_of_steps)
        else:
            number_of_steps = self.number_of_steps

        # functions in batch will step in same place to make batch generation faster
        num_tries = 0
        while num_tries <= 1000000:  # just so the process doesn't run forever if something's off
            step_indices = np.random.randint(0, x.shape[1], number_of_steps)
            step_indices.sort()
            step_x = x[0, :, 0][step_indices]
            step_width = np.abs(step_x[1:] - step_x[:-1])
            if np.any(step_width < self.min_step_width):
                num_tries += 1
                continue
            else:
                break
        else:
            raise RuntimeError("Tried to generate step function with {} steps \
                and minimum step width of {:.3f}, but failed 1000000 times."\
                .format(number_of_steps, self.min_step_width))

        y = np.zeros((self.batch_size, *x.shape[1:]), dtype=np.float32)
        new_values = np.random.uniform(*self.y_range, size=(self.batch_size, 1))
        y[:, :step_indices[0], 0] = np.repeat(new_values, step_indices[0], 1)
        for i in range(number_of_steps - 1):
            old_values = new_values
            new_values = np.random.uniform(*self.y_range, size=(self.batch_size, 1))
            if self.min_step_height > 0:
                diffs = new_values - old_values
                ind = np.where(np.abs(diffs) < self.min_step_height)
                new_values[ind] += (np.sign(diffs)*self.min_step_height)[ind]
            y[:, step_indices[i]:step_indices[i+1], 0] = np.repeat(new_values, step_indices[i+1] - step_indices[i], 1)
        old_values = new_values
        new_values = np.random.uniform(*self.y_range, size=(self.batch_size, 1))
        if self.min_step_height > 0:
            diffs = new_values - old_values
            ind = np.where(np.abs(diffs) < self.min_step_height)
            new_values[ind] += (np.sign(diffs)*self.min_step_height)[ind]
        y[:, step_indices[-1]:, 0] = np.repeat(new_values, y.shape[1] - step_indices[-1], 1)

        return y