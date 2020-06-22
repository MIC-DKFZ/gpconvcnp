import os
import numpy as np
import pandas as pd
import torch

from neuralprocess.data.base import FunctionGenerator


def get_lynx_hare_data():
    """Load the Hudson Bay Company lynx-hare dataset."""
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "lynxhare.csv"))


class LotkaVolterraGenerator(FunctionGenerator):
    """
    Generate population dynamics using a Lotka-Volterra model. Let (X, Y)
    denote the population of (predator, prey).
    Starting with the initial population (X, Y) = (predator_init, prey_init),
    we draw time increments from an exponential distribution and
    let one of the following events occur:

        1. A single predator is born with probability proportional to
        rate = rate1 * X * Y
        2. A single predator dies with probability proportional to
        rate = rate2 * X
        3. A single prey is born with probability proportional to
        rate = rate3 * Y
        4. A single prey dies with probability proportional to
        rate = rate4 * X * Y

    The time increments are drawn from an exponential distribution with
    a rate parameter that is the sum of all the above rates. We are removing
    populations that have died out, that are too large, or that have a time
    axis that is too large.

    Args:
        batch_size (int): Batch size.
        predator_init (int): Intial number of predators.
            Can be an iterable to draw randomly from.
        prey_init (int): Intial number of prey.
            Can be an iterable to draw randomly from.
        rate0 (float): First rate parameter, see above for explanation.
            Can be an iterable to draw randomly from.
        rate1 (float): Second rate parameter, see above for explanation.
            Can be an iterable to draw randomly from.
        rate2 (float): Third rate parameter, see above for explanation.
            Can be an iterable to draw randomly from.
        rate3 (float): Fourth rate parameter, see above for explanation.
            Can be an iterable to draw randomly from.
        sequence_length (int): Sequences will have this many events.
        x_rescale (float): Rescale time axis by this multiplier.
        y_rescale (float): Rescale population axis by this multiplier.
        max_time (float): Remove sequences with time values larger than this.
            This ensures somewhat comparable time axes between sequences.
        max_population (int): Remove populations larger than this.
        super_sample (float): Multiply batch size internally by this number.
            Because we remove some examples, we need to create more to still
            be able to keep our batch size. If you need to set this parameter
            for a new configuration of parameters, set this to a large value
            and check the "rejection_ratio" in the batches. You can use that
            to then set a sensible value that still practically guarantees
            full batches.
        min_batch_size (int): Internal minimum batch size. This is necessary
            for small batch sizes. Assume you have a rejection ratio of
            ~ 30% and you set super_sample to 2. For large batch sizes that
            practically guarantess full batches, but if you want to work with
            batch size of 1, for example, that would still mean there's a
            probability of ~ 50% that a batch is empty.

    """

    def __init__(
        self,
        batch_size,
        predator_init=50,
        prey_init=100,
        rate0=0.01,
        rate1=0.5,
        rate2=1.0,
        rate3=0.01,
        sequence_length=10000,
        x_rescale=0.1,
        y_rescale=0.01,
        max_time=100.0,
        max_population=500,
        super_sample=4.0,
        min_batch_size=16,
        **kwargs,
    ):

        super().__init__(batch_size, **kwargs)

        self.predator_init = predator_init
        self.prey_init = prey_init
        self.rate0 = rate0
        self.rate1 = rate1
        self.rate2 = rate2
        self.rate3 = rate3
        self.sequence_length = sequence_length
        self.x_rescale = x_rescale
        self.y_rescale = y_rescale
        self.max_time = max_time
        self.max_population = max_population
        self.super_sample = super_sample
        self.min_batch_size = min_batch_size

    def generate_train_batch(self):
        """
        We also have to reimplement how x values are drawn,
        so we can't just use self.apply(x) like in other generators.
        """

        # get context and target sizes for batch
        if hasattr(self.num_context, "__iter__"):
            num_context = np.random.randint(*self.num_context)
        else:
            num_context = self.num_context

        if hasattr(self.num_target, "__iter__"):
            if not self.target_larger_than_context:
                num_target = np.random.randint(*self.num_target)
            else:
                num_target = np.random.randint(
                    max(num_context, self.num_target[0]), self.num_target[1]
                )
        else:
            num_target = self.num_target

        if self.target_includes_context and self.target_fixed_size:
            num_target -= num_context

        bs = max(self.min_batch_size, int(self.super_sample * self.batch_size))

        # draw process parameters
        if hasattr(self.predator_init, "__iter__"):
            predator = np.random.randint(*self.predator_init, size=(bs, 1, 1),)
        else:
            predator = np.ones((bs, 1, 1)) * self.predator_init

        if hasattr(self.prey_init, "__iter__"):
            prey = np.random.randint(*self.prey_init, size=(bs, 1, 1))
        else:
            prey = np.ones((bs, 1, 1)) * self.prey_init

        rates = []
        for rate in (self.rate0, self.rate1, self.rate2, self.rate3):
            if hasattr(rate, "__iter__"):
                rates.append(np.random.uniform(*rate, size=(bs, 1, 1)))
            else:
                rates.append(np.ones((bs, 1, 1)) * rate)
        rates = np.concatenate(rates, 1)

        predator_batch = [predator.copy()]
        prey_batch = [prey.copy()]
        time_batch = [np.zeros_like(predator).astype(np.float32)]
        time = time_batch[-1].copy()

        for i in range(self.sequence_length - 1):

            # compute rates
            current_rates = np.concatenate(
                [
                    rates[:, 0:1] * predator * prey,
                    rates[:, 1:2] * predator,
                    rates[:, 2:3] * prey,
                    rates[:, 3:4] * predator * prey,
                ],
                1,
            )
            total_rates = np.sum(current_rates, 1, keepdims=True)
            total_rates[total_rates == 0] = np.inf

            # update times
            time += np.random.exponential(1 / total_rates)

            # draw events and update populations
            # we do this by assuming an interval with different sections
            # corresponding to the possible events
            current_rates = current_rates / total_rates
            current_rates = np.cumsum(current_rates, 1)
            positions = np.random.uniform(size=(bs, 1, 1))
            predator[positions < current_rates[:, 0:1]] += 1
            predator[
                np.logical_and(
                    positions >= current_rates[:, 0:1],
                    positions < current_rates[:, 1:2],
                )
            ] -= 1
            prey[
                np.logical_and(
                    positions >= current_rates[:, 1:2],
                    positions < current_rates[:, 2:3],
                )
            ] += 1
            prey[
                np.logical_and(
                    positions >= current_rates[:, 2:3],
                    positions < current_rates[:, 3:4],
                )
            ] -= 1

            predator_batch.append(predator.copy())
            prey_batch.append(prey.copy())
            time_batch.append(time.copy())

        predator_batch = np.concatenate(predator_batch, 1).astype(np.float32)
        prey_batch = np.concatenate(prey_batch, 1).astype(np.float32)
        time_batch = np.concatenate(time_batch, 1).astype(np.float32)
        predator_batch *= self.y_rescale
        prey_batch *= self.y_rescale
        time_batch *= self.x_rescale

        # filter out bad examples
        time_too_long = np.unique(np.where(time_batch > self.max_time)[0])
        died = np.unique(np.where((predator + prey) == 0)[0])
        exploded = np.unique(np.where((predator + prey) > self.max_population)[0])
        remove_indices = np.union1d(time_too_long, died)
        remove_indices = np.union1d(remove_indices, exploded)
        rejection_ratio = len(remove_indices) / (self.super_sample * self.batch_size)
        if rejection_ratio == 1.0:
            return dict(rejection_ratio=1.0)

        predator_batch = np.delete(predator_batch, remove_indices, 0)
        prey_batch = np.delete(prey_batch, remove_indices, 0)
        time_batch = np.delete(time_batch, remove_indices, 0)
        predator_batch = predator_batch[: self.batch_size]
        prey_batch = prey_batch[: self.batch_size]
        time_batch = time_batch[: self.batch_size]

        # draw random indices
        rand_indices = np.random.choice(
            np.arange(predator_batch.shape[1]), num_context + num_target, replace=False,
        )
        context_indices = rand_indices[:num_context].copy()
        if self.target_includes_context:
            target_indices = rand_indices
        else:
            target_indices = rand_indices[-num_target:]
        context_indices.sort()
        target_indices.sort()

        context_in = time_batch[:, context_indices, :]
        context_out = np.concatenate(
            [predator_batch[:, context_indices, :], prey_batch[:, context_indices, :]],
            -1,
        )
        target_in = time_batch[:, target_indices, :]
        target_out = np.concatenate(
            [predator_batch[:, target_indices, :], prey_batch[:, target_indices, :]],
            -1,
        )

        return dict(
            context_in=torch.from_numpy(context_in),
            context_out=torch.from_numpy(context_out),
            target_in=torch.from_numpy(target_in),
            target_out=torch.from_numpy(target_out),
            time=time_batch,
            predator=predator_batch,
            prey=prey_batch,
            rejection_ratio=rejection_ratio,
        )
