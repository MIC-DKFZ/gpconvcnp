import numpy as np
import torch

from neuralprocess.data.base import FunctionGenerator


class LotkaVolterraGenerator(FunctionGenerator):
    """
    Generate population dynamics using Lotka-Volterra model.

    Args:
        batch_size (int): Batch size.

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
        rescale=0.01,
        max_time=100.0,
        max_population=500,
        super_sample=4.0,
        min_batch_size=4,
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
        self.rescale = rescale
        self.max_time = max_time
        self.max_population = max_population
        self.super_sample = super_sample
        self.min_batch_size = min_batch_size

    @staticmethod
    def generate_instance(
        predator_init, prey_init, rate0, rate1, rate2, rate3, sequence_length, max_time,
    ):

        predator = [predator_init]
        prey = [prey_init]
        time = [0.0]

        X = predator[-1]
        Y = prey[-1]
        T = time[-1]

        for i in range(sequence_length - 1):

            current_rates = [
                rate0 * X * Y,
                rate1 * X,
                rate2 * Y,
                rate3 * X * Y,
            ]
            total_rate = np.sum(current_rates)
            T += np.random.exponential(1 / total_rate)
            event = np.argmax(np.random.multinomial(1, current_rates / total_rate))

            if event == 0:
                X += 1
            elif event == 1:
                X -= 1
            elif event == 2:
                Y += 1
            elif event == 3:
                Y -= 1
            else:
                raise ValueError(
                    "Somehow got an event indicator {},\
                    which is not in (0, 1, 2, 3),\
                    but there are only 4 possible events!".format(
                        event
                    )
                )

            time.append(T)
            predator.append(X)
            prey.append(Y)

            # check some conditions
            if T > max_time or X == 0 or Y == 0:
                break

        predator = np.array(predator)
        prey = np.array(prey)
        time = np.array(time)

        return time, predator, prey

    def generate_train_batch(self):
        """
        We also have to reimplement how x values are drawn,
        so we can't just use self.apply(x).
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
        predator_batch *= self.rescale
        prey_batch *= self.rescale

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
