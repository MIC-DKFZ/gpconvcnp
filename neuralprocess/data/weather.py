import os
import numpy as np
import pandas as pd
import torch

from neuralprocess.data.base import FunctionGenerator


def get_temperature_data():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "temperature.csv"))


class TemperatureGenerator(FunctionGenerator):
    """

    Args:
        batch_size (int): Batch size.

    """

    _train_cities = (
        "Vancouver",
        "Portland",
        "San Francisco",
        "Seattle",
        "Los Angeles",
        "San Diego",
        "Las Vegas",
        "Phoenix",
        "Albuquerque",
        "Denver",
        "San Antonio",
        "Dallas",
        "Houston",
        "Kansas City",
        "Minneapolis",
        "Saint Louis",
        "Chicago",
        "Nashville",
        "Indianapolis",
        "Atlanta",
        "Detroit",
        "Jacksonville",
        "Charlotte",
        "Miami",
        "Pittsburgh",
        "Toronto",
        "Philadelphia",
        "New York",
        "Montreal",
        "Boston",
    )

    _test_cities = (
        "Beersheba",
        "Tel Aviv District",
        "Eilat",
        "Haifa",
        "Nahariyya",
        "Jerusalem",
    )

    def __init__(
        self,
        batch_size,
        sequence_length=30 * 24 * 2,  # ca. 2 months
        x_range=(0, 3),
        train_cities=None,
        test_cities=None,
        **kwargs,
    ):

        data = get_temperature_data()
        data = data.iloc[1:44460, 1:]  # remove NaNs at the end and time axis
        data = data.interpolate()  # fill the few remaining NaNs

        super().__init__(batch_size, data=data, **kwargs)

        self.mean_temp = data.mean()
        self.std_temp = data.std()

        self.test = False
        if train_cities is not None:
            self.train_cities = train_cities
        else:
            self.train_cities = self._train_cities
        if test_cities is not None:
            self.test_cities = test_cities
        else:
            self.test_cities = self._test_cities

        self.sequence_length = sequence_length
        self.x_range = x_range

    def generate_train_batch(self):

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

        if hasattr(self.sequence_length, "__iter__"):
            sequence_length = np.random.randint(self.sequence_length)
        else:
            sequence_length = self.sequence_length

        rand_indices = np.random.choice(
            np.arange(sequence_length), num_context + num_target, replace=False
        )
        context_indices = rand_indices[:num_context].copy()
        if self.target_includes_context:
            target_indices = rand_indices
        else:
            target_indices = rand_indices[-num_target:]
        context_indices.sort()
        target_indices.sort()

        x = np.linspace(*self.x_range, sequence_length)
        x = x.reshape(1, -1, 1).astype(np.float32)
        x = np.repeat(x, self.batch_size, 0)
        context_in = x[:, context_indices]
        target_in = x[:, target_indices]

        y = []
        cities = []
        starting_indices = []
        for i in range(self.batch_size):

            if self.test:
                city = np.random.choice(self.test_cities)
            else:
                city = np.random.choice(self.train_cities)
            cities.append(city)

            start_index = np.random.randint(
                1, self._data.shape[0] - sequence_length + 1
            )
            sequence = self._data.loc[
                start_index : start_index + sequence_length - 1, city
            ].values
            sequence = sequence - self.mean_temp[city]
            sequence = sequence / self.std_temp[city]
            y.append(sequence)
            starting_indices.append(start_index)

        y = np.stack(y)[:, :, None].astype(np.float32)
        context_out = y[:, context_indices]
        target_out = y[:, target_indices]

        return dict(
            context_in=torch.from_numpy(context_in),
            context_out=torch.from_numpy(context_out),
            target_in=torch.from_numpy(target_in),
            target_out=torch.from_numpy(target_out),
            cities=cities,
            x=x,
            y=y,
            start=starting_indices,
        )
