import os
import numpy as np
import pandas as pd
import time
import wfdb
import requests
import zipfile
import glob
import torch

from neuralprocess.data.base import FunctionGenerator


class ECGGenerator(FunctionGenerator):
    def __init__(
        self,
        batch_size,
        sequence_length=1000,  # ca. 4 heartbeats
        x_range=(0, 3),
        working_directory=None,
        test=False,
        percentile_as_target=0,
        **kwargs,
    ):

        if working_directory is None:
            if "ECG_DATA_DIR" in os.environ:
                self.working_directory = os.environ["ECG_DATA_DIR"]
            else:
                self.working_directory = os.getcwd()
        else:
            self.working_directory = working_directory

        f = os.path.join(self.working_directory, "data.zip")

        # download data
        if not os.path.exists(f):
            r = requests.get(
                "https://storage.googleapis.com/mitdb-1.0.0.physionet.org/mit-bih-arrhythmia-database-1.0.0.zip"
            )
            with open(f, "wb") as outfile:
                outfile.write(r.content)
            time.sleep(2)

        # unzip data
        if not os.path.exists(
            os.path.join(self.working_directory, "mit-bih-arrhythmia-database-1.0.0")
        ):
            with zipfile.ZipFile(f, "r") as infile:
                infile.extractall()

        data = []
        subjects = []
        for f in sorted(
            glob.glob(
                os.path.join(
                    self.working_directory, "mit-bih-arrhythmia-database-1.0.0", "*.dat"
                )
            )
        ):
            subjects.append(os.path.basename(f).split(".")[0])
            data.append(
                wfdb.rdrecord(f.replace(".dat", "")).p_signal.astype(np.float32)
            )
        data = np.stack(data)

        super().__init__(batch_size, data=data, **kwargs)

        self.subjects_train = [s for s in subjects if s.startswith("1")]
        self.subjects_test = [s for s in subjects if s.startswith("2")]
        self.sequence_length = sequence_length
        self.x_range = x_range
        self.percentile_as_target = percentile_as_target
        self.test = test

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

        y = []
        subjects = []
        starting_indices = []
        for i in range(self.batch_size):

            if self.test:
                index = np.random.randint(len(self.subjects_test))
                subject = self.subjects_test[index]
                index += len(self.subjects_train)
            else:
                index = np.random.randint(len(self.subjects_train))
                subject = self.subjects_train[index]
            subjects.append(subject)

            start_index = np.random.randint(
                0, self._data.shape[1] - sequence_length + 1
            )
            sequence = self._data[index, start_index : start_index + sequence_length]
            y.append(sequence)
            starting_indices.append(start_index)

        y = np.stack(y).astype(np.float32)

        # if desired, we include the signal peaks in the target set
        if self.percentile_as_target > 0.0:

            lower = np.percentile(y, self.percentile_as_target)
            upper = np.percentile(y, 100 - self.percentile_as_target)

            indices_lower = np.where(y < lower)[1]
            indices_upper = np.where(y > upper)[1]
            indices = np.union1d(indices_lower, indices_upper)

            if len(indices) > num_target:
                indices = np.random.choice(indices, num_target, replace=False)

        else:

            indices = []

        num_target_remaining = num_target - len(indices)
        rand_indices = np.random.choice(
            np.arange(sequence_length),
            num_context + num_target_remaining,
            replace=False,
        )
        context_indices = rand_indices[:num_context].copy()
        if self.target_includes_context:
            target_indices = rand_indices
        else:
            target_indices = rand_indices[-num_target_remaining:]
        target_indices = np.union1d(target_indices, indices).astype(np.int)
        context_indices.sort()
        target_indices.sort()

        x = np.linspace(*self.x_range, sequence_length)
        x = x.reshape(1, -1, 1).astype(np.float32)
        x = np.repeat(x, self.batch_size, 0)

        context_in = x[:, context_indices]
        target_in = x[:, target_indices]
        context_out = y[:, context_indices]
        target_out = y[:, target_indices]

        return dict(
            context_in=torch.from_numpy(context_in),
            context_out=torch.from_numpy(context_out),
            target_in=torch.from_numpy(target_in),
            target_out=torch.from_numpy(target_out),
            subjects=subjects,
            x=x,
            y=y,
            start=starting_indices,
        )
