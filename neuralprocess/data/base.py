import numpy as np
import torch
from batchgenerators.dataloading import SlimDataLoaderBase



class FunctionGenerator(SlimDataLoaderBase):
    """
    Base class for function generators. Inherit from this and implement
    the 'apply' method, which generates the y values for a given x.

    Args:
        batch_size (int): Batch size.
        x_range (tuple): Indicates the bounds for input values.
        num_context (int): Number of context points. If this is a list
            or tuple, we draw randomly from that interval.
        num_target (int): Number of target points. If this is a list
            or tuple, we draw randomly from that interval.
        target_larger_than_context (bool): Ensures the target set is
            at least as big as the context set. Only works if
            num_target is drawn randomly.
        target_includes_context (bool): Will include the context set in
            the target set.
        output_noise (float): Additive noise for the y values in the
            context set.
        linspace (bool): Use linspace instead of random uniform draws
            for x.
        number_of_threads_in_multithreaded (int): Set this to the
            appropriate number when you wrap the generator in a
            MultithreadedAugmenter.

    """

    def __init__(self,
                 batch_size,
                 x_range=(-3, 3),
                 num_context=(3, 100),
                 num_target=(3, 100),
                 target_larger_than_context=True,
                 target_includes_context=True,
                 output_noise=0.,
                 linspace=False,
                 number_of_threads_in_multithreaded=1,
                 **kwargs):

        super().__init__(None, batch_size, number_of_threads_in_multithreaded)

        self.x_range = x_range
        self.num_context = num_context
        self.num_target = num_target
        self.target_larger_than_context = target_larger_than_context
        self.target_includes_context = target_includes_context
        self.output_noise = output_noise
        self.linspace = linspace

    def generate_train_batch(self):
        """This method generates a batch."""

        if hasattr(self.num_context, "__iter__"):
            num_context = np.random.randint(*self.num_context)
        else:
            num_context = self.num_context

        if hasattr(self.num_target, "__iter__"):
            if not self.target_larger_than_context:
                num_target = np.random.randint(*self.num_target)
            else:
                num_target = np.random.randint(
                    max(num_context, self.num_target[0]), self.num_target[1])
        else:
            num_target = self.num_target

        if self.linspace:
            x = np.linspace(*self.x_range, num_context + num_target)
            x = x.reshape(1, -1, 1)
        else:
            x = np.random.uniform(*self.x_range, size=(1, num_context + num_target, 1))
            x.sort(1)
        x = x.astype(np.float32)

        # x now has shape (1, num_context + num_target, 1)
        y = self.apply(x)

        x = np.repeat(x, self.batch_size, 0)
        rand_indices = np.random.choice(np.arange(num_context + num_target),
                                        num_context,
                                        replace=False)
        rand_indices.sort()
        context_in = x[:, rand_indices, :]
        context_out = y[:, rand_indices, :]
        if self.target_includes_context:
            target_in = x
            target_out = y
        else:
            target_in = np.delete(x, rand_indices, 1)
            target_out = np.delete(y, rand_indices, 1)

        if self.output_noise > 0:
            context_out += np.random.uniform(-self.output_noise,
                                             self.output_noise,
                                             size=context_out.shape)

        return dict(
            context_in=torch.from_numpy(context_in),
            context_out=torch.from_numpy(context_out),
            target_in=torch.from_numpy(target_in),
            target_out=torch.from_numpy(target_out)
        )

    def apply(self, x):
        """This method should generate the y values for a given x."""
        
        raise NotImplementedError