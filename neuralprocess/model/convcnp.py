import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gpytorch

from neuralprocess.util import stack_batch, unstack_batch, make_grid
from neuralprocess.model import generic

########################################################################
### THE FOLLOWING IS DIRECTLY COPIED FROM THE OFFICIAL CONVCNP
### REPOSITORY, WITH SOME UTILITY FUNCTIONS EVALUATED IN-PLACE,
### SOME VARIABLES RENAMED TO BE MORE EXPRESSIVE, AND CHANGES
### INTRODUCED TO INTEGRATE A GAUSSIAN PROCESS.
###
### https://github.com/cambridge-mlg/convcnp
########################################################################


def custom_init(m):
    """Custom initialization used in ConvCNP."""

    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, 1.0)
        if hasattr(m, "bias"):
            nn.init.constant_(m.bias, 0.0)
    elif type(m) in (nn.Conv1d, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight, 1.0)
        if hasattr(m, "bias"):
            nn.init.constant_(m.bias, 1e-3)


class ConvDeepSet(nn.Module):
    """
    One-dimensional set convolution layer. Uses an RBF kernel.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        learn_length_scale (bool): Learn the length scales of the channels.
        init_length_scale (float): Initial value for the length scale.
        use_density (bool): Append density channel to inputs.
            Defaults to True.
        use_density_norm (bool): Normalize other channels by density.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        learn_length_scale=True,
        init_length_scale=0.1,
        use_density=True,
        use_density_norm=True,
    ):

        super(ConvDeepSet, self).__init__()

        self.out_channels = out_channels
        self.use_density = use_density
        self.use_density_norm = use_density_norm
        self.in_channels = in_channels + 1 if self.use_density else in_channels
        self.sigma = nn.Parameter(
            np.log(init_length_scale) * torch.ones(self.in_channels),
            requires_grad=learn_length_scale,
        )
        self.sigma_fn = torch.exp

        self.projection = nn.Linear(self.in_channels, self.out_channels)
        self.projection.apply(custom_init)

    def forward(self, context_in, context_out, target_in, *args, **kwargs):
        """
        Forward pass through the layer with evaluations at locations target_in.

        Args:
            context_in (torch.tensor): Inputs of observations, shape (B, N, 1).
            context_out (torch.tensor): Outputs of observations, shape (B, N, Cin).
            target_in (torch.tensor): Inputs to evaluate function at, shape (B, M, 1).

        Returns:
            torch.tensor: Outputs of evaluated function, shape (B, M, Cout).

        """
        # Ensure that context_in, context_out, and target_in are rank-3 tensors.
        if len(context_in.shape) == 2:
            context_in = context_in.unsqueeze(2)
        if len(context_out.shape) == 2:
            context_out = context_out.unsqueeze(2)
        if len(target_in.shape) == 2:
            target_in = target_in.unsqueeze(2)

        # Compute shapes.
        B, N = context_in.shape[:2]
        M = target_in.shape[1]

        # Compute the pairwise distances.
        # Shape: (B, N, M).
        dists = torch.pow(context_in.unsqueeze(2) - target_in.unsqueeze(1), 2)
        dists = dists.sum(-1)

        # Compute the weights.
        # Shape: (B, N, M, Cin).
        weights = self.rbf(dists)

        # Append density channel if required
        if self.use_density:
            density = torch.ones(B, N, 1).to(context_in.device)
            context_out = torch.cat([density, context_out], dim=2)

        # Perform the weighting, then sum over inputs
        # Shape: (B, M, Cin).
        output = context_out.unsqueeze(2) * weights
        output = output.sum(1)

        if self.use_density and self.use_density_norm:
            # Use density channel to normalize convolution
            density, conv = output[..., :1], output[..., 1:]
            conv = conv / (density + 1e-8)
            output = torch.cat((density, conv), dim=-1)

        # Apply the output projection
        # Shape: (B, M, Cout).
        output = stack_batch(output)
        output = self.projection(output)
        output = unstack_batch(output, B)

        return output

    def rbf(self, dists):
        """
        Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (torch.tensor): Pair-wise squared distance matrix

        Returns:
            torch.tensor: RBF kernel applied to dists.

        """

        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        return torch.exp(-0.5 * dists.unsqueeze(-1) / scales ** 2)


class GPConvDeepSet(ConvDeepSet):
    """
    A ConvDeepSet that replaces the kernel interpolation with a GP.
    """

    def __init__(
        self,
        gp_sample_from_posterior=False,
        gp_lambda_learnable=False,
        gp_noise_learnable=False,
        gp_noise_init=-14.0,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.gp_sample_from_posterior = gp_sample_from_posterior
        self.gp_lambda_learnable = gp_lambda_learnable
        self.gp_lambda = nn.Parameter(torch.zeros(1), requires_grad=gp_lambda_learnable)
        self.gp_noise_learnable = gp_noise_learnable
        self.gp_noise = nn.Parameter(
            gp_noise_init * torch.ones(1), requires_grad=gp_noise_learnable
        )

        self.representation = None
        self.last_prediction = None

    def forward(
        self, context_in, context_out, target_in, store_rep=False, *args, **kwargs
    ):
        """
        Forward pass through the layer with evaluations at locations target_in.

        Args:
            context_in (torch.tensor): Inputs of observations, shape (B, N, 1).
            context_out (torch.tensor): Outputs of observations, shape (B, N, 1).
            target_in (torch.tensor): Inputs to evaluate function at, shape (B, M, 1).
            store_rep (bool): Store computations necessary for sampling.

        Returns:
            torch.tensor: Outputs of evaluated function, shape (B, M, Cout).

        """
        # Ensure that context_in, context_out, and target_in are rank-3 tensors.
        if len(context_in.shape) == 2:
            context_in = context_in.unsqueeze(2)
        if len(context_out.shape) == 2:
            context_out = context_out.unsqueeze(2)
        if len(target_in.shape) == 2:
            target_in = target_in.unsqueeze(2)

        # Compute shapes.
        B, N = context_in.shape[:2]
        M = target_in.shape[1]

        # Compute GP kernel and decompose
        # we can't use self.rbf for K when there is also a sigma for
        # the density
        dists_in = torch.pow(context_in.unsqueeze(2) - context_in.unsqueeze(1), 2)
        dists_in = dists_in.sum(-1)
        K = torch.exp(-0.5 * dists_in / self.sigma_fn(self.sigma[-1]) ** 2)
        K = K + torch.eye(K.shape[-1]).type_as(K) * self.sigma_fn(
            self.gp_noise
        ).type_as(K)
        L = gpytorch.utils.cholesky.psd_safe_cholesky(K)  # (B, N, N)

        # Do the same for context_in and target_in
        dists = torch.pow(context_in.unsqueeze(2) - target_in.unsqueeze(1), 2)
        dists = dists.sum(-1)
        K_s = torch.exp(-0.5 * dists / self.sigma_fn(self.sigma[-1]) ** 2)
        L_k, _ = torch.solve(K_s, L)  # (B, N, M)

        # Compute mean prediction
        # Shape (B, M, 1)
        output = torch.bmm(L_k.transpose(1, 2), torch.solve(context_out, L)[0])

        # Sample from the posterior if desired
        if self.gp_sample_from_posterior and self.training:
            K_ss = torch.exp(
                -0.5
                * torch.pow(target_in.unsqueeze(2) - target_in.unsqueeze(1), 2).sum(-1)
                / self.sigma_fn(self.sigma[-1]) ** 2
            )  # (B, M, M)
            L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(
                K_ss - torch.bmm(L_k.transpose(1, 2), L_k)
            )  # (B, M, M)
            sample = torch.randn(output.shape[0], L_ss.shape[-1], 1)
            sample = sample.to(device=L_ss.device)
            self.gp_lambda = self.gp_lambda.type_as(sample)
            sample = self.sigma_fn(self.gp_lambda) * torch.bmm(
                L_ss, sample
            )  # (B, M, 1)
            output = output + sample

        # Append density channel if required
        if self.use_density:
            weights = torch.exp(-0.5 * dists / self.sigma_fn(self.sigma[0]) ** 2)
            density = torch.ones(B, N, M).to(dists.device) * weights
            density = torch.sum(density, 1).unsqueeze(-1)  # (B, M, 1)
            if self.use_density_norm:
                output /= density + 1e-8
            output = torch.cat((density, output), -1)

        # Store for sampling
        if store_rep:
            self.representation = L_k
            self.last_prediction = output

        # Apply the output projection
        # Shape: (B, M, Cout).
        output = stack_batch(output)
        output = self.projection(output)
        output = unstack_batch(output, B)

        return output

    def sample(self, target_in, num_samples, gp_lambda=None):
        """Draw GP samples for target_in.

        Args:
            target_in (torch.tensor): New input values, shape (B, M, 1).
            num_samples (int): Draw this many samples.
            gp_lambda (float): GP covariance will be squeezed by this factor.
                If None, we use the stored attribute.

        Returns:
            torch.tensor: Output values at target_in, shape (num_samples, B, M, Cout).

        """

        if self.representation is None or self.last_prediction is None:
            raise ValueError("Please run a forward pass with store_rep=True!")

        if target_in.shape[1] != self.representation.shape[-1]:
            raise IndexError(
                "target_in shape is {}, representation shape is {}. \
                Second axis of target_in and last axis of representation should \
                match!".format(
                    target_x.shape, self.representation.shape
                )
            )

        if gp_lambda is None:
            gp_lambda = self.gp_lambda
        elif not torch.is_tensor(gp_lambda):
            gp_lambda = torch.tensor(gp_lambda)
        gp_lambda = gp_lambda.type_as(target_in)

        mu = self.last_prediction
        if self.use_density:
            density, mu = mu[..., :1], mu[..., 1:]  # (B, M, 1)
        L_k = self.representation  # (B, N, M)

        dists = torch.pow(target_in.unsqueeze(2) - target_in.unsqueeze(1), 2)
        dists = dists.sum(-1)  # (B, M, M)
        K_ss = torch.exp(-0.5 * dists / self.sigma_fn(self.sigma[-1]) ** 2)
        K_ss -= torch.bmm(L_k.transpose(1, 2), L_k)
        L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(K_ss)

        # Draw samples
        samples = torch.randn(mu.shape[0], L_ss.shape[-1], num_samples)
        samples = samples.to(device=L_ss.device)
        samples = gp_lambda * torch.bmm(L_ss, samples)  # (B, M, num_samples)
        samples = samples.permute(2, 0, 1).unsqueeze(-1)  # (num_samples, B, M, 1)
        samples = mu + samples

        if self.use_density:
            density = density.unsqueeze(0).repeat(num_samples, 1, 1, 1)
            if self.use_density_norm:
                samples = samples / (density + 1e-8)
            samples = torch.cat((density, samples), -1)

        B, M = samples.shape[1:3]
        samples = samples.view(num_samples * B * M, -1)
        samples = self.projection(samples)
        samples = samples.view(num_samples, B, M, -1)

        return samples


class ConvCNP(nn.Module):
    """
    One-dimensional ConvCNP model. At the moment this uses a hardcoded
    RBF kernel, in the future it would make sense to integrate a kernel
    argument that accepts e.g. GPyTorch kernels.

    Args:
        conv_net (torch.nn.Module): A CNN that transforms the input
            interpolation. Needs to have in_channels and out_channels
            attributes.
        use_gp (bool): Use GPConvDeepSet instead of ConvDeepSet for
            input interpolation.
        learn_length_scale (bool): Learn the kernel length scale.
        init_length_scale (float): Initial value for length scale.
        use_density (bool): Append a density channel to interpolation.
        use_density_norm (bool): Normalize interpolation by density.
        points_per_unit (int): Construct a grid with this resolution.
        range_padding (float): Pad the range by this value when constructing
            the grid.
        grid_divisible_by (int): Ensure the grid is divisible by this number.
            Use this when the CNN performs some sort of pooling or strided
            convolution.
            
    """

    def __init__(
        self,
        conv_net,
        use_gp=False,
        learn_length_scale=True,
        init_length_scale=0.1,
        use_density=True,
        use_density_norm=True,
        points_per_unit=20,
        range_padding=0.1,
        grid_divisible_by=64,
        gp_sample_from_posterior=False,
        gp_lambda_learnable=False,
        gp_noise_learnable=False,
        gp_noise_init=-14.0,
        *args,
        **kwargs
    ):

        super(ConvCNP, self).__init__()

        self.activation = nn.Sigmoid()
        self.conv_net = conv_net
        self.conv_net.apply(custom_init)

        self.use_gp = use_gp
        self.points_per_unit = points_per_unit
        self.range_padding = range_padding
        self.grid_divisible_by = grid_divisible_by
        self.grid = None

        if use_gp:
            self.input_interpolation = GPConvDeepSet(
                in_channels=1,
                out_channels=self.conv_net.in_channels,
                learn_length_scale=learn_length_scale,
                init_length_scale=init_length_scale,
                use_density=use_density,
                use_density_norm=use_density_norm,
                gp_sample_from_posterior=gp_sample_from_posterior,
                gp_lambda_learnable=gp_lambda_learnable,
                gp_noise_learnable=gp_noise_learnable,
                gp_noise_init=gp_noise_init,
            )
        else:
            self.input_interpolation = ConvDeepSet(
                in_channels=1,
                out_channels=self.conv_net.in_channels,
                learn_length_scale=learn_length_scale,
                init_length_scale=init_length_scale,
                use_density=use_density,
                use_density_norm=use_density_norm,
            )

        self.output_interpolation_mean = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False,
        )

        self.output_interpolation_sigma = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False,
        )

    def forward(
        self,
        context_in,
        context_out,
        target_in,
        target_out=None,
        store_rep=False,
        *args,
        **kwargs
    ):
        """
        Forward pass in the Convolutional Conditional Neural Process.

        Args:
            context_in (torch.tensor): Shape (N, B, Cin).
            context_out (torch.tensor): Shape (N, B, Cout).
            target_in (torch.tensor): Shape (M, B, Cin).
            target_out (torch.tensor): Unused.
            store_rep (bool): Store representation.

        Returns:
            torch.tensor: Output of 'output_interpolation'.

        """

        # Ensure that context_in, context_out, and target_in are rank-3 tensors.
        if len(context_in.shape) == 2:
            context_in = context_in.unsqueeze(2)
        if len(context_out.shape) == 2:
            context_out = context_out.unsqueeze(2)
        if len(target_in.shape) == 2:
            target_in = target_in.unsqueeze(2)

        grid = make_grid(
            (context_in, target_in),
            self.points_per_unit,
            self.range_padding,
            self.grid_divisible_by,
        )
        if store_rep:
            self.grid = grid

        representation = self.input_interpolation(
            context_in, context_out, grid, store_rep=store_rep
        )
        representation = self.activation(representation)  # (B, gridsize, R1)

        representation = representation.transpose(1, 2)
        representation = self.conv_net(representation)
        representation = representation.transpose(1, 2)  # (B, gridsize, R2)

        mean = self.output_interpolation_mean(grid, representation, target_in)
        sigma = self.output_interpolation_sigma(grid, representation, target_in)

        return torch.cat((mean, sigma), -1)

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod() for param in self.parameters()])

    def sample(self, target_in, num_samples, gp_lambda=None):
        """
        Sample from the Convolutional Conditional Neural Process.

        Args:
            target_in (torch.tensor): Shape (B, M, 1).
            num_samples (int): Draw this many samples.
            gp_lambda (float): GP covariance will be squeezed by this factor.
                If None, will use the stored attribute.

        Returns:
            torch.tensor: Output of 'output_interpolation'.

        """

        if not hasattr(self.input_interpolation, "sample"):
            raise NotImplementedError(
                "This ConvCNP doesn't use a GP \
                interpolator and can't be sampled from."
            )

        grid = self.grid  # (B, G, 1)

        samples = self.input_interpolation.sample(
            grid, num_samples, gp_lambda=gp_lambda
        )
        samples = self.activation(samples)
        samples = stack_batch(samples)  # (num_samples*B, G, R1)

        samples = samples.transpose(1, 2)
        samples = self.conv_net(samples)
        samples = samples.transpose(1, 2)  # (num_samples*B, G, R2)

        means = self.output_interpolation_mean(
            grid.repeat(num_samples, 1, 1), samples, target_in.repeat(num_samples, 1, 1)
        )
        means = unstack_batch(means, num_samples)

        sigmas = self.output_interpolation_sigma(
            grid.repeat(num_samples, 1, 1), samples, target_in.repeat(num_samples, 1, 1)
        )
        sigmas = unstack_batch(sigmas, num_samples)

        return torch.cat((means, sigmas), -1)


########################################################################
### BELOW IS AN ATTEMPT AT A MORE GENERIC VERSION THAT USES GPYTORCH
### KERNELS. SO FAR THE GRADIENTS ARE PRETTY UNSTABLE.
########################################################################

# class ConvDeepSet(nn.Module):
#     """
#     Use a kernel to interpolate values onto a new input range (e.g. a grid)
#     and optionally project to a new space.

#     Args:
#         kernel (type or callable): Kernel used for interpolation. If it's a
#             type, it will be constructed with kernel_kwargs, otherwise used
#             as is.
#         kernel_kwargs (dict): Arguments for kernel construction.
#         use_density (bool): Make use of a separate density channel.
#         use_density_norm (bool): Divide interpolated result by density channel.
#         project_to (int): Project interpolation to this many channels.
#             0 means no projection.
#         project_bias (bool): Activate bias in projection.
#         project_in_channels (int): Input channels for the projection.

#     """

#     def __init__(self,
#                  kernel,
#                  kernel_kwargs=None,
#                  init_lengthscale=0.1,
#                  in_channels=1,
#                  use_density=True,
#                  use_density_norm=True,
#                  project_to=0,
#                  project_bias=True,
#                  **kwargs):

#         super().__init__()

#         self.in_channels = in_channels
#         if use_density:
#             self.in_channels += 1
#         self.use_density = use_density
#         self.use_density_norm = use_density_norm
#         self.project_to = project_to
#         self.project_bias = project_bias

#         if type(kernel) == type:
#             if kernel_kwargs is None:
#                 kernel_kwargs = dict()
#             kernel_kwargs.update(batch_shape=(1, self.in_channels))
#             self.kernel = kernel(**kernel_kwargs)
#         else:
#             self.kernel = kernel
#         if init_lengthscale is not None:
#             self.kernel._set_lengthscale(init_lengthscale)

#         if self.project_to not in (0, None):
#             self.setup_projections()

#     def setup_projections(self):
#         """Set up modules that project to another space."""

#         self.project_out = nn.Linear(self.in_channels,
#                                      self.project_to,
#                                      bias=self.project_bias)

#     def forward(self, context_in, context_out, target_in, *args, **kwargs):
#         """Interpolate context onto new input range.

#         Args:
#             context_in (torch.tensor): Context inputs, shape (B, N, Cin).
#             context_out (torch.tensor): Context outputs, shape (B, N, Cout).
#             target_in (torch.tensor): New input values, shape (B, M, Cin).

#         Returns:
#             torch.tensor: Output values at new input positions,
#                 optionally projected. Shape (B, M, R).

#         """

#         # Compute shapes.
#         B, N = context_in.shape[:2]
#         M = target_in.shape[1]

#         # Compute the weights.
#         K = self.kernel(context_in.unsqueeze(1), target_in.unsqueeze(1))
#         K = K.evaluate().permute(0, 2, 3, 1)  # (B, N, M, in_channels)

#         # Append density channel if required
#         if self.use_density:
#             density = torch.ones(B, N, 1)
#             density = density.to(dtype=context_in.dtype, device=context_in.device)
#             context_out = torch.cat((density, context_out), -1)

#         # Perform the weighting, then sum over inputs
#         output = context_out.unsqueeze(-2) * K
#         output = output.sum(1)  # (B, M, Cout)

#         # Use density channel to normalize convolution
#         if self.use_density and self.use_density_norm:
#             density, conv = output[..., :1], output[..., 1:]
#             conv = conv / (density + 1e-8)
#             output = torch.cat((density, conv), dim=-1)

#         # Apply the output projection
#         if hasattr(self, "project_out"):
#             output = stack_batch(output)
#             output = self.project_out(output)
#             output = unstack_batch(output, B)

#         return output  # (B, M, R)


# class GPConvDeepSet(ConvDeepSet):
#     """
#     A ConvDeepSet that uses a Gaussian Process for interpolation. Note
#     that this implementation only works with scalar output size.
#     To extend this to multiple output channels, see the Multitask GP
#     Regression examples in GPyTorch.

#     Args:
#         gp_lambda (float): GP covariance will be squeezed by this factor.
#         gp_sample_from_posterior (int): Use the mean of this many posterior
#             samples instead of mean prediction. 0 uses mean prediction.

#     """

#     def __init__(self,
#                  gp_lambda=0.2,
#                  gp_sample_from_posterior=0,
#                  *args, **kwargs):

#         # the kernels for signal and density are separate,
#         # so we initialize with use_density=False
#         use_density = kwargs.get("use_density", False)
#         kwargs["use_density"] = False
#         super().__init__(*args, **kwargs)

#         # if we're using the density channel, we need to set up the
#         # projection again
#         if use_density:
#             self.use_density = True
#             self.in_channels += 1
#             self.setup_projections()

#         self.gp_lambda = gp_lambda
#         self.gp_sample_from_posterior = gp_sample_from_posterior

#         self.reset()

#     def reset(self):
#         """Reset storage attributes to None."""

#         self.representation = None
#         self.last_prediction = None

#     def forward(self,
#                 context_in,
#                 context_out,
#                 target_in,
#                 store_rep=False,
#                 *args, **kwargs):
#         """Interpolate context onto target_in.

#         Args:
#             context_in (torch.tensor): Context inputs, shape (B, N, Cin).
#             context_out (torch.tensor): Context outputs, shape (B, N, Cout).
#             target_in (torch.tensor): New input values, shape (B, M, Cin).
#             store_rep (bool): Store cholesky decomposition of the kernel
#                 matrix as well as mean prediction, so we can sample at a
#                 later point.

#         Returns:
#             torch.tensor: Output values at new input positions,
#                 optionally projected. Shape (B, M, R).

#         """

#         B, N = context_in.shape[:2]
#         M = target_in.shape[1]

#         # do GP inference manually, consider replacing with GPyTorch
#         K = self.kernel(context_in, context_in).evaluate()[0]  # (B, N, N)
#         L = gpytorch.utils.cholesky.psd_safe_cholesky(K, jitter=1e-6)  # (B, N, N)
#         K_s = self.kernel(context_in, target_in).evaluate()[0]  # (B, N, M)
#         L_k, _ = torch.solve(K_s, L)  # (B, N, M)
#         output = torch.bmm(L_k.transpose(1, 2), torch.solve(context_out, L)[0])  # (B, M, Cout)

#         if self.gp_sample_from_posterior:
#             K_ss = self.kernel(target_in, target_in).evaluate()[0]  # (B, M, M)
#             K_ss -= torch.bmm(L_k.transpose(1, 2), L_k)
#             L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(K_ss)
#             samples = torch.randn(self.gp_sample_from_posterior, *output.shape)
#             samples = samples.to(device=L_ss.device)  # (num_samples, B, M, Cout)
#             samples = torch.matmul(L_ss.unsqueeze(0), samples)  # (num_samples, B, M, Cout)
#             output = output.unsqueeze(0) + self.gp_lambda * samples  # (num_samples, B, M, Cout)
#             output = output.mean(0)  # (B, M, Cout)

#         if self.use_density:
#             density = K_s.unsqueeze(-1).mean(1)  # (B, M, 1)
#             if self.use_density_norm:
#                 output = output / (density + 1e-8)
#             output = torch.cat((density, output), -1)

#         if store_rep:
#             self.representation = L_k
#             self.last_prediction = output

#         if hasattr(self, "project_out"):
#             output = stack_batch(output)
#             output = self.project_out(output)
#             output = unstack_batch(output, B)

#         return output

#     def sample(self, target_in, num_samples, gp_lambda=None):
#         """Draw GP samples for target_in.

#         Args:
#             target_in (torch.tensor): New input values, shape (B, M, Cin).
#             num_samples (int): Draw this many samples.
#             gp_lambda (float): GP covariance will be squeezed by this factor.
#                 If None, will use the stored attribute.

#         Returns:
#             torch.tensor: Output values at new input positions,
#                 optionally projected. Shape (num_samples, B, M, R).

#         """

#         if self.representation is None or self.last_prediction is None:
#             raise ValueError("No stored kernel and prediction, please run \
#                 a forward pass with store_rep=True.")

#         if target_in.shape[1] != self.representation.shape[-1]:
#             raise IndexError("target_in shape is {}, representation shape is {}. \
#                 First axis of target_x and last axis of representation should match!"\
#                 .format(target_in.shape, self.representation.shape))

#         if gp_lambda is None:
#             gp_lambda = self.gp_lambda

#         if self.use_density:
#             density, mu = self.last_prediction[..., :1], self.last_prediction[..., 1:]
#         else:
#             mu = self.last_prediction
#         L_k = self.representation  # (B, N, M)

#         K_ss = self.kernel(target_in, target_in).evaluate()[0]  # (B, M, M)
#         K_ss -= torch.bmm(L_k.transpose(1, 2), L_k)
#         L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(K_ss)

#         samples = torch.randn(num_samples, *mu.shape)
#         samples = samples.to(device=L_ss.device)
#         samples = torch.matmul(L_ss, samples)  # (num_samples, B, M, Cout)
#         samples = mu.unsqueeze(0) + gp_lambda * samples  # (num_samples, B, M, Cout)

#         if self.use_density:
#             samples = torch.cat((density.unsqueeze(0).repeat(num_samples, 1, 1, 1),
#                                 samples), -1)  # (num_samples, B, M, Cout+1)

#         B, M = samples.shape[1:3]
#         samples = samples.reshape(num_samples*B*M, -1)
#         samples = self.project_out(samples)
#         samples = samples.reshape(num_samples, B, M, -1)

#         return samples


# class ConvCNP(nn.Module):
#     """
#     ConvCNP works by interpolating context observations onto a grid,
#     applying a CNN, and interpolating again to the output queries.

#     Args:
#         input_interpolation (torch.nn.Module): This module performs the
#             input interpolation to pseudo function space.
#         convnet (torch.nn.Module): This module is applied in pseudo
#             function space, i.e. on the grid interpolation.
#         output_interpolation (torch.nn.Module): This module performs the
#             interpolation to the requested target queries.
#         points_per_unit (int): Use this many grid points per unit interval.
#         range_padding (float): Pad the input range by this value.
#         grid_divisible_by (int): Grid size will be divisible by this
#             number. Will often be required by the 'convnet', e.g. to
#             ensure pooling operations can work.

#     """

#     def __init__(self,
#                  input_interpolation,
#                  conv_net,
#                  output_interpolation,
#                  points_per_unit=20,
#                  range_padding=0.1,
#                  grid_divisible_by=None,
#                  *args, **kwargs):

#         super().__init__()

#         self.input_interpolation = input_interpolation
#         self.conv_net = conv_net
#         self.output_interpolation = output_interpolation
#         self.points_per_unit = points_per_unit
#         self.range_padding = range_padding
#         self.grid_divisible_by = grid_divisible_by

#     def forward(self,
#                 context_in,
#                 context_out,
#                 target_in,
#                 target_out=None,
#                 store_rep=False,
#                 *args, **kwargs):
#         """
#         Forward pass in the Convolutional Conditional Neural Process.

#         Args:
#             context_in (torch.tensor): Shape (B, N, Cin).
#             context_out (torch.tensor): Shape (B, N, Cout).
#             target_in (torch.tensor): Shape (B, M, Cin).
#             target_out (torch.tensor): Unused.
#             store_rep (bool): Store representation.

#         Returns:
#             torch.tensor: Output of 'output_interpolation'.

#         """

#         # grid shape is (B, G, Cin)
#         grid = make_grid((context_in, target_in),
#                          self.points_per_unit,
#                          self.range_padding,
#                          self.grid_divisible_by)
#         if store_rep:
#             self.grid = grid

#         # apply interpolation, conv net and final interpolation
#         representation = self.input_interpolation(context_in,
#                                                   context_out,
#                                                   grid,
#                                                   store_rep=store_rep)
#         representation = representation.transpose(1, 2)  # (B, Cr1, G)
#         representation = self.conv_net(representation)  # (B, Cr2, G)
#         representation = representation.transpose(1, 2)

#         return self.output_interpolation(grid, representation, target_in)

#     def sample(self, target_in, num_samples, gp_lambda=None):
#         """
#         Sample from the Convolutional Conditional Neural Process.

#         Args:
#             target_in (torch.tensor): Shape (M, B, Cin).
#             num_samples (int): Draw this many samples.
#             gp_lambda (float): GP covariance will be squeezed by this factor.
#                 If None, will use the stored attribute.

#         Returns:
#             torch.tensor: Output of 'output_interpolation'.

#         """

#         if not hasattr(self.input_interpolation, "sample"):
#             raise NotImplementedError("This ConvCNP doesn't use a GP \
#                 interpolator and can't be sampled from.")

#         grid = self.grid  # (B, G, Cin)

#         samples = self.input_interpolation.sample(grid, num_samples, gp_lambda)  # (num_samples, B, G, 2)
#         samples = samples.transpose(2, 3)
#         samples = stack_batch(samples)  # (num_samples*B, 2, G)
#         samples = self.conv_net(samples)  # (num_samples*B, R, G)
#         samples = samples.transpose(1, 2)  # (num_samples*B, G, R)

#         grid = grid.repeat(num_samples, 1, 1)
#         target_in = target_in.repeat(num_samples, 1, 1)

#         samples = self.output_interpolation(grid, samples, target_in)  # (num_samples*B, M, Cout)
#         samples = unstack_batch(samples, num_samples)

#         return samples  # (num_samples, B, M, Cout)
