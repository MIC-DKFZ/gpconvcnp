import torch
import torch.nn as nn
import gpytorch

from neuralprocess.util import stack_batch, unstack_batch, make_grid



class ConvDeepSet(nn.Module):
    """
    Use a kernel to interpolate values onto a new input range (e.g. a grid)
    and optionally project to a new space.

    Args:
        kernel (gpytorch.kernels.Kernel): Kernel used for interpolation.
        use_density (bool): Make use of a separate density channel.
        use_density_norm (bool): Divide interpolated result by density channel.
        project_to (int): Project interpolation to this many channels.
            0 means no projection.
        project_bias (bool): Activate bias in projection.
        project_in_channels (int): Input channels for the projection.

    """

    def __init__(self,
                 kernel,
                 use_density=True,
                 use_density_norm=True,
                 project_to=0,
                 project_bias=True,
                 project_in_channels=1,
                 **kwargs):

        super().__init__()
        
        self.kernel = kernel
        self.use_density = use_density
        self.use_density_norm = use_density_norm
        self.project_to = project_to
        self.project_bias = project_bias
        self.project_in_channels = project_in_channels

        if self.project_to not in (0, None):
            self.setup_projections()

    def setup_projections(self):
        """Set up modules that project to another space."""

        self.project_out = nn.Linear(self.project_in_channels,
                                     self.project_to,
                                     bias=self.project_bias)

    def forward(self, context_in, context_out, target_in, *args, **kwargs):
        """Interpolate context onto new input range.

        Args:
            context_in (torch.tensor): Context inputs, shape (N, B, Cin).
            context_out (torch.tensor): Context outputs, shape (N, B, Cout).
            target_in (torch.tensor): New input values, shape (M, B, Cin).

        Returns:
            torch.tensor: Output values at new input positions,
                optionally projected. Shape (M, B, R).

        """

        N, B = context_in.shape[:2]
        M = target_in.shape[0]

        if self.use_density:
            density = torch.ones(N, B, 1)
            density = density.to(dtype=context_in.dtype, device=context_in.device)
            context_out = torch.cat((density, context_out), -1)

        context_in = context_in.transpose(0, 1).unsqueeze(1)  # (B, 1, M, Cin)
        target_in = target_in.transpose(0, 1).unsqueeze(1)  # (B, 1, M, Cin)
        K = self.kernel(context_in, target_in).evaluate().permute(0, 2, 3, 1)  # (B, N, M, Cout)

        output = context_out.transpose(0, 1).unsqueeze(-2) * K
        output = output.sum(1)  # (B, M, Cout)
        if self.use_density and self.use_density_norm:
            output[..., 1:] /= (output[..., :1] + 1e-8)

        if hasattr(self, "project_out"):
            output = stack_batch(output)
            output = self.project_out(output)
            output = unstack_batch(output, B)

        return output.transpose(0, 1)



class GPConvDeepSet(ConvDeepSet):
    """
    A ConvDeepSet that uses a Gaussian Process for interpolation. Note
    that this implementation only works with scalar output size.
    To extend this to multiple output channels, see the Multitask GP
    Regression examples in GPyTorch.

    Args:
        gp_lambda (float): GP covariance will be squeezed by this factor.
        gp_sample_from_posterior (int): Use the mean of this many posterior
            samples instead of mean prediction. 0 uses mean prediction.

    """

    def __init__(self,
                 gp_lambda=0.2,
                 gp_sample_from_posterior=0,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.gp_lambda = gp_lambda
        self.gp_sample_from_posterior = gp_sample_from_posterior

        self.reset()

    def reset(self):
        """Reset storage attributes to None."""

        self.representation = None
        self.last_prediction = None

    def forward(self,
                context_in,
                context_out,
                target_in,
                store_rep=False,
                *args, **kwargs):
        """Interpolate context onto target_in.

        Args:
            context_in (torch.tensor): Context inputs, shape (N, B, Cin).
            context_out (torch.tensor): Context outputs, shape (N, B, 1).
            target_in (torch.tensor): New input values, shape (M, B, Cin).
            store_rep (bool): Store cholesky decomposition of the kernel
                matrix as well as mean prediction, so we can sample at a
                later point.

        Returns:
            torch.tensor: Output values at new input positions,
                optionally projected. Shape (M, B, R).

        """

        N, B = context_in.shape[:2]
        M = target_in.shape[0]

        context_in = context_in.transpose(0, 1)
        context_out = context_out.transpose(0, 1)
        target_in = target_in.transpose(0, 1)

        # do GP inference manually, consider replacing with GPyTorch
        K = self.kernel(context_in, context_in).evaluate()  # (B, N, N)
        L = gpytorch.utils.cholesky.psd_safe_cholesky(K)  # (B, N, N)
        K_s = self.kernel(context_in, target_in).evaluate()  # (B, N, M)
        L_k, _ = torch.solve(K_s, L)  # (B, N, M)
        output = torch.bmm(L_k.transpose(1, 2), torch.solve(context_out, L)[0])  # (B, M, 1)

        if self.gp_sample_from_posterior:
            K_ss = self.kernel(target_in, target_in).evaluate()  # (B, M, M)
            K_ss -= torch.bmm(L_k.transpose(1, 2), L_k)
            L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(K_ss)
            samples = torch.randn(output.shape[0],
                                  L_ss.shape[-1],
                                  self.gp_sample_from_posterior)
            samples = samples.to(device=L_ss.device)
            samples = torch.bmm(L_ss, samples)
            output = output + self.gp_lambda * samples  # (B, M, num_samples)
            output = output.mean(-1, keepdims=True)  # (B, M, 1)
        
        if self.use_density:
            density = K_s.unsqueeze(-1).mean(1)  # (B, M, 1)
            if self.use_density_norm:
                output /= (density + 1e-8)
            output = torch.cat((density, output), -1)  

        if store_rep:
            self.representation = L_k
            self.last_prediction = output  

        if hasattr(self, "project_out"):
            output = stack_batch(output)
            output = self.project_out(output)
            output = unstack_batch(output, B)

        return output.transpose(0, 1)

    def sample(self, target_in, num_samples, gp_lambda=None):
        """Draw GP samples for target_in.

        Args:
            target_in (torch.tensor): New input values, shape (M, B, Cin).
            num_samples (int): Draw this many samples.
            gp_lambda (float): GP covariance will be squeezed by this factor.
                If None, will use the stored attribute.

        Returns:
            torch.tensor: Output values at new input positions,
                optionally projected. Shape (num_samples, M, B, R).

        """

        if self.representation is None or self.last_prediction is None:
            raise ValueError("No stored kernel and prediction, please run \
                a forward pass with store_rep=True.")

        if target_in.shape[0] != self.representation.shape[-1]:
            raise IndexError("target_in shape is {}, representation shape is {}. \
                First axis of target_x and last axis of representation should match!"\
                .format(target_in.shape, self.representation.shape))

        if gp_lambda is None:
            gp_lambda = self.gp_lambda

        target_in = target_in.transpose(0, 1)  # (B, M, Cin)
        density, mu = self.last_prediction[..., :1], self.last_prediction[..., 1:]  # (B, M, 1)
        L_k = self.representation  # (B, N, M)

        K_ss = self.kernel(target_in, target_in).evaluate()  # (B, M, M)
        K_ss -= torch.bmm(L_k.transpose(1, 2), L_k)
        L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(K_ss)

        samples = torch.randn(mu.shape[0],
                              L_ss.shape[-1],
                              num_samples)
        samples = samples.to(device=L_ss.device)
        samples = torch.bmm(L_ss, samples)
        samples = mu + gp_lambda * samples  # (B, M, num_samples)

        samples = samples.permute(2, 0, 1)  # (num_samples, B, M)
        samples = torch.cat((density.unsqueeze(0).repeat(num_samples, 1, 1, 1),
                             samples.unsqueeze(-1)), -1)  # (num_samples, B, M, 2)

        B, M = samples.shape[1:3]
        samples = samples.reshape(num_samples*B*M, -1)
        samples = self.project_out(samples)
        samples = samples.reshape(num_samples, B, M, -1)

        return samples.transpose(1, 2)



class ConvCNP(nn.Module):
    """
    ConvCNP works by interpolating context observations onto a grid,
    applying a CNN, and interpolating again to the output queries.

    Args:
        input_interpolation (torch.nn.Module): This module performs the
            input interpolation to pseudo function space.
        convnet (torch.nn.Module): This module is applied in pseudo
            function space, i.e. on the grid interpolation.
        output_interpolation (torch.nn.Module): This module performs the
            interpolation to the requested target queries.
        points_per_unit (int): Use this many grid points per unit interval.
        range_padding (float): Pad the input range by this value.
        grid_divisible_by (int): Grid size will be divisible by this
            number. Will often be required by the 'convnet', e.g. to
            ensure pooling operations can work.

    """

    def __init__(self,
                 input_interpolation,
                 convnet,
                 output_interpolation,
                 points_per_unit=20,
                 range_padding=0.1,
                 grid_divisible_by=None,
                 *args, **kwargs):
        
        super().__init__()

        self.input_interpolation = input_interpolation
        self.convnet = convnet
        self.output_interpolation = output_interpolation
        self.points_per_unit = points_per_unit
        self.range_padding = range_padding
        self.grid_divisible_by = grid_divisible_by

    def forward(self,
                context_in,
                context_out,
                target_in,
                store_rep=False):
        """
        Forward pass in the Convolutional Conditional Neural Process.

        Args:
            context_in (torch.tensor): Shape (N, B, Cin).
            context_out (torch.tensor): Shape (N, B, Cout).
            target_in (torch.tensor): Shape (M, B, Cin).
            store_rep (bool): Store representation.

        Returns:
            torch.tensor: Output of 'output_interpolation'.

        """

        # grid shape is (G, B, Cin)
        grid = make_grid((context_in, target_in),
                         self.points_per_unit,
                         self.range_padding,
                         self.grid_divisible_by)

        # apply interpolation, conv net and final interpolation
        representation = self.input_interpolation(context_in,
                                                  context_out,
                                                  grid,
                                                  store_rep=store_rep)
        representation = representation.permute(1, 2, 0)  # (B, Cr1, G)
        representation = self.convnet(representation)  # (B, Cr2, G)
        representation = representation.permute(2, 0, 1)

        return self.output_interpolation(grid, representation, target_in)

    def sample(self, target_in, num_samples, gp_lambda=None):
        """
        Sample from the Convolutional Conditional Neural Process.

        Args:
            target_in (torch.tensor): Shape (M, B, Cin).
            num_samples (int): Draw this many samples.
            gp_lambda (float): GP covariance will be squeezed by this factor.
                If None, will use the stored attribute.

        Returns:
            torch.tensor: Output of 'output_interpolation'.

        """

        if not hasattr(self.input_interpolation, "sample"):
            raise NotImplementedError("This ConvCNP doesn't use a GP \
                interpolator and can't be sampled from.")

        grid = make_grid(target_in,
                         self.points_per_unit,
                         self.range_padding,
                         self.grid_divisible_by)

        samples = self.input_interpolation.sample(grid, num_samples, gp_lambda)  # (num_samples, G, B, 2)
        samples = samples.permute(0, 2, 3, 1)
        samples = stack_batch(samples)  # (num_samples*B, 2, G)
        samples = self.convnet(samples)  # (num_samples*B, R, G)
        samples = samples.permute(2, 0, 1)
        
        grid = grid.repeat(1, num_samples, 1)
        target_in = target_in.repeat(1, num_samples, 1)

        samples = self.output_interpolation(grid, samples, target_in)  # (M, num_samples*B, Cout)
        samples = unstack_batch(samples.transpose(0, 1), num_samples).transpose(1, 2)

        return samples  # (num_samples, M, B, Cout)