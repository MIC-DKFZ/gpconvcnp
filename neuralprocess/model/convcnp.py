import torch
import torch.nn as nn
import gpytorch

from neuralprocess.util import stack_batch, unstack_batch, make_grid



class ConvDeepSet(nn.Module):
    """
    Use a kernel to interpolate values onto a grid and optionally project
    to a new space.

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

    def forward(self, context_in, context_out, grid, *args, **kwargs):
        """Interpolate context onto grid.

        Args:
            context_in (torch.tensor): Context inputs, shape (N, B, Cin).
            context_out (torch.tensor): Context outputs, shape (N, B, Cout).
            grid (torch.tensor): Grid input values, shape (G, B, Cin).

        Returns:
            torch.tensor: Output values at grid positions,
                optionally projected. Shape (G, B, R).

        """

        N, B = context_in.shape[:2]
        G = grid.shape[0]

        if self.use_density:
            density = torch.ones(N, B, 1)
            density = density.to(dtype=context_in.dtype, device=context_in.device)
            context_out = torch.cat((density, context_out), -1)

        context_in = context_in.transpose(0, 1).unsqueeze(1)  # (B, 1, G, Cin)
        grid = grid.transpose(0, 1).unsqueeze(1)  # (B, 1, G, Cin)
        K = self.kernel(context_in, grid).evaluate().permute(0, 2, 3, 1)  # (B, N, G, Cout)

        output = context_out.transpose(0, 1).unsqueeze(-2) * K
        output = output.sum(1)  # (B, G, Cout)
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

    def forward(self, context_in, context_out, grid, *args, **kwargs):
        """Interpolate context onto grid.

        Args:
            context_in (torch.tensor): Context inputs, shape (N, B, Cin).
            context_out (torch.tensor): Context outputs, shape (N, B, 1).
            grid (torch.tensor): Grid input values, shape (G, B, Cin).

        Returns:
            torch.tensor: Output values at grid positions,
                optionally projected. Shape (G, B, R).

        """

        N, B = context_in.shape[:2]
        G = grid.shape[0]

        context_in = context_in.transpose(0, 1)
        context_out = context_out.transpose(0, 1)
        grid = grid.transpose(0, 1)

        # do GP inference manually, consider replacing with GPyTorch
        K = self.kernel(context_in, context_in).evaluate()  # (B, N, N)
        L = gpytorch.utils.cholesky.psd_safe_cholesky(K)  # (B, N, N)
        K_s = self.kernel(context_in, grid).evaluate()  # (B, N, G)
        L_k, _ = torch.solve(K_s, L)  # (B, N, G)
        output = torch.bmm(L_k.transpose(1, 2), torch.solve(context_out, L)[0])  # (B, G, 1)

        if self.gp_sample_from_posterior:
            K_ss = self.kernel(grid, grid).evaluate()  # (B, G, G)
            K_ss -= torch.bmm(L_k.transpose(1, 2), L_k)
            L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(K_ss)
            samples = torch.randn(output.shape[0],
                                  L_ss.shape[-1],
                                  self.gp_sample_from_posterior)
            samples = samples.to(device=L_ss.device)
            samples = torch.bmm(L_ss, samples)
            output = output + self.gp_lambda * samples  # (B, G, num_samples)
            output = output.mean(-1, keepdims=True)
        
        if self.use_density:  # (B, N, M, 1)
            density = K_s.unsqueeze(-1).mean(1)  # (B, G, 1)
            if self.use_density_norm:
                output /= (density + 1e-8)
            output = torch.cat((density, output), -1)

        if hasattr(self, "project_out"):
            output = stack_batch(output)
            output = self.project_out(output)
            output = unstack_batch(output, B)

        return output.transpose(0, 1)




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
        representation = self.input_interpolation(context_in, context_out, grid)
        representation = representation.permute(1, 2, 0)  # (B, Cr1, G)
        representation = self.convnet(representation)  # (B, Cr2, G)
        representation = representation.permute(2, 0, 1)
        return self.output_interpolation(grid, representation, target_in)