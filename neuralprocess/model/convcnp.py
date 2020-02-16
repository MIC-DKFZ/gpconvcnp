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



def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.

    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple



def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.

    Args:
        model (:class:`nn.Sequential`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.

    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias)
    return model


def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape `(batch, n, 1)`.
        y (tensor): Inputs of shape `(batch, m, 1)`.

    Returns:
        tensor: Pair-wise distances of shape `(batch, n, m)`.
    """
    assert x.shape[2] == 1 and y.shape[2] == 1, \
        'The inputs x and y must be 1-dimensional observations.'
    return (x - y.permute(0, 2, 1)) ** 2



def init_layer_weights(layer):
    """Initialize the weights of a :class:`nn.Layer` using Glorot
    initialization.

    Args:
        layer (:class:`nn.Module`): Single dense or convolutional layer from
            :mod:`torch.nn`.

    Returns:
        :class:`nn.Module`: Single dense or convolutional layer with
            initialized weights.
    """
    nn.init.xavier_normal_(layer.weight, gain=1)
    nn.init.constant_(layer.bias, 1e-3)



def custom_init(m):
    """Custom initialization used in ConvCNP."""

    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, 1.)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, 0.)
    elif type(m) in (nn.Conv1d, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight, 1.)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, 1e-3)



def pad_concat(t1, t2):
    """Concat the activations of two layer channel-wise by padding the layer
    with fewer points with zeros.

    Args:
        t1 (tensor): Activations from first layers of shape `(batch, n1, c1)`.
        t2 (tensor): Activations from second layers of shape `(batch, n2, c2)`.

    Returns:
        tensor: Concatenated activations of both layers of shape
            `(batch, max(n1, n2), c1 + c2)`.
    """
    if t1.shape[2] > t2.shape[2]:
        padding = t1.shape[2] - t2.shape[2]
        if padding % 2 == 0:  # Even difference
            t2 = F.pad(t2, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t2 = F.pad(t2, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')
    elif t2.shape[2] > t1.shape[2]:
        padding = t2.shape[2] - t1.shape[2]
        if padding % 2 == 0:  # Even difference
            t1 = F.pad(t1, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t1 = F.pad(t1, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')

    return torch.cat([t1, t2], dim=1)




class UNet(nn.Module):
    """Large convolutional architecture from 1d experiments in the paper.
    This is a 12-layer residual network with skip connections implemented by
    concatenation.

    Args:
        in_channels (int, optional): Number of channels on the input to
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8):
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = 16
        self.num_halving_layers = 6

        self.l1 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l2 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l3 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l4 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l5 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l6 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=8 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l8 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l9 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=2 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l10 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=2 * self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l11 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l12 = nn.ConvTranspose1d(in_channels=2 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        return pad_concat(x, h12)



# class ConvDeepSet(nn.Module):
#     """One-dimensional set convolution layer. Uses an RBF kernel for
#     `psi(x, x')`.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         learn_length_scale (bool): Learn the length scales of the channels.
#         init_length_scale (float): Initial value for the length scale.
#         use_density (bool, optional): Append density channel to inputs.
#             Defaults to `True`.
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  learn_length_scale,
#                  init_length_scale,
#                  use_density=True):
#         super(ConvDeepSet, self).__init__()
#         self.out_channels = out_channels
#         self.use_density = use_density
#         self.in_channels = in_channels + 1 if self.use_density else in_channels
#         self.g = self.build_weight_model()
#         self.sigma = nn.Parameter(np.log(init_length_scale) *
#                                   torch.ones(self.in_channels),
#                                   requires_grad=learn_length_scale)
#         self.sigma_fn = torch.exp

#     def build_weight_model(self):
#         """Returns a function point-wise function that transforms the
#         `in_channels + 1`-dimensional representation to dimensionality
#         `out_channels`.

#         Returns:
#             :class:`torch.nn.Module`: Linear layer applied point-wise to
#                 channels.
#         """
#         model = nn.Sequential(
#             nn.Linear(self.in_channels, self.out_channels),
#         )
#         init_sequential_weights(model)
#         return model

#     def forward(self, x, y, t, *args, **kwargs):
#         """Forward pass through the layer with evaluations at locations `t`.

#         Args:
#             x (tensor): Inputs of observations of shape `(n, 1)`.
#             y (tensor): Outputs of observations of shape `(n, in_channels)`.
#             t (tensor): Inputs to evaluate function at of shape `(m, 1)`.

#         Returns:
#             tensor: Outputs of evaluated function at `z` of shape
#                 `(m, out_channels)`.
#         """
#         # Ensure that `x`, `y`, and `t` are rank-3 tensors.
#         if len(x.shape) == 2:
#             x = x.unsqueeze(2)
#         if len(y.shape) == 2:
#             y = y.unsqueeze(2)
#         if len(t.shape) == 2:
#             t = t.unsqueeze(2)

#         # Compute shapes.
#         batch_size = x.shape[0]
#         n_in = x.shape[1]
#         n_out = t.shape[1]

#         # Compute the pairwise distances.
#         # Shape: (batch, n_in, n_out).
#         dists = compute_dists(x, t)

#         # Compute the weights.
#         # Shape: (batch, n_in, n_out, in_channels).
#         wt = self.rbf(dists)

#         if self.use_density:
#             # Compute the extra density channel.
#             # Shape: (batch, n_in, 1).
#             density = torch.ones(batch_size, n_in, 1).to(device)

#             # Concatenate the channel.
#             # Shape: (batch, n_in, in_channels).
#             y_out = torch.cat([density, y], dim=2)
#         else:
#             y_out = y

#         # Perform the weighting.
#         # Shape: (batch, n_in, n_out, in_channels).
#         y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

#         # Sum over the inputs.
#         # Shape: (batch, n_out, in_channels).
#         y_out = y_out.sum(1)

#         if self.use_density:
#             # Use density channel to normalize convolution
#             density, conv = y_out[..., :1], y_out[..., 1:]
#             normalized_conv = conv / (density + 1e-8)
#             y_out = torch.cat((density, normalized_conv), dim=-1)

#         # Apply the point-wise function.
#         # Shape: (batch, n_out, out_channels).
#         y_out = y_out.view(batch_size * n_out, self.in_channels)
#         y_out = self.g(y_out)
#         y_out = y_out.view(batch_size, n_out, self.out_channels)

#         return y_out

#     def rbf(self, dists):
#         """Compute the RBF values for the distances using the correct length
#         scales.

#         Args:
#             dists (tensor): Pair-wise distances between `x` and `t`.

#         Returns:
#             tensor: Evaluation of `psi(x, t)` with `psi` an RBF kernel.
#         """
#         # Compute the RBF kernel, broadcasting appropriately.
#         scales = self.sigma_fn(self.sigma)[None, None, None, :]
#         a, b, c = dists.shape
#         return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)



# class GPConvDeepSet(nn.Module):
#     """One-dimensional set convolution layer. Uses an RBF kernel for
#     `psi(x, x')`.

#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of output channels.
#         learn_length_scale (bool): Learn the length scales of the channels.
#         init_length_scale (float): Initial value for the length scale.
#         use_density (bool, optional): Append density channel to inputs.
#             Defaults to `True`.
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  learn_length_scale,
#                  init_length_scale,
#                  use_density=True,
#                  gp_posterior_from_samples=0,
#                  gp_samples_alpha=0.1,
#                  init_gp_noise=1e-5,
#                  gp_density_norm=True,
#                  gp_learnable_noise=False):

#         super().__init__()

#         self.out_channels = out_channels
#         self.use_density = use_density
#         self.in_channels = in_channels + 1 if self.use_density else in_channels

#         self.projection = nn.Linear(self.in_channels, self.out_channels)
#         self.projection.apply(custom_init)

#         self.sigma = nn.Parameter(np.log(init_length_scale) *
#                                   torch.ones(self.in_channels),
#                                   requires_grad=learn_length_scale)

#         self.gp_noise = nn.Parameter(torch.tensor(init_gp_noise), requires_grad=gp_learnable_noise)
#         self.sigma_fn = torch.exp
#         self.gp_posterior_from_samples = gp_posterior_from_samples
#         self.gp_density_norm = gp_density_norm
#         self.gp_samples_alpha = gp_samples_alpha

#         self.representation = None
#         self.last_prediction = None

#     def build_weight_model(self):
#         """Returns a function point-wise function that transforms the
#         `in_channels + 1`-dimensional representation to dimensionality
#         `out_channels`.

#         Returns:
#             :class:`torch.nn.Module`: Linear layer applied point-wise to
#                 channels.
#         """
#         model = nn.Sequential(
#             nn.Linear(self.in_channels, self.out_channels),
#         )
#         init_sequential_weights(model)
#         return model

#     def forward(self, x, y, t, store_rep=False):
#         """Forward pass through the layer with evaluations at locations `t`.

#         Args:
#             x (tensor): Inputs of observations of shape `(n, 1)`.
#             y (tensor): Outputs of observations of shape `(n, in_channels)`.
#             t (tensor): Inputs to evaluate function at of shape `(m, 1)`.

#         Returns:
#             tensor: Outputs of evaluated function at `z` of shape
#                 `(m, out_channels)`.
#         """
#         # Ensure that `x`, `y`, and `t` are rank-3 tensors.
#         if len(x.shape) == 2:
#             x = x.unsqueeze(2)
#         if len(y.shape) == 2:
#             y = y.unsqueeze(2)
#         if len(t.shape) == 2:
#             t = t.unsqueeze(2)

#         # Compute shapes.
#         batch_size = x.shape[0]
#         n_in = x.shape[1]
#         n_out = t.shape[1]

#         # # try GP ------------------------------------------
#         # # x has shape (B, N, 1)
#         # # t has shape (B, M, 1)
#         K = torch.exp(-0.5*torch.pow(x - x.transpose(1, 2), 2) / self.sigma_fn(self.sigma[-1])**2)  # (B, N, N)
#         # dists = torch.pow(x - x.transpose(1, 2), 2)
#         # K = self.rbf(dists).sum(-1)
#         # L = torch.cholesky(K + self.gp_noise * torch.eye(K.shape[-1]).to(device=K.device)[None, :, :])  # (B, N, N)
#         L = gpytorch.utils.cholesky.psd_safe_cholesky(K)
#         dists = torch.pow(x - t.transpose(1, 2), 2)
#         # K_s = self.rbf(dists).sum(-1)
#         K_s = torch.exp(-0.5*dists / self.sigma_fn(self.sigma[-1])**2)  # (B, N, M)
#         L_k, _ = torch.solve(K_s, L)  # (B, N, M)
#         y_out = torch.bmm(L_k.transpose(1, 2), torch.solve(y, L)[0])  # (B, M, 1)
        
#         if self.use_density:  # (B, N, M, 1)
#             weights = torch.exp(-0.5*dists / self.sigma_fn(self.sigma[0])**2).unsqueeze(-1)
#             density = torch.ones(batch_size, 1, 1, 1).to(dists.device) * weights
#             density = torch.sum(density, 1)  # (B, M, 1)
#             if self.gp_density_norm:
#                 y_out /= (density + 1e-8)
#             y_out = torch.cat((density, y_out), -1)

#         if store_rep:
#             self.representation = L_k
#             self.last_prediction = y_out

#         # Apply the point-wise function.
#         # Shape: (batch, n_out, out_channels).
#         y_out = y_out.view(batch_size * n_out, self.in_channels)
#         y_out = self.projection(y_out)
#         y_out = y_out.view(batch_size, n_out, self.out_channels)

#         return y_out

#     def sample(self, target_in, num_samples, gp_lambda=0.2):
#         """Draw GP samples for target_in.

#         Args:
#             target_in (torch.tensor): New input values, shape (B, M, 1).
#             num_samples (int): Draw this many samples.
#             gp_lambda (float): GP covariance will be squeezed by this factor.

#         Returns:
#             torch.tensor: Output values at target_in, shape (num_samples, B, M, Cout).

#         """

#         if self.representation is None or self.last_prediction is None:
#             raise ValueError("Please run a forward pass with store_rep=True!")

#         if target_in.shape[1] != self.representation.shape[-1]:
#             raise IndexError("target_in shape is {}, representation shape is {}. \
#                 Second axis of target_in and last axis of representation should \
#                 match!".format(target_in.shape, self.representation.shape))

#         mu = self.last_prediction
#         if self.use_density:
#             density, mu = mu[..., :1], mu[..., 1:]  # (B, M, 1)
#         L_k = self.representation  # (B, N, M)

#         # dists = torch.pow(target_in - target_in.transpose(1, 2), 2)
#         # K_ss = self.rbf(dists).sum(-1)
#         K_ss = torch.exp(-0.5*torch.pow(target_in - target_in.transpose(1, 2), 2) / self.sigma_fn(self.sigma[-1])**2)  # (B, M, M)
#         # L_ss = torch.cholesky(K_ss + self.gp_noise * torch.eye(K_ss.shape[-1]).to(device=K_ss.device)[None, :, :] - torch.bmm(L_k.transpose(1, 2), L_k))  # (B, M, M)
#         L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(K_ss - torch.bmm(L_k.transpose(1, 2), L_k))

#         # THIS CURRENTLY ONLY WORKS FOR IN_CHANNELS=1 !!!
#         samples = mu + gp_lambda * torch.bmm(L_ss, torch.randn(mu.shape[0], L_ss.shape[-1], num_samples).to(device=L_ss.device))  # (B, M, num_samples)
#         samples = samples.transpose(1, 2).transpose(0, 1)  # (num_samples, B, M)
#         samples = torch.cat((density.unsqueeze(0).repeat(num_samples, 1, 1, 1), samples.unsqueeze(-1)), -1)  # (num_samples, B, M, in_channels)
#         B, M = samples.shape[1:3]
#         samples = samples.view(num_samples*B*M, self.in_channels)
#         samples = self.projection(samples)
#         samples = samples.view(num_samples, B, M, self.out_channels)

#         return samples

#     def rbf(self, dists):
#         """Compute the RBF values for the distances using the correct length
#         scales.

#         Args:
#             dists (tensor): Pair-wise distances between `x` and `t`.

#         Returns:
#             tensor: Evaluation of `psi(x, t)` with `psi` an RBF kernel.
#         """
#         # Compute the RBF kernel, broadcasting appropriately.
#         scales = self.sigma_fn(self.sigma)[None, None, None, :]
#         a, b, c = dists.shape
#         return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)



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

    def __init__(self,
                 in_channels,
                 out_channels,
                 learn_length_scale=True,
                 init_length_scale=1.,
                 use_density=True,
                 use_density_norm=True):

        super(ConvDeepSet, self).__init__()

        self.out_channels = out_channels
        self.use_density = use_density
        self.use_density_norm = use_density_norm
        self.in_channels = in_channels + 1 if self.use_density else in_channels
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels),
                                  requires_grad=learn_length_scale)
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

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.representation = None
        self.last_prediction = None

    def forward(self,
                context_in,
                context_out,
                target_in,
                store_rep=False,
                *args, **kwargs):
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
        K = torch.exp(-0.5 * dists_in / self.sigma_fn(self.sigma[-1])**2)
        L = gpytorch.utils.cholesky.psd_safe_cholesky(K)  # (B, N, N)

        # Do the same for context_in and target_in
        dists = torch.pow(context_in.unsqueeze(2) - target_in.unsqueeze(1), 2)
        dists = dists.sum(-1)
        K_s = torch.exp(-0.5 * dists / self.sigma_fn(self.sigma[-1])**2)
        L_k, _ = torch.solve(K_s, L)  # (B, N, M)

        # Compute mean prediction
        # Shape (B, M, 1)
        output = torch.bmm(L_k.transpose(1, 2), torch.solve(context_out, L)[0])

        # Append density channel if required
        if self.use_density:
            weights = torch.exp(-0.5 * dists / self.sigma_fn(self.sigma[0])**2)
            density = torch.ones(B, N, M).to(dists.device) * weights
            density = torch.sum(density, 1).unsqueeze(-1)  # (B, M, 1)
            if self.use_density_norm:
                output /= (density + 1e-8)
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

    def sample(self, target_in, num_samples, gp_lambda=0.2):
        """Draw GP samples for target_in.

        Args:
            target_in (torch.tensor): New input values, shape (B, M, 1).
            num_samples (int): Draw this many samples.
            gp_lambda (float): GP covariance will be squeezed by this factor.

        Returns:
            torch.tensor: Output values at target_in, shape (num_samples, B, M, Cout).

        """

        if self.representation is None or self.last_prediction is None:
            raise ValueError("Please run a forward pass with store_rep=True!")

        if target_in.shape[1] != self.representation.shape[-1]:
            raise IndexError("target_in shape is {}, representation shape is {}. \
                Second axis of target_in and last axis of representation should \
                match!".format(target_x.shape, self.representation.shape))

        mu = self.last_prediction
        if self.use_density:
            density, mu = mu[..., :1], mu[..., 1:]  # (B, M, 1)
        L_k = self.representation  # (B, N, M)

        dists = torch.pow(target_in.unsqueeze(2) - target_in.unsqueeze(1), 2)
        dists = dists.sum(-1)  # (B, M, M)
        K_ss = torch.exp(-0.5 * dists / self.sigma_fn(self.sigma[-1])**2)
        # K_ss = self.rbf(dists).sum(-1)  # (B, M, M)
        K_ss -= torch.bmm(L_k.transpose(1, 2), L_k)
        # We're allowing a little more jitter for the cholesky decomp,
        # because this part is relatively unstable.
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
        samples = samples.view(num_samples*B*M, -1)
        samples = self.projection(samples)
        samples = samples.view(num_samples, B, M, -1)

        return samples



class ConvCNP(nn.Module):
    """One-dimensional ConvCNP model.

    Args:
        learn_length_scale (bool): Learn the length scale.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
        architecture (:class:`nn.Module`): Convolutional architecture to place
            on functional representation (rho).
    """

    def __init__(self,
                 use_gp=False,
                 points_per_unit=20,
                 learn_length_scale=True,
                 init_length_scale=1.,
                 use_density=True,
                 use_density_norm=True,
                 *args, **kwargs):

        super(ConvCNP, self).__init__()

        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.conv_net = UNet()
        self.multiplier = 2 ** self.conv_net.num_halving_layers

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        # init_length_scale = 2.0 / self.points_per_unit

        if use_gp:
            self.l0 = GPConvDeepSet(
                in_channels=1,
                out_channels=self.conv_net.in_channels,
                learn_length_scale=learn_length_scale,
                init_length_scale=init_length_scale,
                use_density=True,
                use_density_norm=False
            )
        else:
            self.l0 = ConvDeepSet(
                in_channels=1,
                out_channels=self.conv_net.in_channels,
                learn_length_scale=learn_length_scale,
                init_length_scale=init_length_scale,
                use_density=True
            )
        self.mean_layer = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )
        self.sigma_layer = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )

    def forward(self, x, y, x_out, y_out=None, store_rep=False, *args, **kwargs):
        """Run the model forward.

        Args:
            x (tensor): Observation locations of shape
                `(batch, data, features)`.
            y (tensor): Observation values of shape
                `(batch, data, outputs)`.
            x_out (tensor): Locations of outputs of shape
                `(batch, data, features)`.
        Returns:
            tuple[tensor]: Means and standard deviations of shape
                `(batch_out, channels_out)`.
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        if len(y.shape) == 2:
            y = y.unsqueeze(2)
        if len(x_out.shape) == 2:
            x_out = x_out.unsqueeze(2)

        # Determine the grid on which to evaluate functional representation.
        x_min = min(torch.min(x).cpu().numpy(),
                    torch.min(x_out).cpu().numpy()) - 0.1
        x_max = max(torch.max(x).cpu().numpy(),
                    torch.max(x_out).cpu().numpy()) + 0.1
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multiplier))
        x_grid = torch.linspace(x_min, x_max, num_points).to(x.device)
        x_grid = x_grid[None, :, None].repeat(x.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.l0(x, y, x_grid, store_rep=store_rep))  # (B, gridsize, C)
        h = h.permute(0, 2, 1)  # (N, C, gridsize)
        h = h.reshape(h.shape[0], h.shape[1], num_points)  # (N, C, gridsize)
        h = self.conv_net(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations.
        mean = self.mean_layer(x_grid, h, x_out)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))

        return torch.cat((mean, sigma), -1)

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])

    def sample(self, target_x, num_samples, gp_lambda=0.2):

        x_min = target_x.min().item() - 0.1
        x_max = target_x.max().item() + 0.1
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multiplier))
        x_grid = torch.linspace(x_min, x_max, num_points).to(target_x.device)
        x_grid = x_grid[None, :, None].repeat(target_x.shape[0], 1, 1)

        samples = self.l0.sample(x_grid, num_samples, gp_lambda=gp_lambda)  # (num_samples, B, M, l0.out_channels)
        samples = self.activation(samples)
        num_samples, B, M, C = samples.shape
        samples = samples.view(num_samples*B, M, C)
        samples = samples.permute(0, 2, 1)
        samples = self.conv_net(samples)
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1).permute(0, 2, 1)  # (num_samples, B, conv_net.out_channels)
        samples = samples.contiguous()

        means = self.mean_layer(x_grid.repeat(num_samples, 1, 1), samples, target_x.repeat(num_samples, 1, 1))
        means = means.view(num_samples, B, *means.shape[1:])

        sigmas = self.sigma_fn(self.sigma_layer(x_grid.repeat(num_samples, 1, 1), samples, target_x.repeat(num_samples, 1, 1)))
        sigmas = sigmas.view(num_samples, B, *sigmas.shape[1:])

        return torch.cat((means, sigmas), -1)



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
#                  use_density=True,
#                  use_density_norm=True,
#                  project_to=0,
#                  project_bias=True,
#                  project_in_channels=1,
#                  **kwargs):

#         super().__init__()
        
#         if type(kernel) == type:
#             if kernel_kwargs is None:
#                 kernel_kwargs = dict()
#             self.kernel = kernel(**kernel_kwargs)
#         else:
#             self.kernel = kernel
#         self.use_density = use_density
#         self.use_density_norm = use_density_norm
#         self.project_to = project_to
#         self.project_bias = project_bias
#         self.project_in_channels = project_in_channels

#         if self.project_to not in (0, None):
#             self.setup_projections()

#     def setup_projections(self):
#         """Set up modules that project to another space."""

#         self.project_out = nn.Linear(self.project_in_channels,
#                                      self.project_to,
#                                      bias=self.project_bias)

#     def forward(self, context_in, context_out, target_in, *args, **kwargs):
#         """Interpolate context onto new input range.

#         Args:
#             context_in (torch.tensor): Context inputs, shape (N, B, Cin).
#             context_out (torch.tensor): Context outputs, shape (N, B, Cout).
#             target_in (torch.tensor): New input values, shape (M, B, Cin).

#         Returns:
#             torch.tensor: Output values at new input positions,
#                 optionally projected. Shape (M, B, R).

#         """

#         N, B = context_in.shape[:2]
#         M = target_in.shape[0]

#         if self.use_density:
#             density = torch.ones(N, B, 1)
#             density = density.to(dtype=context_in.dtype, device=context_in.device)
#             context_out = torch.cat((density, context_out), -1)

#         context_in = context_in.transpose(0, 1).unsqueeze(1)  # (B, 1, M, Cin)
#         target_in = target_in.transpose(0, 1).unsqueeze(1)  # (B, 1, M, Cin)
#         K = self.kernel(context_in, target_in).evaluate().permute(0, 2, 3, 1)  # (B, N, M, Cout)

#         output = context_out.transpose(0, 1).unsqueeze(-2) * K
#         output = output.sum(1)  # (B, M, Cout)
#         if self.use_density and self.use_density_norm:
#             output[..., 1:] /= (output[..., :1] + 1e-8)

#         if hasattr(self, "project_out"):
#             output = stack_batch(output)
#             output = self.project_out(output)
#             output = unstack_batch(output, B)

#         return output.transpose(0, 1)



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

#         super().__init__(*args, **kwargs)

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
#             context_in (torch.tensor): Context inputs, shape (N, B, Cin).
#             context_out (torch.tensor): Context outputs, shape (N, B, 1).
#             target_in (torch.tensor): New input values, shape (M, B, Cin).
#             store_rep (bool): Store cholesky decomposition of the kernel
#                 matrix as well as mean prediction, so we can sample at a
#                 later point.

#         Returns:
#             torch.tensor: Output values at new input positions,
#                 optionally projected. Shape (M, B, R).

#         """

#         N, B = context_in.shape[:2]
#         M = target_in.shape[0]

#         context_in = context_in.transpose(0, 1)
#         context_out = context_out.transpose(0, 1)
#         target_in = target_in.transpose(0, 1)

#         # do GP inference manually, consider replacing with GPyTorch
#         K = self.kernel(context_in, context_in).evaluate()  # (B, N, N)
#         L = gpytorch.utils.cholesky.psd_safe_cholesky(K, jitter=1e-4)  # (B, N, N)
#         K_s = self.kernel(context_in, target_in).evaluate()  # (B, N, M)
#         L_k, _ = torch.solve(K_s, L)  # (B, N, M)
#         output = torch.bmm(L_k.transpose(1, 2), torch.solve(context_out, L)[0])  # (B, M, 1)

#         if self.gp_sample_from_posterior:
#             K_ss = self.kernel(target_in, target_in).evaluate()  # (B, M, M)
#             K_ss -= torch.bmm(L_k.transpose(1, 2), L_k)
#             L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(K_ss)
#             samples = torch.randn(output.shape[0],
#                                   L_ss.shape[-1],
#                                   self.gp_sample_from_posterior)
#             samples = samples.to(device=L_ss.device)
#             samples = torch.bmm(L_ss, samples)
#             output = output + self.gp_lambda * samples  # (B, M, num_samples)
#             output = output.mean(-1, keepdims=True)  # (B, M, 1)
        
#         if self.use_density:
#             density = K_s.unsqueeze(-1).mean(1)  # (B, M, 1)
#             if self.use_density_norm:
#                 output /= (density + 1e-8)
#             output = torch.cat((density, output), -1)  

#         if store_rep:
#             self.representation = L_k
#             self.last_prediction = output  

#         if hasattr(self, "project_out"):
#             output = stack_batch(output)
#             output = self.project_out(output)
#             output = unstack_batch(output, B)

#         return output.transpose(0, 1)

#     def sample(self, target_in, num_samples, gp_lambda=None):
#         """Draw GP samples for target_in.

#         Args:
#             target_in (torch.tensor): New input values, shape (M, B, Cin).
#             num_samples (int): Draw this many samples.
#             gp_lambda (float): GP covariance will be squeezed by this factor.
#                 If None, will use the stored attribute.

#         Returns:
#             torch.tensor: Output values at new input positions,
#                 optionally projected. Shape (num_samples, M, B, R).

#         """

#         if self.representation is None or self.last_prediction is None:
#             raise ValueError("No stored kernel and prediction, please run \
#                 a forward pass with store_rep=True.")

#         if target_in.shape[0] != self.representation.shape[-1]:
#             raise IndexError("target_in shape is {}, representation shape is {}. \
#                 First axis of target_x and last axis of representation should match!"\
#                 .format(target_in.shape, self.representation.shape))

#         if gp_lambda is None:
#             gp_lambda = self.gp_lambda

#         target_in = target_in.transpose(0, 1)  # (B, M, Cin)
#         density, mu = self.last_prediction[..., :1], self.last_prediction[..., 1:]  # (B, M, 1)
#         L_k = self.representation  # (B, N, M)

#         K_ss = self.kernel(target_in, target_in).evaluate()  # (B, M, M)
#         K_ss -= torch.bmm(L_k.transpose(1, 2), L_k)
#         L_ss = gpytorch.utils.cholesky.psd_safe_cholesky(K_ss)

#         samples = torch.randn(mu.shape[0],
#                               L_ss.shape[-1],
#                               num_samples)
#         samples = samples.to(device=L_ss.device)
#         samples = torch.bmm(L_ss, samples)
#         samples = mu + gp_lambda * samples  # (B, M, num_samples)

#         samples = samples.permute(2, 0, 1)  # (num_samples, B, M)
#         samples = torch.cat((density.unsqueeze(0).repeat(num_samples, 1, 1, 1),
#                              samples.unsqueeze(-1)), -1)  # (num_samples, B, M, 2)

#         B, M = samples.shape[1:3]
#         samples = samples.reshape(num_samples*B*M, -1)
#         samples = self.project_out(samples)
#         samples = samples.reshape(num_samples, B, M, -1)

#         return samples.transpose(1, 2)



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
#                  convnet,
#                  output_interpolation,
#                  points_per_unit=20,
#                  range_padding=0.1,
#                  grid_divisible_by=None,
#                  *args, **kwargs):
        
#         super().__init__()

#         self.input_interpolation = input_interpolation
#         self.convnet = convnet
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
#             context_in (torch.tensor): Shape (N, B, Cin).
#             context_out (torch.tensor): Shape (N, B, Cout).
#             target_in (torch.tensor): Shape (M, B, Cin).
#             target_out (torch.tensor): Unused.
#             store_rep (bool): Store representation.

#         Returns:
#             torch.tensor: Output of 'output_interpolation'.

#         """

#         # grid shape is (G, B, Cin)
#         grid = make_grid((context_in, target_in),
#                          self.points_per_unit,
#                          self.range_padding,
#                          self.grid_divisible_by)

#         # apply interpolation, conv net and final interpolation
#         representation = self.input_interpolation(context_in,
#                                                   context_out,
#                                                   grid,
#                                                   store_rep=store_rep)
#         representation = representation.permute(1, 2, 0)  # (B, Cr1, G)
#         representation = self.convnet(representation)  # (B, Cr2, G)
#         representation = representation.permute(2, 0, 1)

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

#         grid = make_grid(target_in,
#                          self.points_per_unit,
#                          self.range_padding,
#                          self.grid_divisible_by)

#         samples = self.input_interpolation.sample(grid, num_samples, gp_lambda)  # (num_samples, G, B, 2)
#         samples = samples.permute(0, 2, 3, 1)
#         samples = stack_batch(samples)  # (num_samples*B, 2, G)
#         samples = self.convnet(samples)  # (num_samples*B, R, G)
#         samples = samples.permute(2, 0, 1)
        
#         grid = grid.repeat(1, num_samples, 1)
#         target_in = target_in.repeat(1, num_samples, 1)

#         samples = self.output_interpolation(grid, samples, target_in)  # (M, num_samples*B, Cout)
#         samples = unstack_batch(samples.transpose(0, 1), num_samples).transpose(1, 2)

#         return samples  # (num_samples, M, B, Cout)