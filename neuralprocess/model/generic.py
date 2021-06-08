import torch
import torch.nn as nn


class MLP(nn.Sequential):
    """
    A simple multilayer perceptron.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        hidden_channels (int): Number of channels in hidden layers.
            Can also be a list or tuple of length hidden_layers.
        hidden_layers (int): Use this many hidden layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=128,
        hidden_layers=6,
        activation_op=nn.Tanh,
        activation_kwargs=None,
        bias=True,
    ):

        super().__init__()

        if isinstance(hidden_channels, int):
            hidden_channels = [
                hidden_channels,
            ] * hidden_layers
        if activation_kwargs is None:
            activation_kwargs = dict()

        self.add_module("lin0", nn.Linear(in_channels, hidden_channels[0], bias=bias))
        self.add_module("act0", activation_op(**activation_kwargs))
        for h in range(1, hidden_layers):
            self.add_module(
                "lin" + str(h),
                nn.Linear(hidden_channels[h - 1], hidden_channels[h], bias=bias),
            )
            self.add_module("act" + str(h), activation_op(**activation_kwargs))
        self.add_module(
            "lin" + str(hidden_layers),
            nn.Linear(hidden_channels[-1], out_channels, bias=bias),
        )


class ConvNormActivationPool(nn.Sequential):
    """
    A simple block consisting of (Conv, Norm, Activation, Pool) where
    Norm and Pool are optional.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        conv_op (type): The convolution operator.
        conv_kwargs (dict): The convolution arguments.
        norm_op (type): The normalization operator.
        norm_kwargs (dict): The normalization arguments.
        activation_op (type): The activation operator.
        activation_kwargs (dict): The activation arguments.
        pool_op (type): The pooling operator.
        pool_kwargs (dict): The pooling arguments.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_op=nn.Conv1d,
        conv_kwargs=dict(kernel_size=5, stride=2, padding=2),
        norm_op=None,
        norm_kwargs=None,
        activation_op=nn.ReLU,
        activation_kwargs=dict(inplace=True),
        pool_op=None,
        pool_kwargs=None,
        *args,
        **kwargs
    ):

        super().__init__()

        if conv_kwargs is None:
            conv_kwargs = dict()
        self.add_module("conv", conv_op(in_channels, out_channels, **conv_kwargs))

        if norm_op is not None:
            if norm_kwargs is None:
                norm_kwargs = dict()
            self.add_module("norm", norm_op(**norm_kwargs))

        if activation_kwargs is None:
            activation_kwargs = dict()
        self.add_module("activation", activation_op(**activation_kwargs))

        if pool_op is not None:
            if pool_kwargs is None:
                pool_kwargs = dict()
            self.add_module("pool", pool_op(**pool_kwargs))


class UpsampleConvNormActivation(nn.Sequential):
    """
    A simple block consisting of (Upsampling, Conv(Transpose), Norm, Activation)
    where Upsampling and Norm are optional.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        upsample_op (type): The upsampling operator.
        upsample_kwargs (dict): The upsampling arguments.
        conv_op (type): The convolution operator.
        conv_kwargs (dict): The convolution arguments.
        norm_op (type): The normalization operator.
        norm_kwargs (dict): The normalization arguments.
        activation_op (type): The activation operator.
        activation_kwargs (dict): The activation arguments.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        upsample_op=None,
        upsample_kwargs=None,
        conv_op=nn.ConvTranspose1d,
        conv_kwargs=dict(kernel_size=5, stride=2, padding=2, output_padding=1),
        norm_op=None,
        norm_kwargs=None,
        activation_op=nn.ReLU,
        activation_kwargs=dict(inplace=True),
        *args,
        **kwargs
    ):

        super().__init__()

        if upsample_op is not None:
            if upsample_kwargs is None:
                upsample_kwargs = dict()
            self.add_module("upsample", upsample_op(**upsample_kwargs))

        if conv_kwargs is None:
            conv_kwargs = dict()
        self.add_module("conv", conv_op(in_channels, out_channels, **conv_kwargs))

        if norm_op is not None:
            if norm_kwargs is None:
                norm_kwargs = dict()
            self.add_module("norm", norm_op(**norm_kwargs))

        if activation_kwargs is None:
            activation_kwargs = dict()
        self.add_module("activation", activation_op(**activation_kwargs))


class SimpleUNet(nn.Module):
    """
    A simple UNet with a number of identical encoding blocks
    (except for number of channels, which doubles every 2 blocks) and
    symmetric decoding blocks. The defaults are chosen so as to correspond
    to the architecture used for the 1D experiments in the ConvCNP paper.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        num_blocks (int): Use this many blocks for encoding (and also decoding).
        input_bypass (bool): Concatenate input to output.
        encoding_block_type (type): Encoder block, a torch.nn.Module.
        encoding_block_kwargs (dict): Arguments for encoding blocks.
        decoding_block_type (type): Decoder block, a torch.nn.Module.
        decoding_block_kwargs (dict): Arguments for decoding blocks.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=6,
        input_bypass=True,
        encoding_block_type=ConvNormActivationPool,
        encoding_block_kwargs=dict(
            conv_op=nn.Conv1d,
            conv_kwargs=dict(kernel_size=5, stride=2, padding=2),
            norm_op=None,
            norm_kwargs=None,
            activation_op=nn.ReLU,
            activation_kwargs=dict(inplace=True),
            pool_op=None,
            pool_kwargs=None,
        ),
        decoding_block_type=UpsampleConvNormActivation,
        decoding_block_kwargs=dict(
            upsample_op=None,
            upsample_kwargs=None,
            conv_op=nn.ConvTranspose1d,
            conv_kwargs=dict(kernel_size=5, stride=2, padding=2, output_padding=1),
            norm_op=None,
            norm_kwargs=None,
            activation_op=nn.ReLU,
            activation_kwargs=dict(inplace=True),
        ),
        *args,
        **kwargs
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels + int(input_bypass) * in_channels
        self.num_blocks = num_blocks
        self.input_bypass = input_bypass

        self.encoding_blocks = nn.ModuleList()
        for b in range(num_blocks):
            c_in = 2 ** (b // 2) * in_channels
            c_out = 2 ** ((b + 1) // 2) * in_channels
            block = encoding_block_type(c_in, c_out, **encoding_block_kwargs)
            self.encoding_blocks.append(block)

        self.decoding_blocks = nn.ModuleList()
        for b in reversed(range(num_blocks)):
            c_in = 2 ** ((b + 1) // 2) * in_channels
            if b + 1 < num_blocks:
                c_in *= 2
            if b > 0:
                c_out = 2 ** (b // 2) * in_channels
            else:
                c_out = out_channels
            block = decoding_block_type(c_in, c_out, **decoding_block_kwargs)
            self.decoding_blocks.append(block)

    def forward(self, x):
        """
        Forward pass through the convolutional structure.

        Args:
            x (torch.tensor): Input of shape (B, Cin, ...).

        Returns:
            torch.tensor: Output of shape (B, Cout, ...).

        """

        skips = [x]

        for b in range(self.num_blocks):
            x = self.encoding_blocks[b](x)
            if (b + 1) < self.num_blocks:
                skips.append(x)

        for b in range(self.num_blocks):
            if b > 0:
                x = torch.cat((x, skips[-b]), 1)
            x = self.decoding_blocks[b](x)

        if self.input_bypass:
            x = torch.cat((x, skips[0]), 1)

        return x