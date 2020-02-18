import os
import torch

from neuralprocess.model import (
    NeuralProcess as NeuralProcessBase,
    AttentiveNeuralProcess as AttentiveNeuralProcessBase,
    ConvCNP as ConvCNPBase,
    generic
)



class NeuralProcess(NeuralProcessBase):

    def __init__(self, function_type="matern"):

        prior_encoder = generic.MLP(
            in_channels=2,
            out_channels=256,
            hidden_channels=128,
            hidden_layers=6,
            activation_op=torch.nn.Tanh,
            activation_kwargs=None,
            bias=True
        )

        decoder = generic.MLP(
            in_channels=129,
            out_channels=2,
            hidden_channels=128,
            hidden_layers=6,
            activation_op=torch.nn.Tanh,
            activation_kwargs=None,
            bias=True
        )

        super().__init__(
            prior_encoder=prior_encoder,
            posterior_encoder=None,
            deterministic_encoder=None,
            decoder=decoder,
            distribution=torch.distributions.Normal,
        )

        model_path = os.path.join(
            os.path.dirname(__file__),
            "pretrained",
            "np_" + function_type + ".pth.tar"
        )

        self.load_state_dict(torch.load(model_path))



class AttentiveNeuralProcess(AttentiveNeuralProcessBase):

    def __init__(self, function_type="matern"):

        prior_encoder = generic.MLP(
            in_channels=2,
            out_channels=256,
            hidden_channels=128,
            hidden_layers=6,
            activation_op=torch.nn.Tanh,
            activation_kwargs=None,
            bias=True
        )

        decoder = generic.MLP(
            in_channels=257,
            out_channels=2,
            hidden_channels=128,
            hidden_layers=6,
            activation_op=torch.nn.Tanh,
            activation_kwargs=None,
            bias=True
        )

        deterministic_encoder = generic.MLP(
            in_channels=2,
            out_channels=128,
            hidden_channels=128,
            hidden_layers=6,
            activation_op=torch.nn.Tanh,
            activation_kwargs=None,
            bias=True
        )

        attention = torch.nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8
        )

        super().__init__(
            prior_encoder=prior_encoder,
            posterior_encoder=None,
            deterministic_encoder=deterministic_encoder,
            decoder=decoder,
            distribution=torch.distributions.Normal,
            attention=attention,
            project_to=128,
            project_bias=True,
            in_channels=1,
            representation_channels=128
        )

        model_path = os.path.join(
            os.path.dirname(__file__),
            "pretrained",
            "anp_" + function_type + ".pth.tar"
        )

        self.load_state_dict(torch.load(model_path))



class ConvCNP(ConvCNPBase):

    def __init__(self, function_type, use_gp=False):

        conv_net = generic.SimpleUNet(
            in_channels=8,
            out_channels=8,
            num_blocks=6,
            input_bypass=True,
            encoding_block_type=generic.ConvNormActivationPool,
            encoding_block_kwargs=dict(
                conv_op=torch.nn.Conv1d,
                conv_kwargs=dict(
                    kernel_size=5,
                    stride=2,
                    padding=2
                ),
                norm_op=None,
                norm_kwargs=None,
                activation_op=torch.nn.ReLU,
                activation_kwargs=dict(
                    inplace=True
                ),
                pool_op=None,
                pool_kwargs=None
            ),
            decoding_block_type=generic.UpsampleConvNormActivation,
            decoding_block_kwargs=dict(
                upsample_op=None,
                upsample_kwargs=None,
                conv_op=torch.nn.ConvTranspose1d,
                conv_kwargs=dict(
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1
                ),
                norm_op=None,
                norm_kwargs=None,
                activation_op=torch.nn.ReLU,
                activation_kwargs=dict(
                    inplace=True
                )
            )
        )

        super().__init__(
            conv_net=conv_net,
            use_gp=use_gp,
            learn_length_scale=True,
            init_length_scale=0.1,
            use_density=True,
            use_density_norm=True,
            points_per_unit=20,
            range_padding=0.1,
            grid_divisible_by=64
        )

        if use_gp:
            name = "gpconvcnp_"
        else:
            name = "convcnp_"

        model_path = os.path.join(
            os.path.dirname(__file__),
            "pretrained",
            name + function_type + ".pth.tar"
        )

        self.load_state_dict(torch.load(model_path))