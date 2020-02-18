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