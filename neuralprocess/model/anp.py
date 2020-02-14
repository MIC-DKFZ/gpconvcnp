import torch
import torch.nn as nn

from neuralprocess.model import NeuralProcess
from neuralprocess.util import tensor_to_loc_scale, stack_batch, unstack_batch



class AttentiveNeuralProcess(NeuralProcess):
    """
    A Neural Process where the deterministic path learns an attention
    mechanism. We optionally project context, target and representation
    to the embedding dimension of the attention mechanism. If the inputs
    have spatial dimensions, you need to adjust 'in_channels' and
    'representation_channels' to include those, because the attention
    mechanism is assumed to be unable to handle those.
    
    Args:
        attention (torch.nn.Module): Attention mechanism. This should
            support a __call__(query, key, value) signature, where the
            inputs have shapes (N/M, B, Cx, ...). If inputs are required
            to have the same C (like torch.nn.MultiheadAttention) you
            need to provide project_to=C to set up linear mappings to
            that size.
        project_to (int): Project everything to this channel size
            before attention mechanism is applied. 0 means no projection.
        project_bias (bool): Allow bias in projections.
        in_channels (int): Input size for query and key projections.
        representation_channels (int): Input size for value projection.
    
    """

    def __init__(self,
                 attention,
                 project_to=0,
                 project_bias=True,
                 in_channels=1,
                 representation_channels=128,
                 **kwargs):

        super().__init__(**kwargs)

        self.attention = attention
        self.project_to = project_to
        self.project_bias = project_bias
        self.in_channels = in_channels
        self.representation_channels = representation_channels

        if self.project_to not in (0, None):
            self.setup_projections()

    def reset(self):

        super().reset()
        self.attention_weights = None

    def setup_projections(self):
        """Set up modules that project to a common channel size."""

        self.project_query = nn.Linear(self.in_channels,
                                       self.project_to,
                                       bias=self.project_bias)
        self.project_key = nn.Linear(self.in_channels,
                                     self.project_to,
                                     bias=self.project_bias)
        self.project_value = nn.Linear(self.representation_channels,
                                       self.project_to,
                                       bias=self.project_bias)

    def encode_representation(self,
                              context_in,
                              context_out,
                              target_in,
                              store_rep=False):
        """
        Use the 'deterministic_encoder' and the 'attention' mechanism
        to encode a deterministic representation.

        Args:
            context_in (torch.tensor): Shape (N, B, Cin, ...).
            context_out (torch.tensor): Shape (N, B, Cout, ...).
            target_in (torch.tensor): Shape (M, B, Cin, ...).
            store_rep (bool): Store representation.

        Returns:
            torch.tensor: Deterministic representation, shape (M, B, R, ...).

        """

        N = context_in.shape[0]
        M = target_in.shape[0]
        S = tuple(target_in.shape[3:])
                        
        encoder_input = torch.cat(
            (stack_batch(context_in), stack_batch(context_out)), 1)
        representations = self.deterministic_encoder(encoder_input)
        representations = unstack_batch(representations, N)

        # get rid of spatial dimensions for attention
        target_in = target_in.reshape(*target_in.shape[:2], -1)
        context_in = context_in.reshape(*context_in.shape[:2], -1)
        representations = representations.reshape(*representations.shape[:2], -1)

        if self.project_to not in (0, None):
            target_in = self.project_query(stack_batch(target_in))
            target_in = unstack_batch(target_in, M)
            context_in = self.project_key(stack_batch(context_in))
            context_in = unstack_batch(context_in, N)
            representations = self.project_value(stack_batch(representations))
            representations = unstack_batch(representations, N)

        attention_output = self.attention(target_in, context_in, representations)
        if isinstance(attention_output, (tuple, list)):
            attention_output, attention_weights = attention_output
        else:
            attention_weights = None

        # attention_output (M, B, embed_dim)
        # attention_weights (B, M, N)
        # broadcast to appropriate spatial shape
        if len(S) > 0:
            for s in S:
                attention_output.unsqueeze_(-1)
            attention_output = attention_output.repeat(1, 1, 1, *S)

        if store_rep:
            self.representation = attention_output
            self.attention_weights = attention_weights

        return attention_output