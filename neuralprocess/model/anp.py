import torch
import torch.nn as nn

from neuralprocess.model import NeuralProcess



class AttentiveNeuralProcess(NeuralProcess):
    """
    A Neural Process where the deterministic path learns an attention
    mechanism.
    
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
            context_in  (torch.tensor): Shape (N, B, Cin, ...).
            context_out (torch.tensor): Shape (N, B, Cout, ...).
            target_in (torch.tensor): Shape (M, B, Cin, ...).
            store_rep (bool): Store representation.

        Returns:
            torch.tensor: Deterministic representation, shape (M, B, R, ...).

        """

        N = context_in.shape[0]
        M = target_in.shape[0]
                        
        encoder_input = torch.cat(
            (stack_batch(context_in), stack_batch(context_out)), 1)
        representations = self.deterministic_encoder(encoder_input)
        representations = representations.reshape(N, B, *representations.shape[1:])

        if self.project_to not in (0, None):
            query = unstack_batch(self.project_query(stack_batch(target_in)), M)
            key = unstack_batch(self.project_key(stack_batch(context_in)), N)
            value = unstack_batch(self.project_value(stack_batch(representations)), N)
        else:
            query = target_in
            key = context_in
            value = representations

        attention_output = self.attention(query, key, value)
        if isinstance(attention_output, (tuple, list)):
            attention_output, attention_weights = attention_output
        else:
            attention_weights = None

        # WHATS THE OUTPUT SHAPE HERE ???

        # SAVE REP

        # RETURN

    def forward(self, context_in, context_out, target_in, target_out=None, store_rep=False):
        
        # latent path
        if hasattr(self, "prior_encoder"):
            self.encode_prior(context_in, context_out, target_in)
            if self.training:
                self.encode_posterior(context_in, context_out, target_in, target_out)
                sample = self.sample_posterior()
            else:
                sample = self.sample_prior(mean=True)  # (B, Cr)
        else:
            sample = None

        # deterministic path
        representation, representation_weights = self.encode_representation(context_in, context_out, target_in, store_rep)  # (B, Nquery, embed_dim)

        # representation (B, Nquery, embed_dim)
        # sample (B, Cr)
        # target_in (B, Nquery, Cq)
        
        if self.pass_weights:
            return self.decoder(target_in, sample, (representation, representation_weights))
        else:
            return self.decoder(target_in, sample, representation)

    def sample(self, target_in, n=1, from_posterior=False):
        
        if n < 1:
            raise ValueError("n must be <= 1")
            
        if self.prior is None:
            raise ValueError("Please encode prior first.")
            
        if from_posterior and self.posterior is None:
            raise ValueError("Please encode posterior first.")

        with torch.no_grad():

            old_state = self.training
            self.eval()

            if hasattr(self, "representation_weights") and self.representation_weights is not None and self.pass_weights:
                if isinstance(self.representation_weights, (tuple, list)):
                    rep = [(self.representation[r], self.representation_weights[r]) for r in range(len(self.representation_weights))]
                else:
                    rep = (self.representation, self.representation_weights)
            else:
                rep = self.representation

            samples = []
            while len(samples) < n:
                if from_posterior:
                    samples.append(self.decoder(target_in, self.sample_posterior(), rep))
                else:
                    samples.append(self.decoder(target_in, self.sample_prior(mean=False), rep))

            if old_state:
                self.train()

            # we try to stack
            try:
                return torch.stack(samples)
            except TypeError:
                return samples