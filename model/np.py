import torch
import torch.nn as nn

from neuralprocess.util import tensor_to_loc_scale



class NeuralProcess(nn.Module):
    """
    """

    def __init__(self,
                 prior_encoder=None,
                 posterior_encoder=None,
                 deterministic_encoder=None,
                 decoder=None,
                 distribution=torch.distributions.Normal,
                 *args, **kwargs):
        
        super().__init__()

        # check minimum requirements
        if prior_encoder is None and deterministic_encoder is None:
            raise ValueError("Need at least one of 'prior_encoder' and 'deterministic_encoder', but both are None.")
        if decoder is None:
            raise ValueError("'decoder' must not be None.")
        if not issubclass(distribution, torch.distributions.Distribution):
            raise TypeError("'distribution' must be a subclass of torch.distributions.Distribution.")

        self.prior_encoder = prior_encoder
        self.posterior_encoder = posterior_encoder if posterior_encoder is not None else prior_encoder
        self.deterministic_encoder = deterministic_encoder
        self.decoder = decoder
        self.distribution = distribution

        self.reset()

    def reset(self):
        """Set all latent representations to None."""

        self.prior = None
        self.posterior = None
        self.representation = None

    def aggregate(self, representations):
        """Aggregate representations. This implementation averages along first axis.

        Args:
            representations (torch.tensor): Multiple representations stacked along first axis

        Returns:
            torch.tensor: Average representation.

        """

        return representations.mean(1)

    def encode_prior(self, context_in, context_out):
        """
        Use the 'prior_encoder' to encode a prior distribution

        Args:
            context_in  (torch.tensor): Shape (B, N, Cin, ...).
            context_out (torch.tensor): Shape (B, N, Cout, ...).

        Returns:
            torch.distributions.Distribution: Output of 'decoder'

        """

        B, N = context_in.shape[:2]

        # stack along batch axis
        context_in = context_in.reshape(B*N, *context_in.shape[2:])
        context_out = context_out.reshape(B*N, *context_out.shape[2:])

        prior_rep = self.prior_encoder(torch.cat((context_in, context_out), 1))
        prior_rep = prior_rep.reshape(B, N, *prior_rep.shape[1:])
        prior_rep = self.aggregate(prior_rep)

        self.prior = tensor_to_loc_scale(prior_rep, self.distribution, logvar_transform=True)

        return self.prior

    def encode_representation(self, context_in, context_out, store_rep=False):
        """
        Use the 'deterministic_encoder' to encode a representation

        Args:
            context_in  (torch.tensor): Shape (B, N, Cin, ...).
            context_out (torch.tensor): Shape (B, N, Cout, ...).
            store_rep   (bool):         Store representation.

        Returns:
            torch.tensor: Deterministic representation

        """

        B, N = context_in.shape[:2]

        # stack along batch axis
        context_in = context_in.reshape(B*N, *context_in.shape[2:])
        context_out = context_out.reshape(B*N, *context_out.shape[2:])

        rep = self.deterministic_encoder(torch.cat((context_in, context_out), 1))
        rep = prior_rep.reshape(B, N, *prior_rep.shape[1:])
        rep = self.aggregate(rep)
        if store_rep:
            self.representation = rep

        return rep
    
    def encode_posterior(self, context_in, context_out, target_in, target_out):
        """
        Use the 'posterior_encoder' to encode a posterior distribution

        Args:
            context_in  (torch.tensor): Shape (B, N, Cin, ...).
            context_out (torch.tensor): Shape (B, N, Cout, ...).
            target_in   (torch.tensor): Shape (B, M, Cin, ...).
            target_out  (torch.tensor): Shape (B, M, Cout, ...).

        Returns:
            torch.distributions.Distribution: Output of 'decoder'

        """
        
        B, N = context_in.shape[:2]
        M = target_in.shape[1]

        context = torch.cat((context_in, context_out), 2)
        target = torch.cat((target_in, target_out), 2)
        context = torch.cat((context, target), 1)

        context = context.reshape(B*(N+M), *context.shape[2:])

        posterior_rep = self.posterior_encoder(context)
        posterior_rep = posterior_rep.reshape(B, (N+M), *posterior_rep.shape[1:])
        posterior_rep = self.aggregate(posterior_rep)
        
        self.posterior = tensor_to_loc_scale(posterior_rep, self.distribution, logvar_transform=True)

        return self.posterior
    
    def forward(self, context_in, context_out, target_in, target_out=None, store_rep=False):
        """
        Forward pass in the Neural Process. 'prior' (and 'posterior' during training) will be stored automatically,
        'representation' is optional.

        Args:
            context_in  (torch.tensor): Shape (B, N, Cin, ...).
            context_out (torch.tensor): Shape (B, N, Cout, ...).
            target_in   (torch.tensor): Shape (B, M, Cin, ...).
            target_out  (torch.tensor): Shape (B, M, Cout, ...).
            store_rep   (bool):         Store deterministic representation.

        Returns:
            torch.tensor: Output of 'decoder'

        """

        if hasattr(self, "deterministic_encoder"):
            representation = self.encode_representation(context_in, context_out, target_in, store_rep=store_rep)
        else:
            representation = None
        
        if hasattr(self, "prior_encoder"):
            self.encode_prior(context_in, context_out, target_in)
            if self.training:
                self.encode_posterior(context_in, context_out, target_in, target_out)
                sample = self.sample_posterior()
            else:
                sample = self.sample_prior(mean=True)
        else:
            sample = None

        return self.decode(target_in, sample, representation)

    def sample_posterior(self):
        """If the posterior is not a torch.distribution, you should reimplement this!"""

        if self.distribution is None:
            return self.posterior
        elif self.training:
            sample = self.posterior.rsample()
        else:
            sample = self.posterior.sample()
        return sample

    def sample_prior(self, mean=True):
        """If the prior is not a torch.distribution, you should reimplement this!"""

        if self.distribution is None:
            sample = self.prior
        elif mean:
            sample = self.prior.loc
        else:
            sample = self.prior.sample()
        return sample

    def reconstruct(self, target_in, sample=None):

        if sample is None and self.posterior is None:
            raise ValueError("Reconstruction requires either a sample or a posterior.")

        if sample is None:
            sample = self.posterior.loc

        return self.decoder(target_in, sample, self.representation)
    
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

            samples = []
            while len(samples) < n:
                if from_posterior:
                    samples.append(self.decoder(target_in, self.sample_posterior(), self.representation))
                else:
                    samples.append(self.decoder(target_in, self.sample_prior(mean=False), self.representation))

            if old_state:
                self.train()

            # we try to stack
            try:
                return torch.stack(samples)
            except TypeError:
                return samples