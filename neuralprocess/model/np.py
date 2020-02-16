import torch
import torch.nn as nn

from neuralprocess.util import tensor_to_loc_scale, stack_batch, unstack_batch



class NeuralProcess(nn.Module):
    """
    A Neural Process implementation. A few design choices to consider:

        - The model inputs are mostly sets, which are expected to be of
            shape (B, |set|, C, ...).
        - You have to initalize the different submodules (encoders and
            decoder) manually in advance. It is your responsibility to
            ensure they have matching input and output sizes.
        - Either 'prior_encoder' or 'deterministic_encoder' must exist.
        - The submodules are expected to take and return tensors.
            Splitting into loc and scale to define distributions will be
            done by this class (except for the output).

    Args:
        prior_encoder (torch.nn.Module): Encoder for the prior.
        posterior_encoder (torch.nn.Module): Optional posterior encoder.
            If not available, will default to prior encoder.
        deterministic_encoder (torch.nn.Module): Encoder for the
            deterministic path.
        decoder (torch.nn.Module): Decoder (required). Make sure the
            decoder accommodates the correct input size depending on
            availability of prior and deterministic representation.
        distribution (type): A loc-scale distribution that is a subclass
            of torch.distributions.Distribution.

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
            raise ValueError("Need at least one of 'prior_encoder' and \
                'deterministic_encoder', but both are None.")
        if decoder is None:
            raise ValueError("'decoder' must not be None.")
        if not issubclass(distribution, torch.distributions.Distribution):
            raise TypeError("'distribution' must be a subclass of \
                torch.distributions.Distribution.")

        self.prior_encoder = prior_encoder
        if posterior_encoder is not None:
            self.posterior_encoder = posterior_encoder
        else:
            self.posterior_encoder = prior_encoder
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
        """
        Aggregate representations. This implementation averages along
        first axis.

        Args:
            representations (torch.tensor): Multiple representations
                stacked along first axis

        Returns:
            torch.tensor: Average representation.

        """

        return representations.mean(1)

    def encode_prior(self,
                     context_in,
                     context_out,
                     target_in=None):
        """
        Use the 'prior_encoder' to encode a prior distribution

        Args:
            context_in  (torch.tensor): Shape (B, N, Cin, ...).
            context_out (torch.tensor): Shape (B, N, Cout, ...).
            target_in (torch.tensor): Ignored. Only to have consistent
                signature when more complex children need to use it.

        Returns:
            torch.distributions.Distribution: Output of 'decoder'

        """

        B, N = context_in.shape[:2]

        context_in = stack_batch(context_in)
        context_out = stack_batch(context_out)

        prior_rep = self.prior_encoder(torch.cat((context_in, context_out), 1))
        prior_rep = unstack_batch(prior_rep, B)
        prior_rep = self.aggregate(prior_rep)

        self.prior = tensor_to_loc_scale(
            prior_rep, self.distribution, logvar_transform=True)

        return self.prior

    def encode_representation(self,
                              context_in,
                              context_out,
                              target_in=None,
                              store_rep=False):
        """
        Use the 'deterministic_encoder' to encode a representation

        Args:
            context_in  (torch.tensor): Shape (B, N, Cin, ...).
            context_out (torch.tensor): Shape (B, N, Cout, ...).
            target_in (torch.tensor): Ignored. Only to have consistent
                signature when more complex children need to use it.
            store_rep (bool): Store representation.

        Returns:
            torch.tensor: Deterministic representation

        """

        B, N = context_in.shape[:2]

        context_in = stack_batch(context_in)
        context_out = stack_batch(context_out)

        rep = self.deterministic_encoder(torch.cat((context_in, context_out), 1))
        rep = unstack_batch(rep, B)
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

        context = stack_batch(context)

        posterior_rep = self.posterior_encoder(context)
        posterior_rep = unstack_batch(posterior_rep, B)
        posterior_rep = self.aggregate(posterior_rep)
        
        self.posterior = tensor_to_loc_scale(
            posterior_rep, self.distribution, logvar_transform=True)

        return self.posterior

    def decode(self, target_in, sample, representation):
        """
        Make predictions for a target input, given a sample and a
        representation.

        Args:
            target_in (torch.tensor): Shape (B, M, Cin, ...).
            sample (torch.tensor): Shape (B, Rs, ...).
            representation (torch.tensor): Shape (B, Rr, ...).

        Returns:
            torch.tensor: Output of 'decoder'

        """

        if sample is None and representation is None:

            return self.decoder(target_in)

        else:

            concat = [target_in, ]

            if sample is not None:

                # broadcast global sample
                if sample.ndim < target_in.ndim:
                    sample = sample.unsqueeze(1)
                    repeats = [1, target_in.shape[1]] + [1,] * (sample.ndim-2)
                    sample = sample.repeat(*repeats)
                concat.append(sample)

            if representation is not None:

                # broadcast global representation
                if representation.ndim < target_in.ndim:
                    representation = representation.unsqueeze(1)
                    repeats = [1, target_in.shape[1]] + [1,] * (representation.ndim-2)
                    representation = representation.repeat(*repeats)
                concat.append(representation)

            return self.decoder(torch.cat(concat, 2))

    
    def forward(self,
                context_in,
                context_out,
                target_in,
                target_out=None,
                store_rep=False):
        """
        Forward pass in the Neural Process. 'prior' (and 'posterior'
        during training) will be stored automatically, 'representation'
        is optional.

        Args:
            context_in (torch.tensor): Shape (B, N, Cin, ...).
            context_out (torch.tensor): Shape (B, N, Cout, ...).
            target_in (torch.tensor): Shape (B, M, Cin, ...).
            target_out (torch.tensor): Shape (B, M, Cout, ...).
            store_rep (bool): Store deterministic representation.

        Returns:
            torch.tensor: Output of 'decoder'

        """

        if self.deterministic_encoder is not None:
            representation = self.encode_representation(
                context_in, context_out, target_in, store_rep=store_rep)
        else:
            representation = None
        
        if self.prior_encoder is not None:
            self.encode_prior(context_in, context_out, target_in)
            if self.training:
                self.encode_posterior(
                    context_in, context_out, target_in, target_out)
                sample = self.sample_posterior()
            else:
                sample = self.sample_prior(mean=True)
        else:
            sample = None

        return self.decode(target_in, sample, representation)

    def sample_posterior(self):
        """
        Get a random sample from the posterior. Will use
        reparametrization trick during training.
        
        Returns:
            torch.tensor: The sample.
            
        """

        if self.posterior is None:
            raise ValueError("'posterior' is None. Please use \
                'encode_posterior' first.")
        if self.training:
            sample = self.posterior.rsample()
        else:
            sample = self.posterior.sample()
        return sample

    def sample_prior(self, mean=False):
        """
        Get a random sample from the prior.

        Args:
            mean (bool): Return the prior mean.
        
        Returns:
            torch.tensor: The sample.
            
        """

        if self.prior is None:
            raise ValueError("'prior' is None. Please use 'encode_prior' first.")
        elif mean:
            sample = self.prior.loc
        else:
            sample = self.prior.sample()
        return sample
    
    def sample(self, target_in, n=1, from_posterior=False, to_cpu=True):
        """
        Get prediction samples. Will automatically use eval mode. Make
        sure to manually call 'encode_representation' if the deterministic
        representation should be used, as the method can also work with
        it being None.

        Args:
            target_in (torch.tensor): Shape (B, M, Cin, ...).
            n (int): Get this many samples.
            from_posterior (bool): Sample from posterior instead of prior.
            to_cpu (bool): Move predictions to CPU immediately.

        Returns:
            torch.tensor: Stacked samples of shape (n, B, M, Cout, ...).

        """
        
        if n < 1:
            raise ValueError("n must be >= 1, but is {}.".format(n))
            
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
                    s = self.sample_posterior()
                else:
                    s = self.sample_prior(mean=False)
                s = self.decode(target_in, s, self.representation)
                if to_cpu:
                    s = s.cpu()
                samples.append(s)

            if old_state:
                self.train()

            return torch.stack(samples)