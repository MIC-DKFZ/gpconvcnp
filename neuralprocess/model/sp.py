import torch
import torch.nn as nn

from neuralprocess.util import tensor_to_loc_scale, stack_batch, unstack_batch, match_shapes



class SegmentationProcess(nn.Module):
    """

    Args:
        context_encoder (torch.nn.Module): Encoder for the context.
        decoder (torch.nn.Module): Decoder. Make sure the
            decoder accommodates the correct input size depending on
            availability of context and target representations.
        target_encoder (torch.nn.Module): Encoder for the target images.
            Only necessary when working with queries that contain images.

    """

    def __init__(self,
                 context_encoder,
                 decoder,
                 target_encoder=None,
                 *args, **kwargs):
        
        super().__init__()

        self.context_encoder = context_encoder
        self.decoder = decoder
        self.target_encoder = target_encoder

    def aggregate(self, context_query, target_query, context_representation, target_representation=None):
        """
        Aggregate representations. This implementation averages along
        first axis.

        Args:
            context_query (torch.tensor): Shape (B, N, Cq).
                Not used in this implementation. Will be necessary
                for attention based aggregation.
            target_query (torch.tensor): Shape (B, M, Cq).
            context_representation (torch.tensor): Shape (B, N, Ci, ...).
                Can also be a list or tuple of tensors with varying
                number of channels and spatial size.
            target_representation (torch.tensor): Shape (B, M, Ci, ...).
                Can also be a list or tuple of tensors with varying
                number of channels and spatial size. Only used when
                query images need to be encoded.

        Returns:
            torch.tensor: Average representation.
                Will be a list if input is a list or tuple.

        """

        if torch.is_tensor(context_representation):
            context_representation = [context_representation, ]
        if torch.is_tensor(target_representation):
            target_representation = [target_representation, ]
        if target_representation is None:
            target_representation = [None, ] * len(context_representation)

        context_representation = [r.mean(1, keepdim=True) for r in context_representation]

        for r, rep in enumerate(context_representation[:-1]):
            concat = match_shapes(rep, target_representation[r], target_query, ignore_axes=2)
            if target_representation[r] is None:
                concat = concat[:1]
            else:
                concat = concat[:2]
            context_representation[r] = torch.cat(concat, 2)

        context_representation[-1] = torch.cat(
            match_shapes(context_representation[-1], target_representation[-1], target_query, ignore_axes=2),
            2
        )

        return context_representation

    def encode_context(self, context_query, context_seg, context_image=None):
        """
        Use the context encoder to encode a representation.

        Args:
            context_query (torch.tensor): Shape (B, N, Cq)
            context_seg (torch.tensor): Shape (B, N, Cs, ...)
            context_image (torch.tensor): Shape (B, N, Cimg, ...)

        Returns:
            torch.tensor: Shape (B, N, Cr, ...). Can also be a list!

        """

        B = context_query.shape[0]

        input_ = torch.cat(match_shapes(context_query, context_seg, context_image, ignore_axes=2), 2)
        input_ = stack_batch(input_)
        output = self.context_encoder(input_)
        if torch.is_tensor(output):
            output = unstack_batch(output, B)
        else:
            output = [unstack_batch(o, B) for o in output]
        return output

    def encode_target(self, target_query, target_image):
        """
        Use the target encoder to encode a representation.

        Args:
            target_query (torch.tensor): Shape (B, M, Cq)
            target_image (torch.tensor): Shape (B, N, Cimg, ...)

        Returns:
            torch.tensor: Shape (B, N, Cr, ...). Can also be a list!

        """

        if self.target_encoder is None:
            raise ValueError("target_encoder is None, so we can't encode anything!")

        B = target_query.shape[0]

        input_ = torch.cat(match_shapes(target_query, target_image, ignore_axes=2), 2)
        input_ = stack_batch(input_)
        output = self.target_encoder(input_)
        if torch.is_tensor(output):
            output = unstack_batch(output, B)
        else:
            output = [unstack_batch(o, B) for o in output]
        return output

    def decode(self, representation):
        """
        Decode an aggregated representation.

        Args:
            representation (torch.tensor): Shape (B, M, Cr, ...).

        Returns:
            torch.tensor: Output of 'decoder', shape (B, M, Cout, ...).

        """

        if torch.is_tensor(representation):
            representation = [representation, ]

        B = representation[0].shape[0]
        representation = [stack_batch(r) for r in representation]
        representation = self.decoder(*representation)
        return unstack_batch(representation, B)
    
    def forward(self,
                context_query,
                context_seg,
                target_query,
                context_image=None,
                target_image=None):
        """
        Forward pass in the Segmentation Process.

        Args:
            context_query (torch.tensor): Shape (B, N, Cq).
            context_seg (torch.tensor): Shape (B, N, Cs, ...).
            target_query (torch.tensor): Shape (B, M, Cq).
            context_image (torch.tensor): Shape (B, N, Cimg, ...).
            target_image (torch.tensor): Shape (B, M, Cimg, ...).

        Returns:
            torch.tensor: Output of 'decoder', shape (B, M, Cout, ...)

        """

        # encode context
        # returns a tuple of tensors with shape (B, N, Ci, ...)
        context_representation = self.encode_context(context_query,
                                                     context_seg,
                                                     context_image)

        # encode target if there's something to encode
        # returns a tuple of tensors with shape (B, M, Ci, ...)
        if target_image is not None:
            target_representation = self.encode_target(target_query,
                                                       target_image)
        else:
            target_representation = None

        # aggregate (i.e. create the input for the decoder)
        context_representation = self.aggregate(context_query,
                                                target_query,
                                                context_representation,
                                                target_representation)

        # decode
        return self.decode(context_representation)



class AttentiveSegmentationProcess(SegmentationProcess):

    def __init__(self, attention, global_sum=True, **kwargs):

        super().__init__(*args, **kwargs)

        self.attention = attention
        self.global_sum = global_sum

    def aggregate(self, context_query, target_query, context_representation, target_representation=None):
        """
        Aggregate representations. This implementation uses attention over queries/keys.

        Args:
            context_query (torch.tensor): Shape (B, N, Cq).
                These are the keys for the attention mechanism.
            target_query (torch.tensor): Shape (B, M, Cq).
                These are the queries for the attention mechanism.
            context_representation (torch.tensor): Shape (B, N, Ci, ...).
                These are the values for the attention mechanism.
                Can also be a list or tuple of tensors with varying
                number of channels and spatial size.
            target_representation (torch.tensor): Shape (B, M, Ci, ...).
                Will just be concatenated to the output in this implementation.

        Returns:
            torch.tensor: Average representation.
                Will be a list if input is a list or tuple.

        """

        if torch.is_tensor(context_representation):
            context_representation = [context_representation, ]
        if torch.is_tensor(target_representation):
            target_representation = [target_representation, ]
        if target_representation is None:
            target_representation = [None, ] * len(context_representation)

        for r, rep in enumerate(context_representation[:-1]):
            concat = match_shapes(rep, target_representation[r], target_query, ignore_axes=2)
            if target_representation[r] is None:
                concat = concat[:1]
            else:
                concat = concat[:2]
            context_representation[r] = torch.cat(concat, 2)

        # either use summation or attention in the lowest layer
        if self.global_sum:
            context_representation[-1] = context_representation[-1].mean(1, keepdim=True)
            context_representation[-1] = torch.cat(
                match_shapes(context_representation[-1], target_representation[-1], target_query, ignore_axes=2),
                2
            )
        else:
            pass

        return context_representation