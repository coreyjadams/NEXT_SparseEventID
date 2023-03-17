import torch
import torch.nn as nn
import numpy

# Import SCN if available
try:    import sparseconvnet as scn
except: pass

class Shift(nn.Module):

    def __init__(self, max_shift, sparse=False):

        '''
        Shift (reflect) the image around x/z/y or combination of axes.

        If shift_axes is set, Shift will be deterministic.
        shift_axes may be an integer, or a sequence of integers.

        If flip_random is True, shift_axes is ignored and a random Shift is used

        If flip_random is True, the same Shift is used on every tensor.

        '''
        super().__init__()

        self.sparse = sparse
        self.max_shift = max_shift


    def forward(self, inputs):
        '''
        Inputs is assumed to be a Sparse Tensor from SCN.
        '''

        # First, get the spatial size:

        with torch.no_grad():
            if self.sparse:
                spatial_size = inputs.spatial_size
                spatial_locs = inputs.get_spatial_locations()

                # Generate a set of shifts for the batch:
                shifts = torch.randint(
                    low=-self.max_shift, high=self.max_shift, size=(3,) )

                batch_size = len(torch.unique(spatial_locs[:,-1]))

                for d, shift in enumerate(shifts):
                    proposed_locs = spatial_locs[:,d] + shift
                    if torch.min(proposed_locs) >= 0 and torch.max(proposed_locs) < spatial_size[d]:
                        spatial_locs[:,d] = proposed_locs

                return scn.InputLayer(dimension = 3, spatial_size = spatial_size)((
                    spatial_locs,
                    inputs.features,
                    batch_size,
                ))

            else:
                shifts = numpy.random.randint(
                    low=-self.max_shift, high=self.max_shift, size=(3,) )
                shifts = [ s for s in shifts ]
                return torch.roll(inputs, shifts, (0,1,2))
