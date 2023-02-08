import torch
import torch.nn as nn

# Import SCN if available
try:    import sparseconvnet as scn
except: pass

class Flip(nn.Module):

    def __init__(self, flip_axes = None, flip_random = False, sparse=False):

        '''
        Flip (reflect) the image around x/z/y or combination of axes.

        If flip_axes is set, flip will be deterministic.
        flip_axes may be an integer, or a sequence of integers.

        If flip_random is True, flip_axes is ignored and a random flip is used

        If flip_random is True, the same flip is used on every tensor.

        '''
        super().__init__()

        self.sparse = sparse

        if flip_axes is None and flip_random == False:
            raise Exception("Must provide either flip_axes or set flip_random to true")

        if flip_axes is not None:

            self.flip_axes = flip_axes
            return
        if flip_random == True:
            self.flip_axes = None


    def generate_flip(self):

        # This bool determines if we flip x/y/z
        flip_xyz = torch.randint(0,2,size=(3,)).to(dtype=torch.bool)

        # Convert this to axes:
        return torch.where(flip_xyz)

    def forward(self, inputs):
        '''
        Inputs is assumed to be a Sparse Tensor from SCN.
        '''

        # First, get the spatial size:

        with torch.no_grad():
            if self.sparse:
                spatial_size = inputs.spatial_size
                spatial_locs = inputs.get_spatial_locations()


                # Sparse tensor has shape (N, d+1) where N is the total
                # filled locs and d is the dimension.

                # Flipping a sparse tensor is a matter of subtracting the value
                # from the max value, if the flip is active.
                # For example, if spatial size is [100, 200, 300]
                # Then to flip the Y axis only (200), do:
                # new_spatial_locs_y = 200 - spatial_locs[:,1]
                # spatial_locs[:,1] = new_spatial_locs

                batch_size = len(torch.unique(spatial_locs[:,-1]))

                if self.flip_axes is None:
                    flip_axes = self.generate_flip()
                else:
                    flip_axes = self.flip_axes
                for d in flip_axes:
                    M = spatial_size[d]
                    spatial_locs[:,d] = M - spatial_locs[:,d]

                return scn.InputLayer(dimension = 3, spatial_size = spatial_size)((
                    spatial_locs,
                    inputs.features,
                    batch_size,
                ))

            else:
                if self.flip_axes is None:
                    flip_axes = self.generate_flip()
                else:
                    flip_axes = self.flip_axes
                return torch.flip(inputs, flip_axes)
