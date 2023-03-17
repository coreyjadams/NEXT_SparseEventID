import torch
import torch.nn as nn

# Import SCN if available
try:    import sparseconvnet as scn
except: pass

class Rescale(nn.Module):

    def __init__(self, image_sum =1.0, shift_max=0.25, sparse=True):
        super().__init__()
        '''
        Rescale the features in the image.  The image is scaled
        such that the sum of all active pixels is 1.0, and then gaussian noise is added.

        '''


        self.image_sum  = image_sum
        self.shift_max  = shift_max
        self.sparse     = sparse

        self.pool   = None
        self.unpool = None
    def forward(self, inputs):
        '''
        Inputs is assumed to be a Sparse Tensor from SCN.
        '''

        # First, get the spatial size:

        with torch.no_grad():
            if self.sparse:

                spatial_size = inputs.spatial_size
                if self.pool is None:
                    self.pool = scn.AveragePooling(
                        dimension = 3,
                        pool_size = spatial_size,
                        pool_stride = 1)
                if self.unpool is None:
                    self.unpool = scn.UnPooling(
                        dimension = 3,
                        pool_size = spatial_size,
                        pool_stride = 1)

                # To compute the average properly for sparse
                # tensors, we pool:

                original_features = inputs.features
                n_voxels = torch.prod(spatial_size)

                pooled = self.pool(inputs)

                # This average includes all of the zeros.
                # So, it has essentially been divide by the spatial size.
                # if we multiple by the size of the space,
                # we get back to the summed value
                pooled.features *= n_voxels

                # Now, the pooled features contains the sum of all pixels in the
                # event.  We'll fix this, but we want to scale up and down
                # all the events randomly, by +/- 25%.  So, draw from a gaussian
                # with the same shape as the pooled features,
                # scale the pooled features, and go from there.


                # Rand like is uniform in 0.0 to 1.0
                # We want a small number close to 1.0
                rand_scale = (1.0 - 0.5*self.shift_max) + self.shift_max * torch.rand_like(pooled.features)
                pooled.features = rand_scale * pooled.features

                unpooled = self.unpool(pooled)

                # Unpooling maps the average (per image) back to it's original
                # location.  So, it destroys all local information.

                #  If we want the image to sum to 100., we divide:

                random_norm = torch

                unpooled.features = self.image_sum*original_features / unpooled.features


                return unpooled


            else:

                # Create a random number for every voxel:
                rand_scale = (1.0 - 0.5*self.shift_max) + self.shift_max * torch.rand_like(inputs)
                return inputs * rand_scale
