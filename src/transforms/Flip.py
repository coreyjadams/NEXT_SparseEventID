import torch
import torch.nn as nn
import sparseconvnet as scn

class Flip(nn.Module):

    def __init__(self, flip_axes = None, flip_random = False):

        '''
        Flip (reflect) the image around x/z/y or combination of axes.

        If flip_axes is set, flip will be deterministic.  
        flip_axes may be an integer, or a sequence of integers.

        If flip_random is True, flip_axes is ignored and a random flip is used
        
        If flip_random is True, the same flip is used on every tensor.

        '''

        if flip_axes is None and flip_random == False:
            raise Exception("Must provide either flip_axes or set flip_random to true")

        if flip_axes is not None:

            self.flip_axes = flip_axes
            return 
        if flip_random = True:

            self.flip_axes = self.generate_flip()


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

        with torch.nograd():
            return inputs