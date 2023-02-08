import torch
import torch.nn as nn

# Import SCN if available
try:    import sparseconvnet as scn
except: pass

from . Flip     import Flip
from . Rescale  import Rescale
from . Shift    import Shift

def build_transforms(args, image_size=None, include_identity=True):

    transforms = []

    # If it's sparse, we include the input layer:
    if args.framework.sparse:
        input_layer = scn.InputLayer(
            dimension    = 3, 
            spatial_size = torch.tensor(image_size)
        )
    else:
        input_layer = torch.nn.Identity()


    if args.data.transform1:
        t = nn.Sequential(
            input_layer,
            Flip(
                flip_axes=None,
                flip_random = True,
                sparse=args.framework.sparse
            ),
            Rescale(
                image_sum = 100.,
                sparse=args.framework.sparse
            ),
            Shift(
                max_shift= 5,
                sparse=args.framework.sparse
            )
        )
        transforms.append(t)
    if args.data.transform2:
        t = nn.Sequential(
            input_layer,
            Flip(
                flip_axes=None,
                flip_random = True,
                sparse=args.framework.sparse
            ),
            Rescale(
                image_sum = 100.,
                sparse=args.framework.sparse
            ),
            Shift(
                max_shift= 5,
                sparse=args.framework.sparse
            )

        )
        transforms.append(t)

    # If no transforms, add the input/identity:
    if len(transforms) == 0 or include_identity:
        transforms.append(input_layer)

    return transforms