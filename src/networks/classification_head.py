import numpy
import torch

from . resnet import Encoder



def build_networks(params, input_shape):


    if params.framework.sparse:
        import sparseconvnet as scn
        from . sparse_building_blocks import Block, ResidualBlock
        from . sparse_building_blocks import ConvolutionDownsample
        from . sparse_building_blocks import BlockSeries
    else:
        from . building_blocks import Block, ResidualBlock, BlockSeries
        from . building_blocks import ConvolutionDownsample
        from . building_blocks import MaxPooling


    resnet = Encoder(params, input_shape)

    output_shape = resnet.output_shape

    # resnet, output_shape = create_resnet(params, input_shape)

    classification_head = torch.nn.Sequential()
    current_number_of_filters = output_shape[0]

    # We apply a pooling layer to the image:
    if params.framework.sparse:
        classification_head.append(scn.AveragePooling(
            dimension = 3,
            pool_size = output_shape[1:],
            pool_stride = 1
            )
        )
        classification_head.append(scn.SparseToDense(
            dimension=3, nPlanes=current_number_of_filters))

    else:
        classification_head.append(
            torch.nn.AvgPool3d(output_shape[1:])
        )

    classification_head.append(torch.nn.Flatten())

    for i, layer in enumerate(params.head.layers):
        classification_head.append(torch.nn.Linear(
            in_features  = current_number_of_filters,
            out_features = layer))
        if i < len(params.head.layers) - 1:
            classification_head.append(torch.nn.ReLU())
        current_number_of_filters = layer

    classification_head.append(torch.nn.Tanh())
    #
    # if params.framework.sparse:
    #     yolo_head.append(scn.ReLU())
    #     yolo_head.append(scn.SparseToDense(
    #         dimension=3, nPlanes=layer))
    # else:
    #     yolo_head.append(torch.nn.ReLU())

    return resnet, classification_head
