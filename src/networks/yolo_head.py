import torch
import numpy
from . resnet import create_resnet

def build_networks(params, input_shape):

    resnet, output_shape = create_resnet(params, input_shape)

    yolo_head = torch.nn.Sequential()

    print("Output shape: ", output_shape)

    if params.framework.sparse:
        import sparseconvnet as scn
        from . sparse_building_blocks import Block, ResidualBlock
        from . sparse_building_blocks import ConvolutionDownsample
        from . sparse_building_blocks import BlockSeries
    else:
        from . building_blocks import Block, ResidualBlock, BlockSeries
        from . building_blocks import ConvolutionDownsample
        from . building_blocks import MaxPooling


    current_number_of_filters = output_shape[0]

    for layer in params.head.layers:
        yolo_head.append(
            Block(
                nIn  = current_number_of_filters,
                nOut = layer,
                params = params.encoder
            )
        )
        current_number_of_filters = layer

    if params.framework.sparse:
        yolo_head.append(scn.Sigmoid())
        yolo_head.append(scn.SparseToDense(
            dimension=3, nPlanes=layer))
    else:
        yolo_head.append(torch.nn.Sigmoid())

    return resnet, yolo_head
