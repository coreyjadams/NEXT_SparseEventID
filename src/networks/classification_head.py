import torch
import numpy
from . resnet import create_resnet


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


    resnet, output_shape = create_resnet(params, input_shape)

    current_number_of_filters = output_shape[0]

    filter_squeeze = 4

    classification_head = torch.nn.Sequential()
    if params.framework.sparse:
        import sparseconvnet as scn
        classification_head.append(Block(
            nIn  = current_number_of_filters,
            nOut = filter_squeeze,
            params = params.encoder
        ))
        classification_head.append(scn.SparseToDense(
            dimension=3, nPlanes=current_number_of_filters))
    else:
        classification_head.append(Block(
            nIn  = current_number_of_filters,
            nOut = filter_squeeze,
            params = params.encoder
        ))
    # Flatten the images:
    classification_head.append(torch.nn.Flatten())

    # How many neurons?
    output_shape[0] = filter_squeeze
    dense_shape = int(numpy.prod(output_shape))
    print(dense_shape)

    current_number_of_filters = dense_shape

    for layer in params.head.layers:
        classification_head.append(torch.nn.Linear(
            in_features  = current_number_of_filters,
            out_features = layer))
        classification_head.append(torch.nn.ReLU())
        current_number_of_filters = layer

    #
    # if params.framework.sparse:
    #     yolo_head.append(scn.ReLU())
    #     yolo_head.append(scn.SparseToDense(
    #         dimension=3, nPlanes=layer))
    # else:
    #     yolo_head.append(torch.nn.ReLU())

    return resnet, classification_head
