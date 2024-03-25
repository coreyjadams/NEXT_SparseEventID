import numpy
import torch


from src.config.framework import DataMode
from src.config.network   import EncoderType


def build_networks(params, input_shape):

    if params.framework.mode == DataMode.graph:
        from . mpnn import Encoder
        encoder = Encoder(params, input_shape)
    else:
        if params.encoder.type == EncoderType.resnet:
            from . resnet import Encoder
            encoder = Encoder(params, input_shape)
        elif params.encoder.type == EncoderType.vit:
            from . vit import Encoder
            encoder = Encoder(params, input_shape)

    output_shape = encoder.output_shape

    # encoder, output_shape = create_resnet(params, input_shape)

    classification_head = torch.nn.Sequential()
    current_number_of_filters = output_shape[0]
    

    if len(output_shape) > 1:
        # First step of the classification head is to pool the spatial size:
        classification_head.append(torch.nn.AvgPool3d(output_shape[1:]))
        classification_head.append(torch.nn.Flatten(start_dim=1, end_dim=-1))
        classification_head.append(torch.nn.InstanceNorm1d(current_number_of_filters))

    for i, layer in enumerate(params.head.layers):

        classification_head.append(torch.nn.Linear(
            in_features  = current_number_of_filters,
            out_features = layer))
        if layer != params.head.layers[-1]:
            classification_head.append(torch.nn.ReLU())
        current_number_of_filters = layer

    # classification_head.append(torch.nn.Tanh())
    #
    # if params.framework.sparse:
    #     yolo_head.append(scn.ReLU())
    #     yolo_head.append(scn.SparseToDense(
    #         dimension=3, nPlanes=layer))
    # else:
    #     yolo_head.append(torch.nn.ReLU())

    return encoder, classification_head
