import numpy
import torch


from src.config.framework import DataMode


def build_networks(params, input_shape):

    if params.framework.mode != DataMode.graph:
        from . resnet import Encoder
        encoder = Encoder(params, input_shape)
    else:
        from . mpnn import Encoder
        encoder = Encoder(params, input_shape)

    output_shape = encoder.output_shape

    # encoder, output_shape = create_resnet(params, input_shape)

    classification_head = torch.nn.Sequential()
    current_number_of_filters = output_shape[0]




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
