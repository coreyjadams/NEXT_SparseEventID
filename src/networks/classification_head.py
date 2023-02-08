from . resnet import create_resnet

def build_networks(params, input_shape):

    resnet, output_shape = create_resnet(params, input_shape)

    classification_head = torch.nn.Sequential()
    if params.framework.sparse:
        classification_head.append(scn.SparseToDense(
            dimension=3, nPlanes=current_number_of_filters))

    classification_head.append(torch.nn.Flatten())

    dense_shape = int(current_number_of_filters * numpy.prod(final_shape))
    dense_shape = int(numpy.prod(output_shape))

    current_number_of_filters = dense_shape

    for layer in params.head.layers:
        classification_head.append(torch.nn.Linear(
            in_features  = current_number_of_filters,
            out_features = layer))
        classification_head.append(torch.nn.ReLU())
        current_number_of_filters = layer


    return encoder, classification_head
