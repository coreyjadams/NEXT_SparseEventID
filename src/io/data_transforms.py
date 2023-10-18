import numpy

'''
This is a torch-free file that exists to massage data
From sparse to dense or dense to sparse, etc.

This can also convert from sparse to sparse to rearrange formats
For example, larcv BatchFillerSparseTensor2D (and 3D) output data
with the format of
    [B, N_planes, Max_voxels, N_features]

where N_features is 2 or 3 depending on whether or not values are included
(or 3 or 4 in the 3D case)

# The input of a pointnet type format can work with this, but SparseConvNet
# requires a tuple of (coords, features, [batch_size, optional])


'''


def larcvsparse_to_scnsparse_3d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everythin else)

    n_dims = input_array.shape[-1]

    split_tensors = numpy.split(input_array, n_dims, axis=-1)


    # To map out the non_zero locations now is easy:
    non_zero_inds = numpy.where(split_tensors[-1] != -999)

    # The batch dimension is just the first piece of the non-zero indexes:
    batch_size  = input_array.shape[0]
    batch_index = non_zero_inds[0]

    # Getting the voxel values (features) is also straightforward:
    features = numpy.expand_dims(split_tensors[-1][non_zero_inds],axis=-1)

    # Lastly, we need to stack up the coordinates, which we do here:
    dimension_list = []
    for i in range(len(split_tensors) - 1):
        dimension_list.append(split_tensors[i][non_zero_inds])

    # Tack on the batch index to this list for stacking:
    dimension_list.append(batch_index)

    # And stack this into one numpy array:
    dimension = numpy.stack(dimension_list, axis=-1)

    output_array = (dimension.astype("long"), features.astype("float32"), batch_size,)
    return output_array


def larcvsparse_to_dense_3d(input_array, dense_shape):

    batch_size = input_array.shape[0]
    output_array = numpy.zeros((batch_size, 1, *(dense_shape)) , dtype=numpy.float32)
    # This is the "real" size:
    # output_array = numpy.zeros((batch_size, 1, 45, 45, 275), dtype=numpy.float32)
    x_coords   = input_array[:,0,:,0]
    y_coords   = input_array[:,0,:,1]
    z_coords   = input_array[:,0,:,2]
    val_coords = input_array[:,0,:,3]


    # print(x_coords[0:100])
    # print(y_coords[0:100])
    # print(z_coords[0:100])

    # Find the non_zero indexes of the input:
    batch_index, voxel_index = numpy.where(val_coords != -999)

    values  = val_coords[batch_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, voxel_index])
    z_index = numpy.int32(z_coords[batch_index, voxel_index])


    # Fill in the output tensor

    output_array[batch_index, 0, x_index, y_index, z_index] = values
    return output_array



def form_yolo_targets(vertex_depth, vertex_labels, particle_labels, event_labels, dataformat, image_meta, ds):

    batch_size = event_labels.shape[0]
    event_energy = particle_labels['_energy_init'][:,0]

    # Vertex comes out with shape [batch_size, channels, max_boxes, 2*ndim (so 4, in this case)]
    image_shape = [ int(i /ds) for i in image_meta['full_pixels'][0] ]

    vertex_labels = vertex_labels[:,:,0,0:2]
    # The data gets loaded in (W, H) format and we need it in (H, W) format.:
    vertex_labels[:,:,[0,1]] = vertex_labels[:,:,[1,0]]


    # First, determine the dimensionality of the output space of the vertex yolo network:
    vertex_output_space =tuple(d // 2**vertex_depth  for d in image_shape )


    if dataformat == DataFormatKind.channels_last:
        # print(vertex_labels.shape)
        vertex_labels = numpy.transpose(vertex_labels,(0,2,1))
        origin = numpy.transpose(image_meta["origin"])
        size   = numpy.transpose(image_meta["size"])
        vertex_presence_labels = numpy.zeros((batch_size,) + vertex_output_space + (3,), dtype="float32")
        vertex_output_space_a  = numpy.reshape(numpy.asarray(vertex_output_space), (2,1))
    else:
        # Nimages, 3 planes, shape-per-plane
        vertex_presence_labels = numpy.zeros((batch_size, 3,) + vertex_output_space, dtype="float32")
        origin = image_meta["origin"]
        size   = image_meta["size"]
        vertex_output_space_a  = numpy.asarray(vertex_output_space)

    n_pixels_vertex = 2**vertex_depth


    # To create the right bounding box location, we have to map the vertex x/z/y to a set of pixels.
    corrected_vertex_position = vertex_labels - origin
    fractional_vertex_position = corrected_vertex_position / size

    # print(vertex_output_space.shape)
    vertex_output_space_anchor_box_float = vertex_output_space_a * fractional_vertex_position

    vertex_output_space_anchor_box = vertex_output_space_anchor_box_float.astype("int")


    # This part creates indexes into the presence labels values:
    batch_index = numpy.arange(batch_size).repeat(3) # 3 for 3 planes
    plane_index = numpy.tile(numpy.arange(3), batch_size) # Tile 3 times for 3 planes

    if dataformat == DataFormatKind.channels_last:
        h_index = numpy.concatenate(vertex_output_space_anchor_box[:,0,:])
        w_index = numpy.concatenate(vertex_output_space_anchor_box[:,1,:])
    else:
        h_index = numpy.concatenate(vertex_output_space_anchor_box[:,:,0])
        w_index = numpy.concatenate(vertex_output_space_anchor_box[:,:,1])

    if dataformat == DataFormatKind.channels_last:
        vertex_presence_labels[batch_index, h_index, w_index, plane_index] = 1.0
    else:
        vertex_presence_labels[batch_index, plane_index, h_index, w_index] = 1.0



    # Finally, we should exclude any event that is labeled "cosmic only" from having a vertex
    # truth label:
    cosmics = event_labels == 3
    vertex_presence_labels[cosmics,:,:,:] = 0.0




    # Now, compute the location inside of an achor box for x/y.
    # Normalize to (0,1)

    bounding_box_location = vertex_output_space_anchor_box_float - vertex_output_space_anchor_box

    bounding_box_location = bounding_box_location.astype("float32")

    if dataformat == DataFormatKind.channels_last:
        vertex_presence_labels = numpy.split(vertex_presence_labels, 3, axis=-1)
        vertex_presence_labels = [v.reshape((batch_size, ) + vertex_output_space) for v in vertex_presence_labels]
    else:
        vertex_presence_labels = numpy.split(vertex_presence_labels, 3, axis=1)
        vertex_presence_labels = [v.reshape((batch_size, ) + vertex_output_space) for v in vertex_presence_labels]


    return {
        "detection"  : vertex_presence_labels,
        "regression" : bounding_box_location,
        "energy"     : event_energy,
        "xy_loc"     : vertex_labels,
    }


def larcvsparse_to_pytorch_geometric(input_array):

    # Need to create node features and an adjacency matrix.
    # Define points as connected if they fall within some radius R
    # For each node, it's node features can be (x, y, z, E) as well as
    # The number of nearby nodes (within R), the average energy of those
    # nodes, and average distance away from its neighbors.
    # For each edge, dunno

    # Ultimately, need to build this into a graph for pytorch_geometric
    
    batch_size = input_array.shape[0]

    # output_array = numpy.zeros((batch_size, 1, 45, 45, 275), dtype=numpy.float32)
    x_coords   = input_array[:,0,:,0]
    y_coords   = input_array[:,0,:,1]
    z_coords   = input_array[:,0,:,2]
    val_coords = input_array[:,0,:,3]


    # Find the non_zero indexes of the input:
    batch_index, voxel_index = numpy.where(val_coords != -999)

    values  = val_coords[batch_index, voxel_index]
    x_index = numpy.int32(x_coords[batch_index, voxel_index])
    y_index = numpy.int32(y_coords[batch_index, voxel_index])
    z_index = numpy.int32(z_coords[batch_index, voxel_index])


    print(values.shape)
    print(x_index.shape)
    exit()