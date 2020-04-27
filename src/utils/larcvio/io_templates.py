from . import larcv_io


# Here, we set up a bunch of template IO formats in the form of callable functions:

def event_id_io(input_file, name, labeled, augment = True):
    max_voxels = 1000
    data_proc = gen_sparse3d_data_filler(name=name + "data", producer="\"voxels\"", max_voxels=max_voxels, augment=augment)


    config = larcv_io.ThreadIOConfig(name=name)

    config.add_process(data_proc)
    if labeled:
        label_proc = gen_label_filler(name)
        config.add_process(label_proc)

    config.set_param("InputFiles", input_file)
    return config

def cycleGAN_io(input_file, name, augment = True):
    max_voxels = 1000
    data_proc = gen_sparse3d_data_filler(name=name + "data", producer="\"voxels\"", max_voxels=max_voxels, augment=augment)

    config = larcv_io.ThreadIOConfig(name=name)

    config.add_process(data_proc)

    config.set_param("InputFiles", input_file)
    return config


def output_io(input_file, output_file):


    config = larcv_io.IOManagerConfig(name="IOManager")
    # Force ana files to go in order:

    config._params['RandomAccess'] = "0"

    print ('output_io   input_file:', input_file, ' output_file:', output_file)
    config.set_param("InputFiles", input_file)
    config.set_param("OutFileName", output_file)

    # These lines slim down the output file.
    # Without them, 25 output events is 2.8M and takes 38s
    # With the, 25 output events is 119K and takes 36s
    config.set_param("ReadOnlyType", "[\"sparse3d\",\"sparse3d\",\"sparse3d\",\"sparse3d\"]")
    config.set_param("ReadOnlyName", "[\"voxels_E\",\"voxels_E_norm\",\"voxels_E_scaled\",\"voxels_Q\"]")
   # config.set_param("ReadOnlyType", "[\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\",\"particle\"]")
   # config.set_param("ReadOnlyName", "[\"sbndneutrino\",\"sbndsegmerged\",\"cpiID\",\"neutID\",\"npiID\",\"protID\",\"all\"]")

    return config


def gen_sparse3d_data_filler(name, producer, max_voxels, augment = True):

    proc = larcv_io.ProcessConfig(proc_name=name, proc_type="BatchFillerSparseTensor3D")

    proc.set_param("Verbosity",         "3")
    proc.set_param("Tensor3DProducer",  producer)
    proc.set_param("IncludeValues",     "true")
    proc.set_param("MaxVoxels",         max_voxels)
    proc.set_param("UnfilledVoxelValue","-999")
    if augment:
        proc.set_param("Augment",           "true")
    else:
        proc.set_param("Augment",           "false")

    return proc


def gen_label_filler(prepend_names):

    proc = larcv_io.ProcessConfig(proc_name=prepend_names + "label", proc_type="BatchFillerPIDLabel")

    proc.set_param("Verbosity",         "3")
    proc.set_param("ParticleProducer",  "label")
    proc.set_param("PdgClassList",      "[{}]".format(",".join([str(i) for i in range(2)])))

    return proc
