import os
import time

from . import data_transforms
from src.config.framework import DataMode

import numpy
import torch

# Functional programming approach to building up the dataset objects:
from src.config.data import Detector

def lr_meta(detector):

    if detector == Detector.next_white:
        return numpy.array([
            ([480, 480, 320], [480., 480., 576.],[-240., -240., 0])],
            dtype=[
                ('n_voxels', "int", (3)),
                ('size', "float32", (3)),
                ('origin', "float32", (3)),
            ]
        )
    elif detector == Detector.atpc:
        return numpy.array([
            ([600, 600, 600], [6000., 6000., 6000.],[-3000., -3000., 0])],
            dtype=[
                ('n_voxels', "int", (3)),
                ('size', "float32", (3)),
                ('origin', "float32", (3)),
            ]
        )
    else:
        return numpy.array([
            ([640, 640, 1280], [960., 960., 1280.],[-480., -480., 0])],
            dtype=[
                ('n_voxels', "int", (3)),
                ('size', "float32", (3)),
                ('origin', "float32", (3)),
            ]
        )


def pmaps_meta(detector):

    # The size of the images here are padded and expanded.  This lets me downsample
    # and upsample in the networks more smoothly
    if detector == Detector.next_white:
        return numpy.array([
            ([64, 64, 128], [640, 640, 1280],[-320., -320., -45])],
            dtype=[
                ('n_voxels', "int", (3)),
                ('size', "float32", (3)),
                ('origin', "float32", (3)),
            ]
        )
    elif detector == Detector.atpc:
        return numpy.array([
            ([600, 600, 600], [6000., 6000., 6000.],[-3000., -3000., 0])],
            dtype=[
                ('n_voxels', "int", (3)),
                ('size', "float32", (3)),
                ('origin', "float32", (3)),
            ]
        )
    else:
        return numpy.array([
            ([64, 64, 128], [960, 960, 1280],[-480., -480., 0])],
            dtype=[
                ('n_voxels', "int", (3)),
                ('size', "float32", (3)),
                ('origin', "float32", (3)),
            ]
        )
    


def create_larcv_interface(random_access_mode, distributed, seed):

    # Not needed, enforced by data.py
    # if random_access_mode not in ["serial_access", "random_blocks"]:
    #     raise Exception(f"Can not use mode {random_access_mode}")

    if seed == -1:
        seed = int(time.time())
    if distributed:
        from larcv import distributed_queue_interface as queueloader
    else:
        from larcv import queueloader


    larcv_interface = queueloader.queue_interface(
        random_access_mode=str(random_access_mode.name), seed=seed)
    larcv_interface.no_warnings()

    return larcv_interface

def prepare_next_config(batch_size, input_file, data_args, name,
                        is_mc = True, get_vertex=True):


    # First, verify the files exist:
    if not os.path.exists(input_file):
        raise Exception(f"File {input_file} not found")


    from larcv.config_builder import ConfigBuilder
    cb = ConfigBuilder()
    cb.set_parameter([str(input_file)], "InputFiles")
    cb.set_parameter(6, "ProcessDriver", "IOManager", "Verbosity")
    cb.set_parameter(6, "ProcessDriver", "Verbosity")
    cb.set_parameter(6, "Verbosity")

    producer_name = data_args.image_key


    # Sparse Normalization (Do it first!):
    if data_args.normalize:
        cb.add_preprocess(
            datatype = "sparse3d",
            producer = producer_name,
            process  = "Normalize",
            OutputProducer = producer_name,
            Mean = 1.0,
            Std = 0.5
        )


    # Get the pmaps:
    cb.add_batch_filler(
        datatype  = "sparse3d",
        producer  = producer_name,
        name      = name+producer_name,
        MaxVoxels = 6500,
        Augment   = True,
        Channels  = [0]
    )



    # Build up the data_keys:
    data_keys = {
        producer_name: name + producer_name,
    }



    # if data_args.image_key == "lr_hits":
    #     # Get the deconvolved hits:
    #     cb.add_batch_filler(
    #         datatype  = "sparse3d",
    #         producer  = "lr_hits",
    #         name      = name+"lr_hits",
    #         MaxVoxels = 8000,
    #         Augment   = False,
    #         Channels  = [0]
    #     )

    #     # Build up the data_keys:
    #     data_keys['lr_hits'] = name + 'lr_hits'


    if is_mc:

        # Fetch the labels:
        cb.add_batch_filler(
            datatype  = "particle",
            producer  = "event",
            name      = name+"label",
        )
        data_keys.update({'label': name + 'label'})

    if get_vertex:
        # Vertex locations as BBoxes:
        cb.add_batch_filler(
            datatype = "bbox3d",
            producer = "vertex",
            name     = name + "vertex",
            MaxBoxes = 1,
            Channels = [0]
        )
        data_keys.update({'vertex': name + 'vertex'})



    if data_args.transform1:
        out_key = producer_name + "_1"
        add_augment_chain(cb,
            datatype   = "sparse3d",
            producer   = producer_name,
            output_key = out_key
        )
        # Get the pmaps:
        cb.add_batch_filler(
            datatype  = "sparse3d",
            producer  = out_key,
            name      = name+out_key,
            MaxVoxels = 3000,
            Augment   = False,
            Channels  = [0]
        )
        # Build up the data_keys:
        data_keys.update({
            producer_name + '_1': name + out_key,
        })

    if data_args.transform2:
        out_key = producer_name + "_2"

        add_augment_chain(cb,
            datatype   ="sparse3d",
            producer   = producer_name,
            output_key = out_key
        )
        # Get the pmaps:
        cb.add_batch_filler(
            datatype  = "sparse3d",
            producer  = out_key,
            name      = name+out_key,
            MaxVoxels = 3000,
            Augment   = False,
            Channels  = [0]
        )
        # Build up the data_keys:
        data_keys.update({
            producer_name + '_2': name + out_key
        })


    # Prepare data managers:
    io_config = {
        'filler_name' : name,
        'filler_cfg'  : cb.get_config(),
        'verbosity'   : 5,
        'make_copy'   : False
    }



    return io_config, data_keys

def add_augment_chain(config_builder, datatype, producer, output_key):
    


    config_builder.add_preprocess(
            datatype = datatype,
            producer = producer,
            process  = "Mirror",
            Axes = [0,1,2],
            OutputProducer = output_key
        )

    # GaussianBlur of pixels:
    config_builder.add_preprocess(
            datatype = datatype,
            producer = output_key,
            process  = "GaussianBlur",
            Sigma    = 0.05,
            OutputProducer = output_key
        )

    config_builder.add_preprocess(
            datatype = datatype,
            producer = output_key,
            process  = "Translate",
            MaxShiftPerAxis = [15, 15, 25],
            OutputProducer = output_key
        )




    return

def prepare_interface(batch_size, storage_name, larcv_interface, io_config, data_keys, color=0):

    """
    Not a pure function!  it changes state of the larcv_interface
    """
    larcv_interface.prepare_manager(
        storage_name, io_config, batch_size, data_keys, color=color)
    # This queues up the next data
    # self._larcv_interface.prepare_next(name)

    while larcv_interface.is_reading(storage_name):
        time.sleep(0.01)


    return larcv_interface.size(storage_name)



def create_larcv_dataset(data_args, batch_size, batch_keys,
                         input_file, name,
                         distributed=False, data_mode=DataMode.sparse):
    """
    Create a new iterable dataset of the file specified in data_args
    pass

    """

    # Create a larcv interface:
    interface = create_larcv_interface(
        random_access_mode = data_args.mode,
        distributed = distributed,
        seed=data_args.seed)


    # Next, prepare the config info for this interface:
    io_config, data_keys =  prepare_next_config(
        batch_size = batch_size,
        data_args  = data_args,
        input_file = input_file,
        name       = name,
        is_mc      = True if "sim" in name else False,
        get_vertex = data_args.vertex
    )

    # If the file has it, read in the event-wise energy for ALL events:
    import h5py
    f = h5py.File(input_file)
    particle_group_name = 'particle_event_group'
    if particle_group_name in f['Data/']:
        energy_array = f['Data'][particle_group_name]['particles']['energy_deposit']
    else:
        energy_array = None

    # Now, fire up the interface:
    prepare_interface(
        batch_size,
        storage_name    = name,
        larcv_interface = interface,
        io_config       = io_config,
        data_keys       = data_keys)


    # Finally, create the iterable object to hold all of this:
    dataset = larcv_dataset(
        larcv_interface = interface,
        batch_keys      = batch_keys,
        name            = name,
        data_args       = data_args,
        is_mc           = True if "sim" in name else False,
        energy          = energy_array,
        data_mode       = data_mode)


    return dataset

class larcv_dataset(object):
    """ Represents a (possibly distributed) larcv dataset on one file

    Implements __len__ and __iter__ to enable fast, iterable datasets.

    May also in the future implement __getitem__(idx) to enable slower random access.

    """

    def __init__(self, larcv_interface, batch_keys, name, data_args, is_mc=True, energy = None, data_mode=DataMode.sparse):
        """
        Init takes a preconfigured larcv queue interface
        """

        self.larcv_interface = larcv_interface
        self.data_args       = data_args
        self.storage_name    = name
        self.batch_keys      = batch_keys + ['entries']
        # self.vertex_depth    = vertex_depth
        # self.event_id        = event_id
        self.data_mode       = data_mode

        self.energy          = energy
        # self.data_keys = data_keys

        # Get image meta:
        self.lr_meta = lr_meta(data_args.detector)
        self.pmaps_meta = pmaps_meta(data_args.detector)

        self.stop = False

    def __len__(self):
        return self.larcv_interface.size(self.storage_name)


    def __iter__(self):

        while True:
            batch = self.fetch_next_batch(self.storage_name, True)
            yield batch

            if self.stop:
                break

    def __del__(self):
        self.stop = True

    def image_size(self, key):
        meta = self.image_meta(key)
        return meta['n_voxels'][0]

    def image_meta(self, key):
        if "pmaps" in key or "chits" in key or "voxels" in key: return self.pmaps_meta
        else: return self.lr_meta

    def fetch_next_batch(self, name, force_pop=False):

        metadata=True

        pop = True
        if not force_pop:
            pop = False


        minibatch_data = self.larcv_interface.fetch_minibatch_data(self.storage_name,
            pop=pop,fetch_meta_data=metadata)
        minibatch_dims = self.larcv_interface.fetch_minibatch_dims(self.storage_name)

        # If the returned data is None, return none and don't load more:
        if minibatch_data is None:
            return minibatch_data

        # This brings up the next data to current data
        if pop:
            self.larcv_interface.prepare_next(self.storage_name)

        for key in minibatch_data.keys():
            if key == 'entries' or key == 'event_ids':
                continue

            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

        if 'energy' in self.batch_keys:
            # Use the event IDs to pull the energy out:
            entries = minibatch_data["entries"]
            energies = self.energy[entries]
            
            minibatch_data['energy'] = energies


        # We need the event id for vertex classification, even if it's not used.
        # if self.event_id or self.vertex_depth is not None:
        if 'label' in minibatch_data.keys():
            label_particle = minibatch_data['label'][:,0]
            minibatch_data['label'] = label_particle['_pdg'].astype("int64")



        if "vertex" in minibatch_data.keys():
        #     downsample_level = 2**self.data_args.downsample
            batch_size = minibatch_data['vertex'].shape[0]
            minibatch_data['vertex'] = minibatch_data['vertex'].reshape([batch_size, 6])
            minibatch_data['vertex'] = minibatch_data['vertex'][:,0:3]
 

        # Purge unneeded keys:
        minibatch_data = {
            key : minibatch_data[key] for key in minibatch_data if key in self.batch_keys
        }

        # Shape the images:
        for key in self.batch_keys:
            if key == "event_ids": continue
            if "lr_hits" in key or "depositions" in key:
                if self.data_mode == DataMode.dense:
                    minibatch_data[key]  = data_transforms.larcvsparse_to_dense_3d(
                        minibatch_data[key],
                        dense_shape = self.lr_meta['n_voxels'][0],
                    )
                elif self.data_mode == DataMode.graph:
                    minibatch_data[key]  = data_transforms.larcvsparse_to_pytorch_geometric(
                        minibatch_data[key],
                        image_meta = self.lr_meta
                    )
                else:
                    minibatch_data[key]  = data_transforms.larcvsparse_to_scnsparse_3d(
                        minibatch_data[key])
            elif "pmaps" in key or "chits" in key or "voxels" in key:
                if self.data_mode == DataMode.dense:
                    minibatch_data[key]  = data_transforms.larcvsparse_to_dense_3d(
                        minibatch_data[key],
                        dense_shape = self.pmaps_meta['n_voxels'][0],
                    )
                elif self.data_mode == DataMode.graph:
                    minibatch_data[key]  = data_transforms.larcvsparse_to_pytorch_geometric(
                        minibatch_data[key],
                        image_meta = self.pmaps_meta
                    )
                else:
                    minibatch_data[key]  = data_transforms.larcvsparse_to_scnsparse_3d(
                        minibatch_data[key])
            else:
                minibatch_data[key] = torch.tensor(minibatch_data[key])

        return minibatch_data
