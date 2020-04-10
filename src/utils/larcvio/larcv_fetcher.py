import os
import torch

from . import data_transforms
from . import io_templates
import tempfile
import numpy, h5py


class larcv_fetcher(object):

    def __init__(self, distributed, seed=None):

        self._cleanup = []
        self._eventID_labels   = {}
        self._eventID_energies = {}

        self._color = None
        if distributed:
            from larcv import distributed_queue_interface
            self._larcv_interface = distributed_queue_interface.queue_interface()
            self._color = 0
        else:
            from larcv import queueloader
            self._larcv_interface = queueloader.queue_interface(random_access_mode="random_blocks", seed=None)


    def __del__(self):
        for f in self._cleanup:
            os.unlink(f.name)

    def prepare_cycleGAN_sample(self, name, input_file, batch_size):

        config = io_templates.cycleGAN_io(input_file=input_file, name=name)

        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())

        main_file.close()
        self._cleanup.append(main_file)

        io_config = {
            'filler_name' : config._name,
            'filler_cfg'  : main_file.name,
            'verbosity'   : 5,
            'make_copy'   : False
        }

        # Build up the data_keys:
        data_keys = {'image' : name + "data"}
        # for proc in config._process_list._processes:
        #     data_keys[proc._name] = proc._name

        self._larcv_interface.prepare_manager(name, io_config, batch_size, data_keys, color=self._color)


        return self._larcv_interface.size(name)

    def eventID_labels(self,name):
        if name in self._eventID_labels:
            return self._eventID_labels[name]

    def eventID_energies(self,name):
        if name in self._eventID_energies:
            return self._eventID_energies[name]

    def prepare_eventID_sample(self, name, input_file, batch_size):
        config = io_templates.event_id_io(input_file=input_file, name=name)

        # Generate a named temp file:
        main_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        main_file.write(config.generate_config_str())

        main_file.close()
        self._cleanup.append(main_file)

        io_config = {
            'filler_name' : config._name,
            'filler_cfg'  : main_file.name,
            'verbosity'   : 5,
            'make_copy'   : False
        }

        # Build up the data_keys:
        data_keys = {
            'image' : name + "data",
            'label' : name + "label"
        }

        # For this work, we can cache all of the labels and energies:
        import h5py
        f = h5py.File(input_file, 'r')
        self._eventID_labels[name] = f['Data/particle_label_group/particles/']['pdg']
        self._eventID_energies[name] = f['Data/particle_label_group/particles/']['energy_init']



        self._larcv_interface.prepare_manager(name, io_config, batch_size, data_keys,color=self._color)
        self._larcv_interface.prepare_next(name)


        return self._larcv_interface.size(name)

    def prepare_eventID_output(self, name, input_file, output_file):
        config = io_templates.output_io(input_file=input_file, output_file=output_file)

        out_file_config = tempfile.NamedTemporaryFile(mode='w', delete=False)
        out_file_config.write(config.generate_config_str())

        out_file_config.close()
        self._cleanup.append(out_file_config)

        self._larcv_interface.prepare_writer(out_file.name, output_file)


    def fetch_next_cycleGAN_batch(self, name):

        # For the serial mode, call next here:

        minibatch_data = self._larcv_interface.fetch_minibatch_data(name, pop=True, fetch_meta_data=False)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(name)
        self._larcv_interface.prepare_next(name)

        # Here, do some massaging to convert the input data to another format, if necessary:
        # Need to convert sparse larcv into a dense numpy array:
        minibatch_data['image'] = data_transforms.larcvsparse_to_dense_3d(minibatch_data['image'])

        return minibatch_data


    def fetch_next_eventID_batch(self, name):

        # For the serial mode, call next here:

        minibatch_data = self._larcv_interface.fetch_minibatch_data(name, pop=True, fetch_meta_data=False)
        minibatch_dims = self._larcv_interface.fetch_minibatch_dims(name)

        # For the serial mode, call next here:
        self._larcv_interface.prepare_next(name)

        # Here, do some massaging to convert the input data to another format, if necessary:
        # Need to convert sparse larcv into a dense numpy array:
        minibatch_data['image'] = data_transforms.larcvsparse_to_scnsparse_3d(minibatch_data['image'])

        return minibatch_data


    def to_torch_cycleGAN(self, minibatch_data, device=None):

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')


        for key in minibatch_data:
            if key == 'entries' or key =='event_ids':
                continue
            else:
                minibatch_data[key] = torch.tensor(minibatch_data[key],device=device)

        return minibatch_data


    def to_torch_eventID(self, minibatch_data, device=None):

        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')


        for key in minibatch_data:
            if key == 'entries' or key =='event_ids':
                continue
            if key == 'image':
                minibatch_data['image'] = (
                        torch.tensor(minibatch_data['image'][0]).long(),
                        torch.tensor(minibatch_data['image'][1], device=device),
                        minibatch_data['image'][2],
                    )
            else:
                minibatch_data[key] = torch.tensor(minibatch_data[key],device=device)

        return minibatch_data
