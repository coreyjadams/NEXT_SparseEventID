import os
import tempfile
import sys
import time
from collections import OrderedDict

import numpy

import torch

from . larcvio import larcv_fetcher


import datetime

# This uses tensorboardX to save summaries and metrics to tensorboard compatible files.

import tensorboardX

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):
        self._iteration     = 0
        self._global_step   = -1
        self.args           = args
        self._epoch_size    = 1
        self.larcv_fetcher  = larcv_fetcher.larcv_fetcher(self.args.distributed, seed=None)


    def init_network(self):
        '''This function creates all networks needed for processing.
        
        '''
        pass

    def initialize_io(self):
        pass

    def initialize(self, io_only=False):



        self.initialize_io()


        if io_only:
            return


        self.init_network()

        self.init_optimizer()

        self.init_saver()

        state = self.restore_model()

        if state is not None:
            self.load_state(state)
        else:
            self._global_step = 0

        self.set_log_keys()

        self.model_to_device()


    def set_log_keys(self):
        pass

    def get_device(self):
        # Convert the input data to torch tensors
        if self.args.compute_mode == "GPU":
            device = torch.device('cuda')
            # print(device)
        else:
            device = torch.device('cpu')

        return device

    def init_optimizer(self):
        pass

    def init_saver(self):

        # This sets up the summary saver:
        if self.args.training:
            self._saver = tensorboardX.SummaryWriter(self.args.log_directory)




    def restore_model(self):
        ''' This function attempts to restore the model from file
        '''

        _, checkpoint_file_path = self.get_model_filepath()


        if not os.path.isfile(checkpoint_file_path):
            print("Returning none!")
            return None

        # Parse the checkpoint file and use that to get the latest file path

        with open(checkpoint_file_path, 'r') as _ckp:
            for line in _ckp.readlines():
                if line.startswith("latest: "):
                    chkp_file = line.replace("latest: ", "").rstrip('\n')
                    chkp_file = os.path.dirname(checkpoint_file_path) + "/" + chkp_file
                    print("Restoring weights from ", chkp_file)
                    break

        if self.args.compute_mode == "CPU":
            state = torch.load(chkp_file, map_location='cpu')
        else:
            state = torch.load(chkp_file)

        return state

    def load_state(self, state):

        pass


    def get_model_save_dict(self):
        '''Return the save dict for the current models
        
        Expected to vary between cycleGAN and eventID
        '''

        pass

    def save_model(self):
        '''Save the model to file
        
        '''

        current_file_path, checkpoint_file_path = self.get_model_filepath()

        state_dict = self.get_model_save_dict()

        # Make sure the path actually exists:
        if not os.path.isdir(os.path.dirname(current_file_path)):
            os.makedirs(os.path.dirname(current_file_path))

        torch.save(state_dict, current_file_path)

        # Parse the checkpoint file to see what the last checkpoints were:

        # Keep only the last 5 checkpoints
        n_keep = 5


        past_checkpoint_files = {}
        try:
            with open(checkpoint_file_path, 'r') as _chkpt:
                for line in _chkpt.readlines():
                    line = line.rstrip('\n')
                    vals = line.split(":")
                    if vals[0] != 'latest':
                        past_checkpoint_files.update({int(vals[0]) : vals[1].replace(' ', '')})
        except:
            pass
        

        # Remove the oldest checkpoints while the number is greater than n_keep
        while len(past_checkpoint_files) >= n_keep:
            min_index = min(past_checkpoint_files.keys())
            file_to_remove = os.path.dirname(checkpoint_file_path) + "/" + past_checkpoint_files[min_index]
            os.remove(file_to_remove)
            past_checkpoint_files.pop(min_index)



        # Update the checkpoint file
        with open(checkpoint_file_path, 'w') as _chkpt:
            _chkpt.write('latest: {}\n'.format(os.path.basename(current_file_path)))
            _chkpt.write('{}: {}\n'.format(self._global_step, os.path.basename(current_file_path)))
            for key in past_checkpoint_files:
                _chkpt.write('{}: {}\n'.format(key, past_checkpoint_files[key]))


    def get_model_filepath(self):
        '''Helper function to build the filepath of a model for saving and restoring:
        
        
        '''

        # Find the base path of the log directory
        if self.args.checkpoint_directory == None:
            file_path= self.args.log_directory  + "/checkpoints/"
        else:
            file_path= self.args.checkpoint_directory  + "/checkpoints/"


        name = file_path + 'model-{}.ckpt'.format(self._global_step)
        checkpoint_file_path = file_path + "checkpoint"

        return name, checkpoint_file_path



    def log(self, metrics, saver=''):


        if self._global_step % self.args.logging_iteration == 0:
            
            self._current_log_time = datetime.datetime.now()

            s = ""

            if 'it.' in metrics:
                # This prints out the iteration for ana steps
                s += "it.: {}, ".format(metrics['it.'])

            # Build up a string for logging:
            if self._log_keys != []:
                s += ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self._log_keys])
            else:
                s += ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])
      

            try:
                s += " ({:.2}s / {:.2} IOs / {:.2})".format(
                    (self._current_log_time - self._previous_log_time).total_seconds(), 
                    metrics['io_fetch_time'],
                    metrics['step_time'])
            except:
                pass

            self._previous_log_time = self._current_log_time

            print("{} Step {} metrics: {}".format(saver, self._global_step, s))



    def summary(self, metrics,saver=""):

        if self._saver is None:
            return

        if self._global_step % self.args.summary_iteration == 0:
            for metric in metrics:
                name = metric
                if saver == "test":
                    self._aux_saver.add_scalar(metric, metrics[metric], self._global_step)
                else:
                    self._saver.add_scalar(metric, metrics[metric], self._global_step)


            # try to get the learning rate
            # print self._lr_scheduler.get_lr()
            self._saver.add_scalar("learning_rate", self._opt.state_dict()['param_groups'][0]['lr'], self._global_step)
            pass



    def increment_global_step(self):

        previous_epoch = int((self._global_step * self.args.minibatch_size) / self._epoch_size)
        self._global_step += 1
        current_epoch = int((self._global_step * self.args.minibatch_size) / self._epoch_size)

        self.on_step_end()

        if previous_epoch != current_epoch:
            self.on_epoch_end()

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass




    def checkpoint(self):

        if self.args.checkpoint_iteration == -1:
            return

        if self._global_step % self.args.checkpoint_iteration == 0 and self._global_step != 0:
            # Save a checkpoint, but don't do it on the first pass
            self.save_model()


    def batch_process(self):
        

        # This is the 'master' function, so it controls a lot

        # If we're not training, force the number of iterations to the epoch size or less
        if not self.args.training:
            if self.args.iterations > self._epoch_size:
                self.args.iterations = self._epoch_size


        # Run iterations
        for i in range(self.args.iterations):
            if self.args.training and self._iteration >= self.args.iterations:
                print('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break

            if self.args.training:
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step(i)


        if self.args.training:
            if self._saver is not None:
                self._saver.close()
            if self._aux_saver is not None:
                self._aux_saver.close()

