import os
import sys
import time
from collections import OrderedDict
import uuid

import numpy

import torch

from mpi4py import MPI
comm = MPI.COMM_WORLD


from larcv.distributed_queue_interface import queue_interface

from .trainer_eventID import trainer_eventID



class distributed_eventID(trainer_eventID):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args):
        # Rely on the base class for most standard parameters, only
        # search for parameters relevant for distributed computing here
        trainer_eventID.__init__(self, args)

        # Put the IO rank as the last rank in the COMM, since rank 0 does tf saves
        if self.args.distributed_mode == "horovod":
            global hvd
            import horovod.torch as hvd
            hvd.init()
            self._rank            = hvd.rank()
            if self.args.compute_mode == "GPU":
                os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())

        else:

            import socket
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP

            rank = MPI.COMM_WORLD.Get_rank()


            # Pytorch will look for these:
            local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
            size = MPI.COMM_WORLD.Get_size()
            rank = MPI.COMM_WORLD.Get_rank()

            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(size)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

            self._rank = rank
            self._size = size

            # It will want the master address too, which we'll broadcast:
            if rank == 0:
                master_addr = socket.gethostname()
            else:
                master_addr = None

            master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = str(2345)

            # What backend?  nccl on GPU, gloo on CPU
            if self.args.compute_mode == "GPU": backend = 'nccl'
            elif self.args.compute_mode == "CPU": backend = 'gloo'

            torch.distributed.init_process_group(
                backend=backend, init_method='env://')


    def save_model(self):

        if self._rank == 0:
            trainer_eventID.save_model(self)


    def init_optimizer(self):

        # This takes the base optimizer (self._opt) and replaces
        # it with a distributed version

        trainer_eventID.init_optimizer(self)

        if self.args.distributed_mode == "horovod":
            self._opt = hvd.DistributedOptimizer(self._opt, named_parameters=self._net.named_parameters())

            hvd.broadcast_optimizer_state(self._opt, root_rank = 0)



    def init_saver(self):
        self._saver = None
        if self._rank == 0:
            trainer_eventID.init_saver(self)
        else:
            self._saver = None
            self._aux_saver = None


    def restore_model(self):
        if self._rank == 0:
            return trainer_eventID.restore_model(self)
        else:
            return None


        if self.args.distributed_mode == "horovod":

            # Broadcast the global step:
            self._global_step = hvd.broadcast_object(self._global_step, root_rank = 0)

            # Broadcast the state of the model:
            hvd.broadcast_parameters(self._net.state_dict(), root_rank = 0)

            # Broadcast the optimizer state:
            hvd.broadcast_optimizer_state(self._opt, root_rank = 0)

            # Horovod doesn't actually move the optimizer onto a GPU:
            if self.args.compute_mode == "GPU":
                for state in self._opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()



            # Broadcast the LR Schedule state:
            state_dict = hvd.broadcast_object(self.lr_scheduler.state_dict(), root_rank = 0)

        elif self.args.distributed_mode == "DDP":

            if self.args.compute_mode == "GPU":
                self._net.cuda()

            self._net = torch.nn.parallel.DistributedDataParallel(self._net)



            self._global_step = MPI.COMM_WORLD.bcast(self._global_step, root=0)
            state_dict = MPI.COMM_WORLD.bcast(self.lr_scheduler.state_dict(), root=0)

        # Load the state dict:
        self.lr_scheduler.load_state_dict(state_dict)

    def print(self, *argv):
        if self._rank == 0:
            trainer_eventID.print(self, *argv)




    def summary(self, metrics, saver=""):
        if self._rank == 0:
            trainer_eventID.summary(self, metrics, saver)
        return

    def _compute_metrics(self, logits, minibatch_data, loss):
        # This function calls the parent function which computes local metrics.
        # Then, it performs an all reduce on all metrics:
        metrics = trainer_eventID._compute_metrics(self, logits, minibatch_data, loss)


        if self.args.distributed_mode == "horovod":
            for key in metrics:
                metrics[key] = hvd.allreduce(metrics[key], name = key)
        elif self.args.distributed_mode == "DDP":
            for key in metrics:
                torch.distributed.all_reduce(metrics[key])
                metrics[key] /= self._size

        return metrics

    def on_epoch_end(self):
        pass

    def on_step_end(self):
        # self._lr_scheduler.step()
        pass


    # def get_device(self):
    #             # Convert the input data to torch tensors
    #     if FLAGS.COMPUTE_MODE == "GPU":
    #         device = torch.device('cuda:{}'.format(hvd.local_rank()))
    #         # print(device)
    #     else:
    #         device = torch.device('cpu')


    #     return device

    #
    # def to_torch(self, minibatch_data):
    #
    #     # This function wraps the to-torch function but for a gpu forces
    #
    #     device = self.get_device()
    #
    #     minibatch_data = trainer_eventID.to_torch(self, minibatch_data, device)
    #
    #     return minibatch_data

    def log(self, metrics, saver=""):
        if self._rank == 0:
            trainer_eventID.log(self, metrics, saver)
