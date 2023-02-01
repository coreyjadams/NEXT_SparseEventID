#!/usr/bin/env python
import os,sys,signal
import time
import pathlib

import numpy

try:
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
except:
    pass

# For configuration:
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.experimental import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log

hydra.output_subdir = None

#############################

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

from src.config import Config
from src.config.mode import ModeKind

from src.io import create_larcv_dataset

from src import logging

import atexit

class exec(object):

    def __init__(self, config):

        self.args = config

        rank = self.init_mpi()


        # Add to the output dir:
        # self.args.output_dir += f"/{self.args.network.name}/"
        self.args.output_dir += f"/{self.args.run.id}/"

        # Create the output directory if needed:
        if rank == 0:
            outpath = pathlib.Path(self.args.output_dir)
            outpath.mkdir(exist_ok=True, parents=True)

        self.configure_logger(rank)

        self.validate_arguments()

        # Print the command line args to the log file:
        logger = logging.getLogger("NEXT")
        logger.info("Dumping launch arguments.")
        logger.info(sys.argv)
        logger.info(self.__str__())

        logger.info("Configuring Datasets.")
        self.datasets = self.configure_datasets()
        logger.info("Data pipeline ready.")


    def run(self):
        if self.args.mode.name == ModeKind.train:
            self.train()
        if self.args.mode.name == ModeKind.iotest:
            self.iotest()
        if self.args.mode.name == ModeKind.inference:
            self.inference()

    def exit(self):
        if hasattr(self, "trainer"):
            self.trainer.exit()

    def init_mpi(self):
        if not self.args.run.distributed:
            return 0
        else:
            from src.utils import mpi_init_and_local_rank
            local_rank = mpi_init_and_local_rank(set_env=True, verbose=False)

            return int(os.environ["RANK"])

    def configure_lr_schedule(self, epoch_length, max_epochs):



        if self.args.mode.optimizer.lr_schedule.name == "one_cycle":
            from src.utils import OneCycle
            lr_schedule = OneCycle(self.args.mode.optimizer.lr_schedule)
        elif self.args.mode.optimizer.lr_schedule.name == "standard":
            from src.utils import WarmupFlatDecay
            schedule_args = self.args.mode.optimizer.lr_schedule
            lr_schedule = WarmupFlatDecay(
                peak_learning_rate = schedule_args.peak_learning_rate,
                decay_floor  = schedule_args.decay_floor,
                epoch_length = epoch_length,
                decay_epochs = schedule_args.decay_epochs,
                total_epochs = max_epochs
            )

        return lr_schedule

    def configure_datasets(self):
        """
        This function creates the non-framework iterable datasets used in this app.

        They get converted to framework specific tools, if needed, in the
        framework specific code.
        """

        from src.io import create_torch_larcv_dataloader

        self.batch_keys = ["pmaps", "label"]
        ds = {}
        for active in self.args.data.active:
            larcv_ds = create_larcv_dataset(self.args.data,
                batch_size   = self.args.run.minibatch_size,
                input_file   = getattr(self.args.data, active),
                name         = active,
                distributed  = self.args.run.distributed,
                batch_keys   = self.batch_keys,
                sparse       = self.args.framework.sparse
            )

            ds.update({
                active :  create_torch_larcv_dataloader(larcv_ds, self.args.run.minibatch_size)
            })

        return ds


    def configure_logger(self, rank):

        logger = logging.getLogger("NEXT")


    def train(self):

        logger = logging.getLogger("NEXT")

        logger.info("Running Training")

        self.make_trainer()

        from src.utils.lightning import train
        train(self.args, self.trainer, self.datasets)


    def iotest(self):

        logger = logging.getLogger("NEXT")

        logger.info("Running IO Test")


        # self.trainer.initialize(io_only=True)

        if self.args.run.distributed:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0




        for key, dataset in self.datasets.items():
            logger.info(f"Reading dataset {key}")
            global_start = time.time()
            total_reads = 0


            # Determine the stopping point:
            break_i = self.args.run.length * len(dataset)

            start = time.time()
            for i, minibatch in enumerate(dataset):

                end = time.time()
                if i >= break_i: break
                logger.info(f"{i}: Time to fetch a minibatch of data: {end - start:.2f}s")
                start = time.time()
                total_reads += 1

            total_time = time.time() - global_start
            images_read = total_reads * self.args.run.minibatch_size
            logger.info(f"{key} - Total IO Time: {total_time:.2f}s")
            logger.info(f"{key} - Total images read per batch: {self.args.run.minibatch_size}")
            logger.info(f"{key} - Average Image IO Throughput: { images_read / total_time:.3f}")

    def make_trainer(self):

        if 'environment_variables' in self.args.framework:
            for env in self.args.framework.environment_variables.keys():
                os.environ[env] = self.args.framework.environment_variables[env]

        dataset_length = max([len(ds) for ds in self.datasets.values()])


        if self.args.mode.name == ModeKind.train:
            lr_schedule = self.configure_lr_schedule(dataset_length, self.args.run.length)
        else:
            lr_schedule = None

        from src.utils import create_lightning_module
        self.trainer = create_lightning_module(
            self.args,
            self.datasets,
            lr_schedule,
            self.batch_keys,
        )


    def inference(self):


        logger = logging.getLogger("NEXT")

        logger.info("Running Inference")
        logger.info(self.__str__())

        self.make_trainer()

        self.trainer.initialize()
        self.trainer.batch_process()

    def dictionary_to_str(self, in_dict, indentation = 0):
        substr = ""
        for key in sorted(in_dict.keys()):
            if type(in_dict[key]) == DictConfig or type(in_dict[key]) == dict:
                s = "{none:{fill1}{align1}{width1}}{key}: \n".format(
                        none="", fill1=" ", align1="<", width1=indentation, key=key
                    )
                substr += s + self.dictionary_to_str(in_dict[key], indentation=indentation+2)
            else:
                if hasattr(in_dict[key], "name"): attr = in_dict[key].name
                else: attr = in_dict[key]
                s = '{none:{fill1}{align1}{width1}}{message:{fill2}{align2}{width2}}: {attr}\n'.format(
                   none= "",
                   fill1=" ",
                   align1="<",
                   width1=indentation,
                   message=key,
                   fill2='.',
                   align2='<',
                   width2=30-indentation,
                   attr = attr,
                )
                substr += s
        return substr

    def __str__(self):

        s = "\n\n-- CONFIG --\n"
        substring = s +  self.dictionary_to_str(self.args)

        return substring




    def validate_arguments(self):
        pass




@hydra.main(version_base=None, config_path="../src/config/recipes/", config_name="config")
def main(cfg : OmegaConf) -> None:

    s = exec(cfg)
    atexit.register(s.exit)

    s.run()

if __name__ == '__main__':
    #  Is this good practice?  No.  But hydra doesn't give a great alternative
    import sys
    if "--help" not in sys.argv and "--hydra-help" not in sys.argv:
        sys.argv += [
            'hydra/job_logging=disabled',
            'hydra.output_subdir=null',
            'hydra.job.chdir=False',
            'hydra.run.dir=.',
            'hydra/hydra_logging=disabled',
        ]
    main()
