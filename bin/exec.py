#!/usr/bin/env python
import os,sys,signal
import time

import numpy

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)

# # import the necessary
# # from src.utils import flags
# from src.networks import resnet
# from src.networks import sparseresnet
# from src.networks import sparseresnet3d
# from src.networks import pointnet
# from src.networks import gcn
# from src.networks import dgcnn


import argparse

class exec(object):

    def __init__(self):

        # This technique is taken from: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
        parser = argparse.ArgumentParser(
            description='Run neural networks on NEXT Event ID dataset',
            usage='''exec.py <command> [<args>]

The most commonly used commands are:
   train-eventID         Train a network, either from scratch or restart
   train-cycleGAN        Train a network, either from scratch or restart
   inference-eventID     Run inference with a trained network
   inference-cycleGAN    Run inference with a trained network
   iotest                Run IO testing without training a network
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command.replace("-","_")):
            print(f'Unrecognized command {args.command}')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command.replace("-","_"))()

    def add_shared_training_arguments(self, parser):

        parser.add_argument('-lr','--learning-rate',
            type    = float,
            default = 0.003,
            help    = 'Initial learning rate')
        parser.add_argument('-ci','--checkpoint-iteration',
            type    = int,
            default = 500,
            help    = 'Period (in steps) to store snapshot of weights')
        parser.add_argument('-si','--summary-iteration',
            type    = int,
            default = 1,
            help    = 'Period (in steps) to store summary in tensorboard log')
        parser.add_argument('-li','--logging-iteration',
            type    = int,
            default = 1,
            help    = 'Period (in steps) to print values to log')
        parser.add_argument('-cd','--checkpoint-directory',
            type    = str,
            default = None,
            help    = 'Directory to store model snapshots')
        self.parser.add_argument('--optimizer',
            type    = str,
            choices = ['adam', 'rmsprop',],
            default = 'rmsprop',
            help    = 'Optimizer to use')
        self.parser.add_argument('--weight-decay',
            type    = float,
            default = 0.0,
            help    = "Weight decay strength")

    def train_cycleGAN(self):
        self.parser = argparse.ArgumentParser(
            description     = 'Run Network Training',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)

        self.add_io_arguments_cycleGAN(self.parser)
        self.add_core_configuration(self.parser)
        self.add_shared_training_arguments(self.parser)

        self.parser.add_argument('--cycle-lambda',
            type    = float,
            default = '10',
            help    = 'Lambda balancing between cycle loss and GAN loss')

        self.add_cycleGAN_parsers(self.parser)

        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = True


        self.make_trainer_cycleGAN()

        print("Running Training")
        print(self.__str__())

        self.trainer.initialize()
        self.trainer.batch_process()

    def train_eventID(self):
        self.parser = argparse.ArgumentParser(
            description     = 'Run Network Training',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)

        self.add_io_arguments_eventID(self.parser, training=True)
        self.add_core_configuration(self.parser)
        self.add_shared_training_arguments(self.parser)

        # Define parameters exclusive to training eventID:

        self.parser.add_argument('--lr-schedule',
            type    = str,
            choices = ['flat', '1cycle', 'triangle_clr', 'exp_range_clr', 'decay', 'expincrease'],
            default = 'flat',
            help    = 'Apply a learning rate schedule')



        self.add_eventID_parsers(self.parser)

        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = True


        self.make_trainer_eventID()

        print("Running Training")
        print(self.__str__())

        self.trainer.initialize()
        self.trainer.batch_process()

    def add_eventID_parsers(self, parser):

        # Add the sparse resnet:
        from src.networks.sparseresnet3d import ResNetFlags
        ResNetFlags().build_parser(parser)


    def add_cycleGAN_parsers(self, parser):

        # Add the sparse resnet:
        from src.networks.generator import GeneratorFlags
        GeneratorFlags().build_parser(parser)
        from src.networks.discriminator import DiscriminatorFlags
        DiscriminatorFlags().build_parser(parser)
        pass



    def iotest(self):
        self.parser = argparse.ArgumentParser(
            description     = 'Run IO Testing',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        self.add_io_arguments(self.parser)
        self.add_core_configuration(self.parser)

        # now that we're inside a subcommand, ignore the first
        # TWO argvs, ie the command (exec.py) and the subcommand (iotest)
        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = False
        print("Running IO Test")
        print(self.__str__())

        self.make_trainer()

        self.trainer.initialize(io_only=True)

        # label_stats = numpy.zeros((36,))

        time.sleep(0.1)
        for i in range(self.args.iterations):
            start = time.time()
            mb = self.trainer.fetch_next_batch()
            # print(mb.keys())
            # label_stats += numpy.sum(mb['label'], axis=0)

            end = time.time()
            if not self.args.distributed:
                print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            else:
                if self.trainer._rank == 0:
                    print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            # time.sleep(0.5)
        # print(label_stats)

    def make_trainer_eventID(self):

        if self.args.distributed:
            from src.utils import distributed_eventID

            self.trainer = distributed_eventID.distributed_eventID(self.args)
        else:
            from src.utils import trainer_eventID
            self.trainer = trainer_eventID.trainer_eventID(self.args)

    def make_trainer_cycleGAN(self):

        if self.args.distributed:
            from src.utils import distributed_trainer

            self.trainer = distributed_trainer.distributed_trainer(self.args)
        else:
            from src.utils import trainer_cycleGAN
            self.trainer = trainer_cycleGAN.trainer_cycleGAN(self.args)


    def inference_cycleGAN(self):
        pass


    def inference_eventID(self):
        self.parser = argparse.ArgumentParser(
            description     = 'Run Network Training',
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)

        self.add_io_arguments_eventID(self.parser, training=False)
        self.add_core_configuration(self.parser)
        self.add_shared_training_arguments(self.parser)


        self.add_eventID_parsers(self.parser)

        self.args = self.parser.parse_args(sys.argv[2:])
        self.args.training = False


        self.make_trainer_eventID()

        print("Running Inference")
        print(self.__str__())

        self.trainer.initialize()
        self.trainer.batch_process()

    def __str__(self):
        s = "\n\n-- CONFIG --\n"
        for name in iter(sorted(vars(self.args))):
            # if name != name.upper(): continue
            attribute = getattr(self.args,name)
            # if type(attribute) == type(self.parser): continue
            # s += " %s = %r\n" % (name, getattr(self, name))
            substring = ' {message:{fill}{align}{width}}: {attr}\n'.format(
                   message=name,
                   attr = getattr(self.args, name),
                   fill='.',
                   align='<',
                   width=30,
                )
            s += substring
        return s




    def add_core_configuration(self, parser):
        # These are core parameters that are important for all modes:
        parser.add_argument('-i', '--iterations',
            type    = int,
            default = 5000,
            help    = "Number of iterations to process")

        parser.add_argument('-d','--distributed',
            action  = 'store_true',
            default = False,
            help    = "Run with the MPI compatible mode")
        parser.add_argument('-m','--compute-mode',
            type    = str,
            choices = ['CPU','GPU'],
            default = 'GPU',
            help    = "Selection of compute device, CPU or GPU ")
        parser.add_argument('-ld','--log-directory',
            default ="log/",
            help    ="Prefix (directory) for logging information")


        return parser

    def add_io_arguments_eventID(self, parser, training):

        data_directory = "/home/cadams/NEXT/cycleGAN/"

        # IO PARAMETERS FOR INPUT:

        if training:
            parser.add_argument('-f','--train-file',
                type    = str,
                default = data_directory + "next_new_classification_train.h5",
                help    = "IO Input File")

            # IO PARAMETERS FOR AUX INPUT:
            parser.add_argument('--test-file',
                type    = str,
                default = data_directory + "next_new_classification_test.h5",
                help    = "IO Aux Input File, or output file in inference mode")

        else:

            parser.add_argument('-f','--sim-file',
                type    = str,
                default = data_directory + "next_new_classification_val.h5",
                help    = "IO Input File")

            # IO PARAMETERS FOR AUX INPUT:
            parser.add_argument('--data-file',
                type    = str,
                default = data_directory + "nextDATA_RUNS.h5",
                help    = "IO Aux Input File, or output file in inference mode")

        parser.add_argument('-mb','--minibatch-size',
            type    = int,
            default = 2,
            help    = "Number of images in the minibatch size")

        parser.add_argument('--aux-iteration',
            type    = int,
            default = 10,
            help    = "Iteration to run the aux operations")

        parser.add_argument('--aux-minibatch-size',
            type    = int,
            default = 2,
            help    = "Number of images in the minibatch size")

        return

    def add_io_arguments_cycleGAN(self, parser):
        data_directory="/lus/theta-fs0/projects/datascience/cadams/datasets/NEXT/cycleGAN/"

        # IO PARAMETERS FOR DATA INPUT:
        parser.add_argument('--data-file',
            type    = str,
            default = data_directory + "nextDATA_RUNS.h5",
            help    = "Real data file")

        parser.add_argument('-mb','--minibatch-size',
            type    = int,
            default = 2,
            help    = "Number of images in the minibatch size")

        parser.add_argument('--sim-file',
            type    = str,
            default = data_directory + "next_new_classification_val.h5",
            help    = "Simulated data file")

        return



if __name__ == '__main__':
    s = exec()
