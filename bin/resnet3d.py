#!/usr/bin/env python
import os,sys,signal
import time

import numpy

# Add the local folder to the import path:
network_dir = os.path.dirname(os.path.abspath(__file__))
network_dir = os.path.dirname(network_dir)
sys.path.insert(0,network_dir)


# import the necessary
from src.utils import flags
from src.networks import resnet3d
try:
    from src.networks import sparseresnet3d
except:
    print("Couldn't import sparseresnet3d")

def main():

    FLAGS = flags.resnet3D()
    FLAGS.parse_args()
    # FLAGS.dump_config()

    


    

    if FLAGS.MODE is None:
        raise Exception()

    if FLAGS.DISTRIBUTED:
        from src.utils import distributed_trainer

        trainer = distributed_trainer.distributed_trainer()
    else:
        from src.utils import trainercore
        trainer = trainercore.trainercore()
        
    if FLAGS.MODE == 'train' or FLAGS.MODE == 'inference':
        if FLAGS.SPARSE:
            net = sparseresnet3d.ResNet
        else:
            net = resnet3d.ResNet
        FLAGS.set_net(net)
        trainer.initialize()
        trainer.batch_process()

    if FLAGS.MODE == 'iotest':
        trainer.initialize(io_only=True)

        total_start_time = time.time()
        # time.sleep(0.1)
        start = time.time()
        for i in range(FLAGS.ITERATIONS):
            mb = trainer.fetch_next_batch()
            end = time.time()
            if not FLAGS.DISTRIBUTED:
                print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            else:
                if trainer._rank == 0:
                    print(i, ": Time to fetch a minibatch of data: {}".format(end - start))
            # time.sleep(0.5)
            start = time.time()

        total_time = time.time() - total_start_time
        print("Time to read {} batches of {} images each: {}".format(
            FLAGS.ITERATIONS, 
            FLAGS.MINIBATCH_SIZE,
            time.time() - total_start_time
            ))

    trainer.stop()

if __name__ == '__main__':
    main()