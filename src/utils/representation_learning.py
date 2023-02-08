import torch
import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import datetime

from . training_utils import init_optimizer, format_log_message

from src import logging

logger = logging.getLogger("NEXT")

class lightning_trainer(pl.LightningModule):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args, encoder, head, transforms,
                 image_key   = "pmaps",
                 lr_scheduler=None ):
        super().__init__()

        self.args         = args
        self.encoder      = encoder
        self.head         = head
        self.transforms   = transforms
        self.image_key    = image_key
        self.lr_scheduler = lr_scheduler

        self.log_keys = ["loss"]


    def forward(self, batch):

        # print(batch)
        # print(self.transforms)


        # batch = (
        #     batch[0].cpu().long(),
        #     batch[1],
        #     batch[2]
        # )

        augmented_data = [ t(batch) for t in self.transforms]

        # print(augmented_data)

        representation = [ self.encoder(ad) for ad in augmented_data ]

        logits = [ self.head(r) for r in representation]

        # print(representation)

        return logits


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        image = batch[self.image_key]

        decoded_images = self(image)

        if self.args.framework.sparse:
            loss = self.calculate_ae_loss(batch.features, decoded_images.features)
        else:
            loss = self.calculate_ae_loss(batch, decoded_images)

        metrics = {
            'loss' : loss,
            'lr' : self.optimizers().state_dict()['param_groups'][0]['lr']
        }

        # self.log()
        self.print_log(metrics)
        self.log_dict(metrics)
        return loss

    def print_log(self, metrics, mode=""):

        if self.global_step % self.args.mode.logging_iteration == 0:
        
            message = format_log_message(
                self.log_keys, 
                metrics, 
                self.run.minibatch_size, 
                self.global_step(), 
                mode
            )

            logger.info(message)

    def exit(self):
        pass

    def calculate_ae_loss(self, input_images, decoded_images):
        loss = torch.nn.functional.mse_loss(input_images, decoded_images)
        return loss

    def configure_optimizers(self):
        learning_rate = 1.0
        # learning_rate = self.args.mode.optimizer.learning_rate
        opt = init_optimizer(self.args.mode.optimizer.name, self.parameters())

        lr_fn = lambda x : self.lr_scheduler[x]

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn, last_epoch=-1)

        return [opt],[{"scheduler" : lr_scheduler, "interval": "step"}]


# from src.networks import create_vertex_meta

# from lightning_fabric.plugins.environments import MPIEnvironment

# class OversubscribeMPIEnv(MPIEnvironment):

#     def __init__(self, oversubscribe=None, **kwargs):
#         super().__init__(**kwargs)
#         self.o = oversubscribe

#     def local_rank(self):
#         lr = super().local_rank()
#         if self.o is None: return lr
#         return lr // self.o

def create_lightning_module(args, datasets, transforms, lr_scheduler=None, batch_keys=None):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))


    image_shape = example_ds.dataset.image_size(args.data.image_key)

    # vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Next, create the network:
    from src.networks import autoencoder
    encoder, classification_head = autoencoder.create_models(args, image_shape)

    model = lightning_trainer(
        args,
        encoder,
        classification_head,
        transforms,
        args.data.image_key,
        lr_scheduler
    )
    return model
