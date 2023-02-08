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
                 image_meta,
                 image_key   = "pmaps",
                 lr_scheduler=None): 
        super().__init__()

        self.args         = args
        self.transforms   = transforms
        self.encoder      = encoder
        self.head         = head
        self.image_key    = image_key
        self.lr_scheduler = lr_scheduler

        self.image_size   = torch.tensor(image_meta['size'][0])
        self.image_origin = torch.tensor(image_meta['origin'][0])

        self.log_keys = ["loss"]


    def forward(self, batch):


        t = self.transforms[0](batch)

        representation = self.encoder(t)

        logits = self.head(representation)


        return logits


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        image = batch[self.image_key]

        prediction = self(image)

        loss = self.vertex_loss(batch, prediction)


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
                log_keys = self.log_keys, 
                metrics  = metrics, 
                batch_size = self.args.run.minibatch_size, 
                global_step = self.global_step, 
                mode = mode
            )

            logger.info(message)

    def exit(self):
        pass

    def vertex_loss(self, batch, prediction):

        self.image_size   = self.image_size.to(prediction.device)
        self.image_origin = self.image_origin.to(prediction.device)

        # Start with the prediction shape to define the labels
        class_labels = torch.zeros_like(prediction[:,0,:,:,:])
        regression_labels = torch.zeros_like(prediction[:,1:,:,:,:])

        vertex_label = batch['vertex']

        # The first goal is to figure out which binary labels to turn on, if any.
        # We take the 3D vertex location and figure out which pixel it aligns with in X/Y/Z.

        # the size of each anchor box can be found first:
        if not hasattr(self, "anchor_size"):
            labels_spatial_size = torch.tensor(class_labels.shape[-3:]).to(prediction.device)
            self.anchor_size = torch.tensor(self.image_size / labels_spatial_size)

        relative_vertex = batch['vertex'][:,0:3] - self.image_origin


        anchor_index = (relative_vertex // self.anchor_size).to(torch.int64)

        active_anchors = batch['vertex'][:,2] != 0.0

        # Select the target anchor indexes:
        target_anchors = anchor_index[active_anchors]

        # Set the class labels
        class_labels[active_anchors,target_anchors[:,0],target_anchors[:,1],target_anchors[:,2]] = 1.
        class_prediction = prediction[:,0,:,:,:]

        print(torch.max(class_prediction))

        # Compute the loss:
        class_loss = torch.nn.functional.binary_cross_entropy(class_prediction, class_labels, reduction="none")
        focus = (class_prediction - class_labels)**2

        anchor_loss = torch.mean(class_loss*focus)

        # # Now, set the regression anchors

        # print(class_labels[active_anchors,target_anchors[:,0],target_anchors[:,1],target_anchors[:,2]])        

        # print(torch.where(class_labels != 0))


        # Now, set the binary labels:
        # batch_index =

        return anchor_loss


    def configure_optimizers(self):
        learning_rate = 1.0
        # learning_rate = self.args.mode.optimizer.learning_rate
        opt = init_optimizer(self.args.mode.optimizer.name, self.parameters())

        lr_fn = lambda x : self.lr_scheduler[x]

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn, last_epoch=-1)

        return [opt],[{"scheduler" : lr_scheduler, "interval": "step"}]


def create_lightning_module(args, datasets, transforms=None, lr_scheduler=None, batch_keys=None):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))


    image_shape = example_ds.dataset.image_size(args.data.image_key)
    image_meta = example_ds.dataset.image_meta(args.data.image_key)
    # vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Next, create the network:
    from src.networks.yolo_head import build_networks
    encoder, yolo_head = build_networks(args, image_shape)

    model = lightning_trainer(
        args,
        encoder,
        yolo_head,
        transforms,
        image_meta,
        args.data.image_key,
        lr_scheduler,
    )
    return model
