import torch
import pytorch_lightning as pl

# torch.set_float32_matmul_precision('high')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import datetime

from . training_utils import init_optimizer, format_log_message

from src.config.optimizer import LossBalanceScheme

from src import logging

logger = logging.getLogger("NEXT")

class supervised_eventID(pl.LightningModule):
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

        if args.mode.optimizer.loss_balance_scheme == LossBalanceScheme.even:
            weight = torch.tensor([0.582, 1.417])
        else: weight = None

        self.criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)

    def on_train_start(self):
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def forward(self, batch):


        representation = self.encoder(batch)

        print(representation.shape)
        # logits = representation
        logits = self.head(representation)
        print(logits.shape)
        return logits


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        image = batch[self.image_key]

        logits = self(image)

        prediction = self.predict_event(logits)
        loss = self.calculate_loss(batch, logits, prediction)

        accuracy_dict = self.calculate_accuracy(prediction, batch['label'])



        metrics = {
            'loss/loss' : loss,
            'opt/lr' : self.optimizers().state_dict()['param_groups'][0]['lr']
        }


        metrics.update(accuracy_dict)

        # self.log()
        self.print_log(metrics, mode="train")
        metrics = { "/train/" + key : metrics[key] for key in metrics}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):


        image = batch[self.image_key]

        logits = self(image)


        prediction = self.predict_event(logits)
        loss = self.calculate_loss(batch, logits, prediction)

        accuracy_dict = self.calculate_accuracy(prediction, batch['label'])



        metrics = {
            'loss/loss' : loss,
            'opt/lr' : self.optimizers().state_dict()['param_groups'][0]['lr']
        }


        metrics.update(accuracy_dict)

        # self.log()
        self.print_log(metrics, mode="val")
        metrics = { "/val/" + key : metrics[key] for key in metrics}
        self.log_dict(metrics, logger)
        return



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

    def unravel_index(self,index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    def predict_event(self, prediction):

        batch_size = prediction.shape[0]

        class_prediction = torch.argmax(prediction, dim=-1)

        return class_prediction


    def calculate_accuracy(self, prediction, labels):
        SIGNAL = 1
        BACKGROUND = 0

        accuracy = prediction == labels

        is_signal     = labels == SIGNAL
        is_background = labels == BACKGROUND

        sig_acc  = torch.mean(accuracy[is_signal].to(torch.float32))
        bkg_acc = torch.mean(accuracy[is_background].to(torch.float32))
        accuracy = torch.mean(accuracy.to(torch.float32))


        return {
            "acc/accuracy" : accuracy,
            "acc/sig_acc": sig_acc,
            "acc/bkg_acc": bkg_acc,
        }


    def calculate_loss(self, batch, logits, prediction=None):

        if self.args.mode.optimizer.loss_balance_scheme == LossBalanceScheme.focal:
            # This section computes the loss via focal loss, since the classes are imbalanced:
            # Create the full label:
            y = torch.nn.functional.one_hot(batch["label"], logits.size(-1))
            softmax = torch.nn.functional.softmax(logits) 
            softmax = softmax.clamp(1e-7, 1. - 1e-7)
            loss = - y * torch.log(softmax)

            # Apply the focal part:
            loss = loss * (1 - softmax)**2
            loss = loss.sum(axis=-1).mean()
        else:
            # print(logits.shape)
            # print(batch['label'].shape)
            # print(batch['label'])
            loss = self.criterion(logits, target = batch["label"])

        return loss


    def configure_optimizers(self):
        learning_rate = 1.0
        # learning_rate = self.args.mode.optimizer.learning_rate
        opt = init_optimizer(self.args.mode.optimizer.name, self.parameters(), self.args.mode.optimizer.weight_decay)

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
    from src.networks.classification_head import build_networks
    encoder, class_head = build_networks(args, image_shape)

    # For this, we use the augmented data if its available:
    image_key = args.data.image_key
    if args.data.transform1:
        image_key += "_1"
    elif args.data.transform2:
        image_key += "_2"

    model = supervised_eventID(
        args,
        encoder,
        class_head,
        transforms,
        image_meta,
        image_key,
        lr_scheduler,
    )
    return model
