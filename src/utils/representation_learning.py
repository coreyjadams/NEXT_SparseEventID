import torch
import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import datetime

from . training_utils import init_optimizer, format_log_message

from src import logging

logger = logging.getLogger("NEXT")

class rep_trainer(pl.LightningModule):
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

        self.save_hyperparameters()

    def on_train_start(self):
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    # @profile
    def forward(self, augmented_images):

        # print(batch.keys())



        representation = [ self.encoder(ad) for ad in augmented_images ]


        logits = [ self.head(r) for r in representation]

        return logits

    # @profile
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        augmented_images = [batch[k] for k in self.transforms]

        encoded_images = self(augmented_images)

        loss, loss_metrics = self.calculate_loss(encoded_images[0], encoded_images[1])

        metrics = {
            'opt/loss' : loss,
            'opt/lr' : self.optimizers().state_dict()['param_groups'][0]['lr'],
            'opt/alignment' : loss_metrics["alignment"],
            'opt/log_sum_exp' : loss_metrics["log_sum_exp"]
        }

        # self.log()
        self.print_log(metrics, mode="train")
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):

        augmented_images = [batch[k] for k in self.transforms]

        encoded_images = self(augmented_images)

        loss, loss_metrics = self.calculate_loss(encoded_images[0], encoded_images[1])

        metrics = {
            'opt/loss' : loss,
        }

        # self.log()
        self.print_log(metrics, mode="val")
        # self.log_dict(metrics)
        return loss


    def print_log(self, metrics, mode=""):

        if self.global_step % self.args.mode.logging_iteration == 0:

            message = format_log_message(
                self.log_keys,
                metrics,
                self.args.run.minibatch_size,
                self.global_step,
                mode
            )

            logger.info(message)

    def exit(self):
        pass

    def calculate_loss(self, first_images, second_images, temperature=0.1):
        # Each image is represented with k parameters,
        # Assume the batch size is N, so the
        # inputs have shape (N, k)

        N = first_images.shape[0]
        k = first_images.shape[1]

        # Need to dig in here to fix the loss function:
        # https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7

        # Also it has LARS to use.

        # print(first_images)
        # print("Min first_images: ", torch.min(first_images))
        # print("Max first_images: ", torch.max(first_images))
        # print("Min second_images: ", torch.min(second_images))
        # print("Max second_images: ", torch.max(second_images))


        first_images = first_images / torch.norm(first_images,dim=1).reshape((-1,1))
        second_images = second_images / torch.norm(second_images,dim=1).reshape((-1,1))

        # Take the two tuples, and concatenate them.
        # Then, reshape into Y = (1, 2N, k) and Z = (2N, 1, k)

        c = torch.concat([first_images, second_images], dim=0)

        Y = c.reshape((1, c.shape[0], c.shape[1]))
        Z = c.reshape((c.shape[0], 1, c.shape[1]))



        # Compute the product of these tensors, which gives shape
        # (2N, 2N, k)
        mat =  Y*Z


        # We need to compute the function (sim(x,y)) for each element in the 2N sequent.
        # Since the are normalized, we're computing x^T . Y / (||x||*||y||),
        # but the norms are equal to 1.
        # So, summing the matrix over the dim = 0 and dim = 1 computes this for each pair.

        sim = torch.sum(mat, dim=-1)


        # This yields a symmetric matrix, diagonal entries equal 1.  Off diagonal are symmetrics and < 1.

        # sim = torch.exp(sim / temperature)
        # Now, for every entry i in C (concat of both batches), the sum of sim[i] - sim[i][i] is the denominator

        device = sim.device

        positive = torch.tile(torch.eye(N, device=device), (2,2))
        # Unsure if this line is needed?
        positive = positive - torch.eye(2*N, device=device)

        negative = - (torch.eye(2*N, device=device) - 1)




        negative_examples = sim * negative
        positive_examples = sim * positive



        # negative_factor = torch.sum(negative)

        # Include a corrective factor that accounts for the fact that the loss has a floor > 0:
        #
        # print("Min positive: ", torch.min(positive_examples))
        # print("Max positive: ", torch.max(positive_examples))
        #
        #
        # print("Min negative: ", torch.min(negative_examples))
        # print("Max negative: ", torch.max(negative_examples))


        alignment = torch.sum(positive_examples, dim=0)
        # print(f"alignment: {alignment}")

        exp = torch.sum(torch.exp(negative_examples), dim=0)

        # print(exp)


        log_sum_exp = torch.log(exp + 1e-8)

        loss_metrics = {
            "alignment"   : torch.mean(alignment),
            "log_sum_exp" : torch.mean(log_sum_exp)
        }

        loss = torch.mean( - alignment + log_sum_exp)

        # loss = torch.mean( - torch.log(ratio))
        # loss = torch.mean( - torch.log(ratio)) - torch.log(batch_size) + 1.

        return loss, loss_metrics

    def configure_optimizers(self):
        learning_rate = 1.0
        # learning_rate = self.args.mode.optimizer.learning_rate
        opt = init_optimizer(self.args.mode.optimizer.name, self.parameters())

        lr_fn = lambda x : self.lr_scheduler[x]

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn, last_epoch=-1)

        return [opt],[{"scheduler" : lr_scheduler, "interval": "step"}]

def create_lightning_module(args, datasets, transforms, lr_scheduler=None, batch_keys=None):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))


    image_shape = example_ds.dataset.image_size(args.data.image_key)

    # vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Next, create the network:
    from src.networks import classification_head
    encoder, classification_head = classification_head.build_networks(args, image_shape)

    model = rep_trainer(
        args,
        encoder,
        classification_head,
        transforms,
        args.data.image_key,
        lr_scheduler
    )
    return model
