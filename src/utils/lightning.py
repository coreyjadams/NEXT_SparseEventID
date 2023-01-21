import torch
import pytorch_lightning as pl

import datetime

from src.config import OptimizerKind
from src import logging

logger = logging.getLogger("NEXT")

class lightning_trainer(pl.LightningModule):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args, encoder, decoder,
                 image_key   = "pmaps",
                 lr_scheduler=None ):
        super().__init__()

        self.args         = args
        self.encoder      = encoder
        self.decoder      = decoder
        self.image_key    = image_key
        self.lr_scheduler = lr_scheduler

        self.log_keys = ["loss"]


    def forward(self, batch):

        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        return decoded


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.

        input_images   = batch[self.image_key]
        decoded_images = self(input_images)

        loss = self.calculate_ae_loss(input_images, decoded_images)

        metrics = {'loss' : loss}
        # self.log()
        self.print_log(metrics)
        self.log_dict(metrics)
        return loss

    def print_log(self, metrics, mode=""):

        if self.global_step % self.args.mode.logging_iteration == 0:

            self._current_log_time = datetime.datetime.now()

            # Build up a string for logging:
            if self.log_keys != []:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in self.log_keys])
            else:
                s = ", ".join(["{0}: {1:.3}".format(key, metrics[key]) for key in metrics])

            time_string = []

            if hasattr(self, "_previous_log_time"):
            # try:
                total_images = self.args.run.minibatch_size
                images_per_second = total_images / (self._current_log_time - self._previous_log_time).total_seconds()
                time_string.append("{:.2} Img/s".format(images_per_second))

            if 'io_fetch_time' in metrics.keys():
                time_string.append("{:.2} IOs".format(metrics['io_fetch_time']))

            if 'step_time' in metrics.keys():
                time_string.append("{:.2} (Step)(s)".format(metrics['step_time']))

            if len(time_string) > 0:
                s += " (" + " / ".join(time_string) + ")"

            # except:
            #     pass


            self._previous_log_time = self._current_log_time
            logger.info("{} Step {} metrics: {}".format(mode, self.global_step, s))


    def calculate_ae_loss(self, input_images, decoded_images):
        loss = torch.nn.functional.mse_loss(input_images, decoded_images)
        return loss

    def configure_optimizers(self):
        learning_rate = 1.0
        # learning_rate = self.args.mode.optimizer.learning_rate

        if self.args.mode.optimizer.name == OptimizerKind.rmsprop:
            opt = torch.optim.RMSprop(self.parameters(), learning_rate, eps=1e-6)
        elif self.args.mode.optimizer.name == OptimizerKind.adam:
            opt = torch.optim.Adam(self.parameters(), learning_rate, eps=1e-6, betas=(0.8,0.9))
        elif self.args.mode.optimizer.name == OptimizerKind.adagrad:
            opt = torch.optim.Adagrad(self.parameters(), learning_rate)
        elif self.args.mode.optimizer.name == OptimizerKind.adadelta:
            opt = torch.optim.Adadelta(self.parameters(), learning_rate, eps=1e-6)
        else:
            opt = torch.optim.SGD(self.parameters(), learning_rate)

        lr_fn = lambda x : self.lr_scheduler[x]

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn, last_epoch=-1)

        return [opt],[{"scheduler" : lr_scheduler, "interval": "step"}]


def build_networks(args, image_size):

    from src.networks import autoencoder
    models = autoencoder.create_models(args, image_size)

    return models

from src.io.data import create_torch_larcv_dataloader
# from src.networks import create_vertex_meta

def create_lightning_module(args, datasets, lr_scheduler=None, batch_keys=None):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))

    if 'pmaps' in batch_keys:
        image_key = 'pmaps'
    else:
        image_key = 'lr_hits'


    image_shape = example_ds.image_size(image_key)

    # vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Turn the datasets into dataloaders:
    for key in datasets.keys():
        datasets[key] = create_torch_larcv_dataloader(
            datasets[key], args.run.minibatch_size)

    print(next(iter(datasets['tl208']))['pmaps'])

    # Next, create the network:
    encoder, decoder = build_networks(args, image_shape)

    # if args.network.classification.active:
    #     weight = torch.tensor([0.16, 0.1666, 0.16666, 0.5])
    #     loss_calc = LossCalculator(args, weight=weight)
    # else:
    #     loss_calc = LossCalculator(args)
    # acc_calc = AccuracyCalculator(args)


    model = lightning_trainer(
        args, 
        encoder, 
        decoder,
        image_key,
        lr_scheduler
    )
    return model

def train(args, lightning_model, datasets):

    from src.config import Precision

    # Map the precision to lightning args:
    if args.run.precision == Precision.mixed:
        precision = 16
    elif args.run.precision == Precision.bfloat16:
        precision = "bf16"
    else:
        precision = 32

    # Map the profiling to lightning args:
    if args.run.profile:
        profiler = "advanced"
    else:
        profiler  = None

    # Distributed strategy:
    if args.run.distributed:
        from src.config import DistributedMode
        if args.framework.distributed_mode == DistributedMode.horovod:
            strategy = "horovod"
        elif args.framework.distributed_mode == DistributedMode.DDP:
            from pytorch_lightning.strategies import DDPStrategy
            strategy = DDPStrategy(
                cluster_environment = MPIClusterEnvironment()
            )
        elif args.framework.distributed_mode == DistributedMode.deepspeed:
            strategy = "deepspeed"

        devices   = int(os.environ['LOCAL_SIZE'])
        num_nodes = int(os.environ['N_NODES'])
        plugins   = []
        # if args.run.compute_mode == ComputeMode.CUDA:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
        #     devices=1
    else:
        plugins   = []
        strategy  = None
        devices   = 1
        num_nodes = 1

    # Configure the logger:
    from pytorch_lightning.loggers import TensorBoardLogger

    tb_logger = TensorBoardLogger(args.output_dir + "/train/")


    trainer = pl.Trainer(
        accelerator             = args.run.compute_mode.name.lower(),
        devices                 = devices,
        num_nodes               = num_nodes,
        auto_select_gpus        = True,
        default_root_dir        = args.output_dir,
        precision               = precision,
        profiler                = profiler,
        strategy                = strategy,
        enable_progress_bar     = False,
        replace_sampler_ddp     = True,
        logger                  = tb_logger,
        max_epochs              = args.run.length,
        plugins                 = plugins,
        # benchmark               = True,
        accumulate_grad_batches = args.mode.optimizer.gradient_accumulation,
    )

    print(datasets)

    trainer.fit(
        lightning_model,
        train_dataloaders=datasets["tl208"],
    )