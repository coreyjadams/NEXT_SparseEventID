import torch
import pytorch_lightning as pl




class lightning_trainer(pl.LightningModule):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, args, encoder, decoder, loss_calc,
                 acc_calc, lr_scheduler=None, ):
        super().__init__()

        self.args         = args
        self.encoder      = encoder
        self.decoder      = decoder
        self.lr_scheduler = lr_scheduler
        self.loss_calc    = loss_calc
        self.acc_calc     = acc_calc

    def forward(self, batch):
        encoded = self.encoder(batch)

        return network_dict


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.


        network_dict = self(batch["image"])
        prepped_labels = self.prep_labels(batch)

        loss, loss_metrics = self.loss_calc(prepped_labels, network_dict)

        acc_metrics = self.calculate_accuracy(network_dict, prepped_labels)
        

        self.print_log(acc_metrics, mode="train")
        self.summary(acc_metrics)
        self.log_dict(acc_metrics)
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

    if

        from src.networks.torch.uresnet2D import UResNet
        net = UResNet(args.network, image_size)

    else:
        if args.framework.sparse and args.mode.name != ModeKind.iotest:
            from src.networks.torch.sparseuresnet3D import UResNet3D
        else:
            from src.networks.torch.uresnet3D       import UResNet3D

        net = UResNet3D(args.network, image_size)

    return net

from . data import create_torch_larcv_dataloader
from src.networks.torch import create_vertex_meta

def create_lightning_module(args, datasets, lr_scheduler=None, log_keys = [], hparams_keys = []):

    # Going to build up the lightning module here.

    # Take the first dataset:
    example_ds = next(iter(datasets.values()))

    vertex_meta = create_vertex_meta(args, example_ds.image_meta, example_ds.image_size())

    # Turn the datasets into dataloaders:
    for key in datasets.keys():
        datasets[key] = create_torch_larcv_dataloader(
            datasets[key], args.run.minibatch_size)

    # Next, create the network:
    network = build_network(args, example_ds.image_size())

    if args.network.classification.active:
        weight = torch.tensor([0.16, 0.1666, 0.16666, 0.5])
        loss_calc = LossCalculator(args, weight=weight)
    else:
        loss_calc = LossCalculator(args)
    acc_calc = AccuracyCalculator(args)


    model = lightning_trainer(args, network, loss_calc,
        acc_calc, lr_scheduler,
        log_keys     = log_keys,
        hparams_keys = hparams_keys,
        vertex_meta  = vertex_meta)
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
        max_epochs              = 2,
        plugins                 = plugins,
        # benchmark               = True,
        accumulate_grad_batches = args.mode.optimizer.gradient_accumulation,
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=datasets["train"],
    )
