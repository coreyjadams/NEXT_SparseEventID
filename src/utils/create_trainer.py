import torch
import pytorch_lightning as pl


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

    oversubscribe = args.framework.oversubscribe

    # Distributed strategy:
    if args.run.distributed:
        from src.config import DistributedMode
        if args.framework.distributed_mode == DistributedMode.horovod:
            strategy = "horovod"
        elif args.framework.distributed_mode == DistributedMode.DDP:
            from pytorch_lightning.strategies import DDPStrategy
            backend = "nccl"
            if oversubscribe > 1:
                backend = "gloo"
            strategy = DDPStrategy(
                cluster_environment = OversubscribeMPIEnv(
                    oversubscribe),
                process_group_backend=backend
            )
        elif args.framework.distributed_mode == DistributedMode.deepspeed:
            strategy = "deepspeed"

        # devices   = int(os.environ['LOCAL_SIZE'])
        # num_nodes = int(os.environ['N_NODES'])
        plugins   = []
        # if args.run.compute_mode == ComputeMode.CUDA:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['LOCAL_RANK']
        #     devices=1
    else:
        from pytorch_lightning.strategies import SingleDeviceStrategy
        plugins   = []
        strategy  = SingleDeviceStrategy("cuda:0")
        devices   = 1
        num_nodes = 1

    # Configure the logger:
    from pytorch_lightning.loggers import TensorBoardLogger

    tb_logger = TensorBoardLogger(args.output_dir + "/train/")


    trainer = pl.Trainer(
        accelerator             = args.run.compute_mode.name.lower(),
        default_root_dir        = args.output_dir,
        precision               = precision,
        profiler                = profiler,
        strategy                = strategy,
        enable_progress_bar     = False,
        logger                  = tb_logger,
        log_every_n_steps       = 1,
        max_epochs              = args.run.length,
        # plugins                 = plugins,
        accumulate_grad_batches = args.mode.optimizer.gradient_accumulation,
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=datasets["train"],
        # val_dataloaders = datasets["val"]
    )
