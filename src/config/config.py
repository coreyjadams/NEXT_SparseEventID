from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any, Tuple
from omegaconf import MISSING

from .network   import Network, Encoder, Decoder
from .mode      import Mode
from .framework import Framework
from .data      import Data

class ComputeMode(Enum):
    CPU   = 0
    CUDA  = 1
    DPCPP = 2
    XPU   = 3

class Precision(Enum):
    float32  = 0
    mixed    = 1
    bfloat16 = 2
    float16  = 3


@dataclass
class Run:
    distributed:        bool        = True
    compute_mode:       ComputeMode = ComputeMode.CUDA
    run_length:         int         = 20
    minibatch_size:     int         = 2
    id:                 str         = MISSING
    precision:          Precision   = Precision.float32
    profile:            bool        = False

cs = ConfigStore.instance()

cs.store(group="run", name="base_run", node=Run)

cs.store(
    name="disable_hydra_logging",
    group="hydra/job_logging",
    node={"version": 1, "disable_existing_loggers": False, "root": {"handlers": []}},
)


defaults = [
    {"run"       : "base_run"},
    {"mode"      : "train"},
    {"data"      : "mc_tl208"},
    {"framework" : "lightning"},
    # {"encoder"   : "encoder"},
    # {"decoder"   : "decoder"},
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)


    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Data      = MISSING
    framework:  Framework = MISSING
    # encoder:    Network   = Encoder()
    # decoder:    Network   = Decoder()
    output_dir: str       = "output/"


cs.store(name="base_config", node=Config)
