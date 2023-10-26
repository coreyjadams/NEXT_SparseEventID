from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Any, Tuple
from omegaconf import MISSING

from .network   import Representation, ClassificationHead, YoloHead
from .mode      import Mode
from .framework import Framework
from .data      import Data


class ComputeMode(Enum):
    CPU   = 0
    CUDA  = 1
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
    length:             int         = 20
    minibatch_size:     int         = 2
    id:                 str         = MISSING
    precision:          Precision   = Precision.float32
    profile:            bool        = False
    world_size:         int         = 1

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
]

@dataclass
class LearnRepresentation:
    defaults: List[Any] = field(default_factory=lambda: defaults)


    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Data      = MISSING
    framework:  Framework = MISSING
    encoder:    Representation = field(default_factory= lambda : Representation())
    head:       ClassificationHead = field(default_factory= lambda : ClassificationHead())
    output_dir: str       = "output/"
    name:       str       = "simclr"

cs.store(name="representation", node=LearnRepresentation)

@dataclass
class DetectVertex:
    defaults: List[Any] = field(default_factory=lambda: defaults)


    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Data      = MISSING
    framework:  Framework = MISSING
    encoder:    Representation = field(default_factory= lambda : Representation())
    head:       YoloHead  = field(default_factory= lambda : YoloHead())
    output_dir: str       = "output/"
    name:       str       = "yolo"


@dataclass
class SupervisedClassification:
    defaults: List[Any] = field(default_factory=lambda: defaults)


    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Data      = MISSING
    framework:  Framework = MISSING
    encoder:    Representation = field(default_factory= lambda : Representation())
    head:       ClassificationHead = field(default_factory= lambda : ClassificationHead())
    output_dir: str       = "output/"
    name:       str       = "supervised_eventID"

@dataclass
class UnsupervisedClassification:
    defaults: List[Any] = field(default_factory=lambda: defaults)


    run:        Run       = MISSING
    mode:       Mode      = MISSING
    data:       Data      = MISSING
    framework:  Framework = MISSING
    encoder:    Representation = field(default_factory= lambda : Representation())
    head:       ClassificationHead = field(default_factory= lambda : ClassificationHead())
    output_dir: str       = "output/"
    name:       str       = "unsupervised_eventID"


cs.store(name="supervised_classification",   node=SupervisedClassification)
cs.store(name="unsupervised_classification", node=UnsupervisedClassification)
cs.store(name="detect_vertex",               node=DetectVertex)
