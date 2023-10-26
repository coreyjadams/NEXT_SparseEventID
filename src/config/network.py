from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any
from omegaconf import MISSING


class GrowthRate(Enum):
    multiplicative = 0
    additive       = 1

class DownSampling(Enum):
    convolutional = 0
    pooling       = 1

class Norm(Enum):
    none  = 0
    batch = 1
    layer = 2
    group = 3


@dataclass
class Representation:
    normalization:        Norm         = Norm.batch
    bias:                 bool         = True
    blocks_per_layer:     int          = 4
    residual:             bool         = True
    weight_decay:         float        = 0.00
    growth_rate:          GrowthRate   = GrowthRate.additive
    downsampling:         DownSampling = DownSampling.convolutional
    depth:                int          = 4
    n_initial_filters:    int          = 8
    n_output_filters:     int          = 128

@dataclass
class ClassificationHead:
    layers: Tuple[int] = field(default_factory=list)

@dataclass
class YoloHead:
    layers: Tuple[int] = field(default_factory=list)

cs = ConfigStore.instance()
cs.store(group="network", name="representation", node=Representation)
cs.store(group="network", name="classification", node=ClassificationHead)
cs.store(group="network", name="yolo",           node=YoloHead)
