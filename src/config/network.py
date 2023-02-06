from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class GrowthRate(Enum):
    multiplicative = 0
    additive       = 1

class DownSampling(Enum):
    convolutional = 0
    max_pooling   = 1

class Norm(Enum):
    none  = 0
    batch = 1
    layer = 2


@dataclass
class Representation:
    normalization:        Norm         = Norm.none
    bias:                 bool         = True
    blocks_per_layer:     int          = 2
    residual:             bool         = True
    weight_decay:         float        = 0.00
    growth_rate:          GrowthRate   = GrowthRate.additive
    downsampling:         DownSampling = DownSampling.convolutional
    depth:                int          = 4
    n_initial_filters:    int          = 16

@dataclass
class ClassificationHead:
    n_layers:     int = 2
    output_space: int = 256


cs = ConfigStore.instance()
cs.store(group="network", name="representation",   node=Representation)
cs.store(group="network", name="classification",   node=ClassificationHead)
