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

class UpSampling(Enum):
    convolutional = 0
    interpolation = 1


class ConvMode(Enum):
    conv_2D = 0
    conv_3D = 1

class Norm(Enum):
    none  = 0
    batch = 1
    layer = 2


@dataclass
class Network:
    normalization:        Norm         = Norm.none
    bias:                 bool         = True
    blocks_per_layer:     int          = 2
    residual:             bool         = True
    weight_decay:         float        = 0.00
    conv_mode:            ConvMode     = ConvMode.conv_2D
    growth_rate:          GrowthRate   = GrowthRate.additive
    depth:                int          = 7
    n_initial_filters:    int          = 16

@dataclass
class Encoder(Network):
    name:                 str          = "encoder"
    downsampling:         DownSampling = DownSampling.max_pooling

@dataclass
class Decoder(Network):
    name:                 str          = "decoder"
    upsampling:           UpSampling   = UpSampling.interpolation

cs = ConfigStore.instance()
cs.store(group="network", name="encoder",   node=Encoder)
cs.store(group="network", name="decoder",   node=Decoder)
