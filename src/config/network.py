from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any, List
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
    instance = 4

class EncoderType(Enum):
    resnet = 0
    mpnn   = 1
    vit    = 2
    cvt    = 3
    
class BlockStyle(Enum):
    none     = 0
    residual = 1
    convnext = 2

@dataclass
class Representation:
    depth:            int = 3
    type:     EncoderType = EncoderType.resnet
    bias:            bool = False

@dataclass
class ResNet(Representation):
    normalization:        Norm         = Norm.batch
    blocks_per_layer:     int          = 4
    block_style:          BlockStyle   = BlockStyle.residual
    filter_size:          int          = 3
    growth_rate:          GrowthRate   = GrowthRate.additive
    downsampling:         DownSampling = DownSampling.convolutional
    n_initial_filters:    int          = 32
    n_output_filters:     int          = 128

@dataclass
class ConvNext(Representation):
    normalization:        Norm         = Norm.group
    blocks_per_layer:     int          = 4
    block_style:          BlockStyle   = BlockStyle.convnext
    filter_size:          int          = 3
    growth_rate:          GrowthRate   = GrowthRate.additive
    downsampling:         DownSampling = DownSampling.convolutional
    n_initial_filters:    int          = 32
    n_output_filters:     int          = 128

@dataclass
class ViT(Representation):
    num_heads:    int = 8
    embed_dim:    int = 64
    patch_size:   int = 8
    type: EncoderType = EncoderType.vit
    depth:        int = 8
    dropout:    float = 0.5

@dataclass
class CvT(Representation):
    num_heads:        int = 8
    depth:            int = 2
    embed_dim:        int = 64
    type:     EncoderType = EncoderType.cvt
    blocks_per_layer: int = 2
    n_output_filters: int= 128

@dataclass
class MLPConfig():
    layers:     List[int] = field(default_factory=lambda: [16,])
    bias:            bool = True

@dataclass
class GraphRepresentation(Representation):
    mlp_config: MLPConfig = field(default_factory= lambda : MLPConfig(layers=[32,32]))
    graph_layer:      str = "GINConv"
    type:     EncoderType = EncoderType.mpnn


@dataclass
class ClassificationHead:
    layers: Tuple[int] = field(default_factory=list)

@dataclass
class YoloHead:
    layers: Tuple[int] = field(default_factory=list)

cs = ConfigStore.instance()
cs.store(group="encoder", name="resnet",      node=ResNet)
cs.store(group="encoder", name="convnext",    node=ConvNext)
cs.store(group="encoder", name="gnn",         node=GraphRepresentation)
cs.store(group="encoder", name="vit",         node=ViT)
cs.store(group="encoder", name="cvt",         node=CvT)
cs.store(group="head", name="classification", node=ClassificationHead)
cs.store(group="head", name="yolo",           node=YoloHead)
