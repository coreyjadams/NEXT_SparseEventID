from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any
from omegaconf import MISSING

class RandomMode(Enum):
    random_blocks = 0
    serial_access = 1

@dataclass
class Data:
    mc:         bool = False
    mode: RandomMode = RandomMode.random_blocks
    seed:        int = -1
    
@dataclass
class MCTl208(Data):
    path: str = "/data/datasets/NEXT/officialProduction/Calibration/NEXT_v1_05_02_NEXUS_v5_07_10_bkg_v9/larcv/Tl208_NEW_v1.2.0_v9.dhits_0.filtered_larcv.h5"
    mc:  bool = True



cs = ConfigStore.instance()
cs.store(group="data", name="mc_tl208", node=MCTl208)
# cs.store(group="data", name="val", node=Val)
# cs.store(group="data", name="test", node=Test)
# cs.store(group="data", name="synthetic", node=Synthetic)
