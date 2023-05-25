from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any
from omegaconf import MISSING

# from . transforms import TransformChain

# dataset_top   = "/data/datasets/NEXT/officialProduction/"
dataset_top   = "/lus/grand/projects/datascience/cadams/datasets/NEXT/polarisProduction/"
mc_bkg_dir    = dataset_top + "Background/NEXT_v1_05_02_NEXUS_v5_07_10_bkg_v9/larcv/merged_final/"
mc_tl_208_dir = dataset_top + "simCLR_train/"
# mc_tl_208_dir = dataset_top + "Calibration/NEXT_v1_05_02_NEXUS_v5_07_10_bkg_v9/larcv/merged/"


class RandomMode(Enum):
    random_blocks = 0
    serial_access = 1

@dataclass
class Data:
    name:        str = ""
    mc:         bool = False
    mode: RandomMode = RandomMode.random_blocks
    seed:        int = -1
    train:       str = ""
    test:        str = ""
    val:         str = ""
    image_key:   str = "pmaps"
    active: Tuple[str] = field(default_factory=list)
    transform1: bool = True
    transform2: bool = True

@dataclass
class MCTl208(Data):
    name:  str = "mc_tl_208"
    mc:   bool = True
    # train: str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_train.h5"
    # test:  str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_test.h5"
    # val:   str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_val.h5"

    train: str = mc_tl_208_dir + "NEXT_MC208_NN_larcv_all_train.h5"
    test:  str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_train.h5"
    val:   str = mc_tl_208_dir + "NEXT_MC208_NN_larcv_all_val.h5"


@dataclass
class MCBackground(Data):
    name:  str = "mc_bkg"
    mc:   bool = True
    train: str = mc_bkg_dir + "next_white_background_train.h5"
    test:  str = mc_bkg_dir + "next_white_background_test.h5"
    val:   str = mc_bkg_dir + "next_white_background_val.h5"




cs = ConfigStore.instance()
cs.store(group="data", name="mc_tl208", node=MCTl208)
cs.store(group="data", name="mc_bkg",   node=MCBackground)
