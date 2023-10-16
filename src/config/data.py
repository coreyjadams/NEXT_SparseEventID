from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any
from omegaconf import MISSING

# from . transforms import TransformChain

# dataset_top   = "/data/datasets/NEXT/"
dataset_top   = "/lus/grand/projects/datascience/cadams/datasets/NEXT/"
mc_bkg_dir    = dataset_top + "Background/NEXT_v1_05_02_NEXUS_v5_07_10_bkg_v9/larcv/merged_final/"
mc_tl_208_dir = dataset_top + "polarisProduction/simCLR_train/"
mc_mk_tl_208_dir = dataset_top + "dnn-dataset/simulation/larcv_2023/"
old_mk_tl208_dir = dataset_top + "dnn-dataset/simulation/outdated_larcv/"
# mc_tl_208_dir = dataset_top + "Calibration/NEXT_v1_05_02_NEXUS_v5_07_10_bkg_v9/larcv/merged/"


class RandomMode(Enum):
    random_blocks = 0
    serial_access = 1
    random_events = 2

@dataclass
class Data:
    name:        str = ""
    mc:         bool = False
    mode: RandomMode = RandomMode.random_events
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
class MCMKTl208(Data):
    name:  str = "mc_mk_tl208"
    mc:   bool = True
    # train: str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_train.h5"
    # test:  str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_test.h5"
    # val:   str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_val.h5"

    train: str = mc_mk_tl_208_dir + "representation_learning_tl208_cuts_train.h5"
    test:  str = mc_mk_tl_208_dir + ""
    val:   str = mc_mk_tl_208_dir + "representation_learning_tl208_cuts_val.h5"
    image_key:   str = "chitslowTh"


@dataclass
class OldMCMKTl208(Data):
    name:  str = "mc_mk_tl208"
    mc:   bool = True
    # train: str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_train.h5"
    # test:  str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_test.h5"
    # val:   str = mc_tl_208_dir + "Tl208_NEW_v1.2.0_v9.dhist_larcv_val.h5"

    train: str = old_mk_tl208_dir + "NEXT_White_train_randomized.h5"
    test:  str = old_mk_tl208_dir + ""
    val:   str = old_mk_tl208_dir + "run_6206_larcv_merged.h5"
    image_key:   str = "voxels"


@dataclass
class MCMKTl208_CLS(Data):
    name:  str = "mc_mk_tl208_cls"
    mc:   bool = True
    
    train: str = mc_mk_tl_208_dir + "eventID_tl208_cuts_train.h5"
    test:  str = mc_mk_tl_208_dir + ""
    val:   str = mc_mk_tl_208_dir + "eventID_tl208_cuts_val.h5"
    image_key:   str = "chitslowTh"

@dataclass
class MCBackground(Data):
    name:  str = "mc_bkg"
    mc:   bool = True
    train: str = mc_bkg_dir + "next_white_background_train.h5"
    test:  str = mc_bkg_dir + "next_white_background_test.h5"
    val:   str = mc_bkg_dir + "next_white_background_val.h5"




cs = ConfigStore.instance()
cs.store(group="data", name="mc_mk_tl208", node=MCMKTl208)
cs.store(group="data", name="mc_bkg",   node=MCBackground)
cs.store(group="data", name="mc_mk_tl208_cls", node = MCMKTl208_CLS)
cs.store(group="data", name="old_mc_mk_tl208_cls", node = OldMCMKTl208)