from enum import Enum

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Tuple, Any
from omegaconf import MISSING


# Go through a series of locations where the data might be stored.
import os
data_top = "/missing/data/folder/"
for test_path in [
    "/data/datasets/NEXT/",
]:
    if os.path.isdir(test_path):
        data_top = test_path
        break


class RandomMode(Enum):
    random_blocks = 0
    serial_access = 1
    random_events = 2

class Detector(Enum):
    next_white = 0
    next_100   = 1
    atpc       = 2

@dataclass
class Data:
    name:        str = ""
    vertex:     bool = False
    mode: RandomMode = RandomMode.random_events
    seed:        int = -1
    image_key:   str = "chitslowTh"
    active: Tuple[str] = field(default_factory=list)
    normalize:  bool = True 
    transform1: bool = True
    transform2: bool = True
    detector: Detector = Detector.next_white

@dataclass
class NEXT_White_Supervised(Data):
    name:  str = "white-supervised"
    sim_train: str = data_top + "NEXT-White-DNN-dataset/sim/eventID_tl208_cuts_train.h5"
    sim_val:   str = data_top + "NEXT-White-DNN-dataset/sim/eventID_tl208_cuts_val.h5"
    sim_test:  str = data_top + "NEXT-White-DNN-dataset/sim/eventID_tl208_cuts_test.h5"
    sim_comp:  str = data_top + "NEXT-White-DNN-dataset/sim/r6206_new_tl208_cuts.h5"
    data_comp: str = data_top + "NEXT-White-DNN-dataset/data/larcv_merged_r7473_new_tl208_cuts_0.h5"
    data_runs: str = data_top + "NEXT-White-DNN-dataset/data/larcv_merged_runs_7471-2-3_tl208_cuts.h5"
    active: Tuple[str] = field(default_factory= lambda : ["sim_train", "sim_val", "sim_test", "sim_comp", "data_comp", "data_runs"])

    
@dataclass
class NEXT_White_Representation(Data):
    name:  str = "white-repr"
    sim_train: str = data_top + "NEXT-White-DNN-dataset/sim/representation_learning_tl208_all_train.h5"
    sim_val:   str = data_top + "NEXT-White-DNN-dataset/sim/representation_learning_tl208_all_val.h5"
    sim_comp:  str = data_top + "NEXT-White-DNN-dataset/sim/r6206_new_tl208_all.h5"
    data_comp: str = data_top + "NEXT-White-DNN-dataset/data/larcv_merged_r7473_new_tl208_all_0.h5"
    data_runs: str = data_top + "NEXT-White-DNN-dataset/data/larcv_merged_runs_7471-2-3_tl208_all.h5"
    active: Tuple[str] = field(default_factory= lambda : ["sim_train", "sim_val", "sim_comp", "data_comp", "data_runs"])




# @dataclass
# class NEXT100Sim(Data):
#     name:      str = "next100_sim"
#     mc:       bool = True
#     train:     str = next_100_dir + "next100_train.h5"
#     val:       str = next_100_dir + "next100_val.h5"
#     image_key: str = "lr_hits"
#     vertex:   bool = True
#     detector: Detector = Detector.next_100

# @dataclass
# class ATPC_0nubb(Data):
#     name:      str = "ATPC_0nubb"
#     mc:       bool = True
#     train:     str = ATPC_dir + "atpc_0nubb_train.h5"
#     val:       str = ATPC_dir + "atpc_0nubb_val.h5"
#     image_key: str = "depositions"
#     vertex:   bool = False
#     detector: Detector = Detector.atpc

cs = ConfigStore.instance()
cs.store(group="data", name="next-white-supervised", node=NEXT_White_Supervised)
cs.store(group="data", name="next-white-representation", node=NEXT_White_Representation)
# cs.store(group="data", name="mc_mk_tl208", node=MCMKTl208)
# cs.store(group="data", name="mc_bkg",   node=MCBackground)
# cs.store(group="data", name="mc_mk_tl208_cls", node = MCMKTl208_CLS)
# cs.store(group="data", name="old_mc_mk_tl208_cls", node = OldMCMKTl208)
# cs.store(group="data", name="next100_sim", node = NEXT100Sim)
# cs.store(group="data", name="atpc_0nubb", node = ATPC_0nubb)

