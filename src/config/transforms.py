# from enum import Enum

# from dataclasses import dataclass, field
# from hydra.core.config_store import ConfigStore
# from typing import Tuple, Any
# from omegaconf import MISSING

# # class TransformKind(Enum):
# #     identity = 0
# #     flip     = 1
# #     rescale  = 2
# #     resize   = 3
# #     shift    = 4



# @dataclass 
# class Transform:
#     # kind: TransformKind : TransformKind.identity 
#     # name: str = ""
#     pass

# @dataclass
# class Flip(Transform):
#     axes: Tuple[int] = field(default_factory=list)
#     random:     bool = False

# # @dataclass
# # class Transform:
# #     transforms : Tuple[TransformBase] =  field(default_factory=list)

# # class Transform2: 
#     # transforms : Tuple[TransformBase] =  field(default_factory=list)

# cs = ConfigStore.instance()
# cs.store(group="transforms", name="flip", node=Flip)
# # cs.store(group="transforms", name="transform2", node=Transform2)
# cs.store(name="transforms", node=Transform)
