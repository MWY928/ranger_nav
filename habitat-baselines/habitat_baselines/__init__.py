#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.base_trainer import BaseRLTrainer, BaseTrainer
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.rl.ver.ver_trainer import VERTrainer
from habitat_baselines.rl.ppo.orca_trainer import ORCANoTrainer
from habitat_baselines.rl.ppo.falcon_trainer import FalconTrainer
from habitat_baselines.version import VERSION as __version__  # noqa: F401
import sys
sys.path.append("/home/mobile/ranger_nav/habitat-baselines/")
sys.path.append("/home/mobile/ranger_nav/habitat-lab/")
sys.path.append("/home/mobile/ranger_nav")

# Some local checkouts omit habitat_baselines/il/data assets. Keep RL imports usable.
try:
    from habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer import (
        EQACNNPretrainTrainer,
    )
    from habitat_baselines.il.trainers.pacman_trainer import PACMANTrainer
    from habitat_baselines.il.trainers.vqa_trainer import VQATrainer
except ModuleNotFoundError:
    EQACNNPretrainTrainer = None
    PACMANTrainer = None
    VQATrainer = None

__all__ = [
    "BaseTrainer",
    "BaseRLTrainer",
    "BaseILTrainer",
    "PPOTrainer",
    "FalconTrainer",
    "RolloutStorage",
    "VERTrainer",
    "ORCANoTrainer",
]

if EQACNNPretrainTrainer is not None:
    __all__.extend(
        [
            "EQACNNPretrainTrainer",
            "PACMANTrainer",
            "VQATrainer",
        ]
    )
