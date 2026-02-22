#!/usr/bin/env python

from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.value.configuration import SiglipGemmaValueConfig, ValueModelConfig


def test_value_model_config_from_dict():
    payload = {
        "type": "siglip_gemma_value",
        "num_bins": 101,
        "bin_min": -1.0,
        "bin_max": 0.0,
        "state_feature": "observation.state",
        "task_index_feature": "task_index",
        "task_field": "task",
        "camera_features": ["observation.images.front"],
        "language_repo_id": "google/gemma-3-270m",
        "vision_repo_id": "google/siglip-so400m-patch14-384",
        "dropout": 0.2,
    }
    cfg = ValueModelConfig.from_dict(payload)
    assert isinstance(cfg, SiglipGemmaValueConfig)
    assert cfg.type == "siglip_gemma_value"
    assert cfg.num_bins == 101
    assert cfg.camera_features == ["observation.images.front"]


def test_value_model_preset_uses_cosine_decay_with_warmup():
    cfg = SiglipGemmaValueConfig()
    scheduler_cfg = cfg.get_scheduler_preset()
    assert isinstance(scheduler_cfg, CosineDecayWithWarmupSchedulerConfig)
    assert scheduler_cfg.peak_lr == cfg.optimizer_lr
    assert scheduler_cfg.decay_lr == cfg.scheduler_decay_lr
    assert scheduler_cfg.num_warmup_steps == cfg.scheduler_warmup_steps
    assert scheduler_cfg.num_decay_steps == cfg.scheduler_decay_steps
