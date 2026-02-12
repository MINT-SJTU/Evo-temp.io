#!/usr/bin/env python

from lerobot.value.configuration import MLPValueConfig, ValueModelConfig


def test_value_model_config_from_dict():
    payload = {
        "type": "mlp",
        "num_bins": 101,
        "bin_min": -1.0,
        "bin_max": 0.0,
        "state_feature": "observation.state",
        "task_index_feature": "task_index",
        "hidden_dims": [128, 64],
        "task_embedding_dim": 16,
        "dropout": 0.2,
    }
    cfg = ValueModelConfig.from_dict(payload)
    assert isinstance(cfg, MLPValueConfig)
    assert cfg.type == "mlp"
    assert cfg.num_bins == 101
    assert cfg.hidden_dims == [128, 64]
