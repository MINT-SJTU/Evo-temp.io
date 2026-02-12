#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lerobot.configs import parser
from lerobot.configs.value_train import ValueTrainPipelineConfig
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.value.pipeline import run_value_training_pipeline


@parser.wrap()
def value_train(cfg: ValueTrainPipelineConfig):
    return run_value_training_pipeline(cfg)


def main():
    register_third_party_plugins()
    value_train()


if __name__ == "__main__":
    main()
