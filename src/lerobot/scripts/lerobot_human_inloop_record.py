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

"""
Records with policy execution and teleop-device action mirroring enabled.

Phase A/B goals:
- Same policy action is executed on the follower robot and mirrored to the teleop arm.
- Keyboard-toggled intervention lets teleop temporarily take over execution.
"""

import logging

from lerobot.configs import parser
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.utils.import_utils import register_third_party_plugins


@parser.wrap()
def human_inloop_record(cfg: RecordConfig):
    if cfg.teleop is None:
        raise ValueError("`lerobot-human-inloop-record` requires `teleop` config.")
    if cfg.policy is None:
        raise ValueError("`lerobot-human-inloop-record` requires `policy` config.")

    cfg.policy_sync_to_teleop = True
    cfg.intervention_state_machine_enabled = True
    logging.info(
        "Intervention state machine enabled. Press '%s' to toggle takeover. "
        "Recorded `action` is the executed action. "
        "Policy output is stored in `complementary_info.policy_action`.",
        cfg.intervention_toggle_key,
    )
    return record(cfg)


def main():
    register_third_party_plugins()
    human_inloop_record()


if __name__ == "__main__":
    main()
