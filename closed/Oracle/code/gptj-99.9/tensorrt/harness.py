# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from code.common import logging, args_to_string
from code.common.harness import BaseBenchmarkHarness
import code.common.arguments as common_args


class GPTJHarness(BaseBenchmarkHarness):
    """GPTJ harness."""

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)
        custom_args = [
            "gpu_inference_streams",
            "gpu_copy_streams",
            "devices",
            "tensor_parallelism",
            "enable_sort",
            # TODO add soft drop and cuda graph for GPTJ
            # "soft_drop",
            # "graphs_max_seqlen",
            # "graph_specs"
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + custom_args

    def _get_harness_executable(self):
        """Return path to GPT harness binary."""
        return "./build/bin/harness_gpt"

    def _build_custom_flags(self, flag_dict):
        # eviction last override
        s = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name
        return s
