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

# TODO: This harness might be unified with GPTJ harness


class GPT175Harness(BaseBenchmarkHarness):
    """GPT175 harness."""

    def __init__(self, args, benchmark):
        super().__init__(args, benchmark)
        custom_args = [
            "gpu_inference_streams",
            "gpu_copy_streams",
            "devices",
            "tensor_parallelism",
            # TODO add soft drop and cuda graph for GPTJ
            # "soft_drop",
            # "graphs_max_seqlen",
            # "graph_specs"
        ]
        assert self.tensor_parallelism == 8, "GPT175 must be run on a single-node-right-GPU system!"
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + custom_args

    def _get_harness_executable(self):
        """Return path to GPT harness binary."""
        return "./build/bin/harness_gpt"

    # TODO changethe engine path when builder is finished
    def _get_engine_fpath(self, device_type, batch_size, rank):
        return "/tmp/{:}-{:}-{:}-b{:}-{:}-rank{:}.{:}.plan".format(self.name, self.scenario.valstr(),
                                                                   device_type, batch_size, self.precision, rank, self.config_ver)

    def _build_custom_flags(self, flag_dict):
        # eviction last override
        s = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name
        return s

    def enumerate_engines(self):
        gpu_engine_list = []
        for rank in range(self.tensor_parallelism):
            engine_path = self._get_engine_fpath("gpu", self.args["gpu_batch_size"], rank)
            self.check_file_exists(engine_path)
            gpu_engine_list.append(engine_path)
        self.gpu_engine = ','.join(gpu_engine_list)
