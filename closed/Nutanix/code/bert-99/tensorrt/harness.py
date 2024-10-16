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

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import pycuda
    import pycuda.autoprimaryctx

from code.common import logging, dict_get, run_command, args_to_string
from code.common.harness import BaseBenchmarkHarness
import code.common.arguments as common_args


class BertHarness(BaseBenchmarkHarness):
    """BERT harness."""

    def __init__(self, args, benchmark):
        # TODO file check will not work for current way to pass multiple engines.
        #super().__init__(args, name)
        self.enable_interleaved = False
        self.is_int8 = args['precision'] == 'int8'
        if self.is_int8 and 'enable_interleaved' in args:
            self.enable_interleaved = args['enable_interleaved']
        args["skip_file_checks"] = True
        super(BertHarness, self).__init__(args, benchmark)
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS +\
            ["gpu_inference_streams", "gpu_copy_streams", "gpu_batch_size", "graphs_max_seqlen", "soft_drop", "devices", "graph_specs"]

    def _get_harness_executable(self):
        """Return path to BERT harness binary."""
        return "./build/bin/harness_bert"

    def _build_custom_flags(self, flag_dict):
        # eviction last override
        s = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name
        if self.system_id == 'L4x1':
            s += " --eviction_last=0.5"
        return s

    def _get_engine_fpath(self, device_type, batch_size):
        """Return file path of engine."""
        num_profiles = 1
        if 'gpu_inference_streams' in self.args:
            # use gpu_inference_streams to determine the number of duplicated profiles
            # in the engine when not using lwis mode
            num_profiles = self.args['gpu_inference_streams']

        seq_len = 384  # default sequence length
        engine_name = "{:}/{:}-{:}-{:}-{:}_S_{:}_B_{:}_P_{:}_vs{:}.{:}.plan".format(
            self.engine_dir, self.name, self.scenario.valstr(),
            device_type, self.precision, seq_len, batch_size, num_profiles, '_il' if self.enable_interleaved else '', self.config_ver)

        return engine_name
