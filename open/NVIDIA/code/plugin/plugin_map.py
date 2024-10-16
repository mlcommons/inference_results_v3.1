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

import ctypes

from code.common.constants import *
from code.common.systems.system_list import SystemClassifications
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Callable, List


@dataclass
class LoadablePlugin:
    """
    Dataclass which describes a loadable TensorRT plugin, with constraints
    """

    path: str
    """str: Path to the TRT plugin library"""

    constraints: List[Callable[[], bool]] = field(default_factory=lambda: list())
    """List[Callable[[], bool]: list of constraints that describes whether the plugin can be loaded """

    def get_full_path(self):
        return os.path.join("build", "plugins", self.path)

    def load(self, args: dict):
        if self.can_load(args):
            print(f"Loading TensorRT plugin from {self.get_full_path()}")
            ctypes.CDLL(self.get_full_path())

    def can_load(self, args: dict):
        for constraint in self.constraints:
            if not constraint(args):
                return False
        return True


base_plugin_map = {
    Benchmark.BERT: [],
}
