# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


from __future__ import annotations

import os

from code.common.constants import *
from code.common.systems.base import *
from code.common.systems.accelerator import AcceleratorConfiguration, GPU, MIG
from code.common.systems.cpu import CPUConfiguration, CPU
from code.common.systems.memory import MemoryConfiguration
from code.common.systems.systems import SystemConfiguration
from code.common.systems.known_hardware import *


custom_systems = dict()


# Do not manually edit any lines below this. All such lines are generated via scripts/add_custom_system.py

###############################
### START OF CUSTOM SYSTEMS ###
###############################

custom_systems['AS_8125GS_TNHR_H100_SXM_80GBx8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 9554 64-Core Processor", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.584787192, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_SXM_80GB.value: 8}), numa_conf=None, system_id="AS_8125GS_TNHR_H100_SXM_80GBx8")
custom_systems['SYS_821GE_TNHR_H100_SXM_80GBx8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8468", architecture=CPUArchitecture.x86_64, core_count=48, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.0564650160000002, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_SXM_80GB.value: 8}), numa_conf=None, system_id="SYS_821GE_TNHR_H100_SXM_80GBx8")
custom_systems['SYS_421GU_TNXR'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8470Q", architecture=CPUArchitecture.x86_64, core_count=52, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056497272, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_SXM_80GB.value: 4}), numa_conf=None, system_id="SYS_421GU_TNXR")
custom_systems['SYS-521GE-TNRT_8_H100-PCIe-80GB'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8462Y+", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056497272, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_PCIe_80GB.value: 8}), numa_conf=None, system_id="SYS-521GE-TNRT_8_H100-PCIe-80GB")


###############################
#### END OF CUSTOM SYSTEMS ####
###############################
