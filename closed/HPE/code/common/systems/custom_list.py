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

custom_systems['HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8480+", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056598868, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_PCIe_80GB.value: 4}), numa_conf=None, system_id="HPE_ProLiant_DL380a_Gen11_H100_PCIe_80GBx4")

custom_systems['HPE_ProLiant_DL320_Gen11_L4x4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 5412U", architecture=CPUArchitecture.x86_64, core_count=24, threads_per_core=2): 1}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.68625199999997, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA Graphics Device", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=22.494140625, byte_suffix=ByteSuffix.GiB), max_power_limit=75.0, pci_id="0x27B810DE", compute_sm=89): 4}), numa_conf=None, system_id="HPE_ProLiant_DL320_Gen11_L4x4")

custom_systems['HPE_ProLiant_XL675d_A100_SXM_80GBx8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="AMD EPYC 7763 64-Core Processor", architecture=CPUArchitecture.x86_64, core_count=64, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056016688, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A100-SXM-80GB", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=80.0, byte_suffix=ByteSuffix.GiB), max_power_limit=400.0, pci_id="0x20B210DE", compute_sm=80): 4}), numa_conf=None, system_id="HPE_ProLiant_XL675d_A100_SXM_80GBx8")


###############################
#### END OF CUSTOM SYSTEMS ####
###############################
