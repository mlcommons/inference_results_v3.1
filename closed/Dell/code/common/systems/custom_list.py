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

custom_systems['R750xa_A100_PCIe_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.79472, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_PCIe_80GB.value: 4}), numa_conf=None, system_id="R750xa_A100_PCIe_80GBx4")
custom_systems['R750xa_A100_PCIE_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.79472, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.A100_PCIe_80GB.value: 4}), numa_conf=None, system_id="R750xa_A100_PCIE_80GBx4")
custom_systems['R750xa_H100_PCIe_80GBx2'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.7918440000001, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA H100 PCIe", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=79.6474609375, byte_suffix=ByteSuffix.GiB), max_power_limit=310.0, pci_id="0x233110DE", compute_sm=90): 2}), numa_conf=None, system_id="R750xa_H100_PCIe_80GBx2")
custom_systems['R750xa_H100_PCIe_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.7918440000001, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA H100 PCIe", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=79.6474609375, byte_suffix=ByteSuffix.GiB), max_power_limit=310.0, pci_id="0x233110DE", compute_sm=90): 4}), numa_conf=None, system_id="R750xa_H100_PCIe_80GBx4")
custom_systems['R760xa_H100_PCIe_80GBx2'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Silver 4416+", architecture=CPUArchitecture.x86_64, core_count=20, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.84958, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_PCIe_80GB.value: 2}), numa_conf=None, system_id="R760xa_H100_PCIe_80GBx2")
custom_systems['R760xa_H100_PCIe_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8480+", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056325428, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_PCIe_80GB.value: 4}), numa_conf=None, system_id="R760xa_H100_PCIe_80GBx4")
custom_systems['R760xa_L40x4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8460Y+", architecture=CPUArchitecture.x86_64, core_count=40, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=527.8482319999999, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L40.value: 4}), numa_conf=None, system_id="R760xa_L40x4")
custom_systems['XE8640_H100_SXM_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8468", architecture=CPUArchitecture.x86_64, core_count=48, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056298628, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_SXM_80GB.value: 4}), numa_conf=None, system_id="XE8640_H100_SXM_80GBx4")
custom_systems['XE9640_H100_SXM_80GBx4'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8480+", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056316, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_SXM_80GB.value: 4}), numa_conf=None, system_id="XE9640_H100_SXM_80GBx4")
custom_systems['XE9680_A100_SXM_80GBx8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8480+", architecture=CPUArchitecture.x86_64, core_count=56, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=2.1132477560000003, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA A100-SXM4-80GB", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=80.0, byte_suffix=ByteSuffix.GiB), max_power_limit=500.0, pci_id="0x20B210DE", compute_sm=80): 8}), numa_conf=None, system_id="XE9680_A100_SXM_80GBx8")
custom_systems['XE9680_H100_SXM_80GBx8'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) Platinum 8470", architecture=CPUArchitecture.x86_64, core_count=52, threads_per_core=2): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=1.056294732, byte_suffix=ByteSuffix.TB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.H100_SXM_80GB.value: 8}), numa_conf=None, system_id="XE9680_H100_SXM_80GBx8")
custom_systems['XR4520c_L4x1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Intel(R) Xeon(R) D-2776NT CPU @ 2.10GHz", architecture=CPUArchitecture.x86_64, core_count=16, threads_per_core=1): 1}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=131.50882000000001, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L4.value: 1}), numa_conf=None, system_id="XR4520c_L4x1")
custom_systems['XR5610_L4x1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Genuine Intel(R) CPU 0000%@", architecture=CPUArchitecture.x86_64, core_count=20, threads_per_core=2): 1}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=263.64824, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={GPU(name="NVIDIA L4", accelerator_type=AcceleratorType.Discrete, vram=Memory(quantity=22.494140625, byte_suffix=ByteSuffix.GiB), max_power_limit=75.0, pci_id="0x27B810DE", compute_sm=89): 1}), numa_conf=None, system_id="XR5610_L4x1")
custom_systems['XR7620_L4x1'] = SystemConfiguration(host_cpu_conf=CPUConfiguration(layout={CPU(name="Genuine Intel(R) CPU 0000%@", architecture=CPUArchitecture.x86_64, core_count=32, threads_per_core=1): 2}), host_mem_conf=MemoryConfiguration(host_memory_capacity=Memory(quantity=461.815824, byte_suffix=ByteSuffix.GB), comparison_tolerance=0.05), accelerator_conf=AcceleratorConfiguration(layout={KnownGPU.L4.value: 1}), numa_conf=None, system_id="XR7620_L4x1")

###############################
#### END OF CUSTOM SYSTEMS ####
###############################

