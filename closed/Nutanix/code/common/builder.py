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

import os
from abc import ABC, abstractmethod

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import tensorrt as trt

from code.common import logging, dict_get
from code.common.constants import TRT_LOGGER, Scenario
from code.common.fields import Fields
from code.common.constants import Benchmark


class AbstractBuilder(ABC):
    """Interface base class for calibrating and building engines."""

    @abstractmethod
    def build_engines(self):
        """
        Builds the engine using assigned member variables as parameters.
        """
        pass

    @abstractmethod
    def calibrate(self):
        """
        Performs INT8 calibration using variables as parameters. If INT8 calibration is not supported for the Builder,
        then this method should print a message saying so and return immediately.
        """
        pass


class TensorRTEngineBuilder(AbstractBuilder):
    """
    Base class for calibrating and building engines for a given benchmark. Has the steps common to most benchmarks that
    use TensorRT on top of NVIDIA GPUs.
    """

    def __init__(self, args, benchmark, workspace_size=(1 << 30)):
        """
        Initializes a TensorRTEngineBuilder. The settings for the builder are set on construction, but can be modified
        to be reflected in a built engine as long as the fields are modified before `self.build_engines` is called.

        Args:
            args (Dict[str, Any]):
                Arguments represented by a dictionary. This is expected to be the output (or variation of the output) of
                a BenchmarkConfiguration.as_dict(). This is because the BenchmarkConfiguration should be validated if it
                was registered into the global ConfigRegistry, and therefore must contain mandatory fields for engine
                build-time.
            benchmark (Benchmark):
                An enum member representing the benchmark this EngineBuilder is constructing an engine for.
        """

        self.benchmark = benchmark
        self.name = benchmark.valstr()
        self.args = args

        # Configuration variables
        self.verbose = dict_get(args, "verbose", default=False)
        if self.verbose:
            logging.info("========= TensorRTEngineBuilder Arguments =========")
            for arg in args:
                logging.info(f"{arg}={args[arg]}")

        self.system_id = args["system_id"]
        self.scenario = args["scenario"]
        self.config_ver = args["config_ver"]
        self.engine_dir = f"./build/engines/{self.system_id}/{self.name}/{self.scenario.valstr()}"

        # Set up logger, builder, and network.
        self.logger = TRT_LOGGER  # Use the global singleton, which is required by TRT.
        self.logger.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO
        trt.init_libnvinfer_plugins(self.logger, "")
        self.builder = trt.Builder(self.logger)
        self.builder_config = self.builder.create_builder_config()
        self.builder_config.max_workspace_size = workspace_size
        if dict_get(args, "verbose_nvtx", default=False) or dict_get(args, "verbose", default=False):
            self.builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # Precision variables
        self.input_dtype = dict_get(args, "input_dtype", default="fp32")
        self.input_format = dict_get(args, "input_format", default="linear")
        self.precision = dict_get(args, "precision", default="int8")
        self.clear_flag(trt.BuilderFlag.TF32)
        if self.precision == "fp16":
            self.apply_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            self.apply_flag(trt.BuilderFlag.INT8)

        # Batch size variable
        if self.scenario == Scenario.SingleStream:
            # 3D-UNet can have batchsize > 1 for sliding window while handling one sample at a time
            self.batch_size = self.args.get("batch_size", 1) if self.benchmark in [Benchmark.UNET3D, ] else 1
        elif self.scenario in [Scenario.Server, Scenario.Offline, Scenario.MultiStream]:
            self.batch_size = self.args.get("batch_size", 1)
            self.multi_stream_samples_per_query = self.args.get("multi_stream_samples_per_query",
                                                                Fields.multi_stream_samples_per_query.value.default)
            if self.scenario == Scenario.MultiStream:
                if self.batch_size > self.multi_stream_samples_per_query:
                    raise ValueError(f"MultiStream cannot have batch size greater than "
                                     "number of samples per query: {self.multi_stream_samples_per_query}")
                if self.multi_stream_samples_per_query % self.batch_size != 0:
                    raise ValueError(f"In MultiStream, harness only supports cases where "
                                     "number of samples per query ({self.multi_stream_samples_per_query}) "
                                     "is divisible by batch size ({self.batch_size})")
        else:
            raise ValueError(f"Invalid scenario: {self.scenario}")

        # Device variables
        self.device_type = "gpu"
        self.dla_core = args.get("dla_core", None)
        self.dla_sram = 1 << 20
        # For 3.0 submission, Orin NX does not have DLA 3.10.1, use half sram
        if self.system_id == "Orin_NX":
            self.dla_sram = 1 << 19
        if self.dla_core is not None:
            logging.info(f"Using DLA: Core {self.dla_core}")
            self.device_type = "dla"
            self.apply_flag(trt.BuilderFlag.GPU_FALLBACK)
            self.builder_config.default_device_type = trt.DeviceType.DLA
            self.builder_config.DLA_core = int(self.dla_core)
            # https://nvbugs/3812184
            # TensorRT by default uses 0.5 MB as the DLA SRAM size, which is suboptimal
            # Set the SRAM size to 1 MB manually (maximum avaible for each DLA on Orin) here
            self.builder_config.set_memory_pool_limit(trt.MemoryPoolType.DLA_MANAGED_SRAM, self.dla_sram)

        # Currently, TRT has limitation that we can only create one execution
        # context for each optimization profile. Therefore, create more profiles
        # so that LWIS can create multiple contexts.
        self.num_profiles = self.args.get("gpu_copy_streams", 4)
        self.profiles = []
        self.initialized = False

    def initialize(self):
        """Builds the network in preparation for building the engine. This method must be implemented by
        the subclass.

        The implementation should also set self.initialized to True.
        """
        raise NotImplementedError("TensorRTEngineBuilder.initialize() should build the network")

    def apply_flag(self, flag):
        """Apply a TRT builder flag."""
        self.builder_config.flags = (self.builder_config.flags) | (1 << int(flag))

    def clear_flag(self, flag):
        """Clear a TRT builder flag."""
        self.builder_config.flags = (self.builder_config.flags) & ~(1 << int(flag))

    def _get_engine_fpath(self, device_type, batch_size):
        # Use default if not set
        if device_type is None:
            device_type = self.device_type
        if batch_size is None:
            batch_size = self.batch_size

        # If the name ends with .plan, we assume that it is a custom path / filename
        if self.name.endswith(".plan"):
            return f"{self.engine_dir}/{self.name}"
        else:
            return f"{self.engine_dir}/{self.name}-{self.scenario.valstr()}-{device_type}-b{batch_size}-{self.precision}.{self.config_ver}.plan"

    def build_engines(self, engine_name=None):
        """Calls self.initialize() if it has not been called yet. Builds and saves the engine."""

        if not self.initialized:
            self.initialize()

        # Create output directory if it does not exist.
        if not os.path.exists(self.engine_dir):
            os.makedirs(self.engine_dir)

        if engine_name is None:
            engine_name = self._get_engine_fpath(self.device_type, self.batch_size)
        logging.info(f"Building {engine_name}")

        if self.network.has_implicit_batch_dimension:
            self.builder.max_batch_size = self.batch_size
        else:
            # Create optimization profiles if on GPU
            if self.dla_core is None:
                for i in range(self.num_profiles):
                    # If profile has already been created, reuse the profile
                    if i < len(self.profiles):
                        logging.info(f"Reusing profile: {i}")
                        profile = self.profiles[i]
                    else:
                        profile = self.builder.create_optimization_profile()
                    # Set profile input shapes
                    for input_idx in range(self.network.num_inputs):
                        input_shape = self.network.get_input(input_idx).shape
                        input_name = self.network.get_input(input_idx).name
                        min_shape = trt.Dims(input_shape)
                        min_shape[0] = 1
                        max_shape = trt.Dims(input_shape)
                        max_shape[0] = self.batch_size
                        profile.set_shape(input_name, min_shape, max_shape, max_shape)
                    if not profile:
                        raise RuntimeError("Invalid optimization profile!")
                    # If profile has already been created, reuse the profile
                    if i < len(self.profiles):
                        continue
                    self.builder_config.add_optimization_profile(profile)
                    self.profiles.append(profile)
            else:
                # Use fixed batch size if on DLA
                for input_idx in range(self.network.num_inputs):
                    input_shape = self.network.get_input(input_idx).shape
                    input_shape[0] = self.batch_size
                    self.network.get_input(input_idx).shape = input_shape

        # Build engines
        engine = self.builder.build_engine(self.network, self.builder_config)
        engine_inspector = engine.create_engine_inspector()
        layer_info = engine_inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
        logging.info("========= TensorRT Engine Layer Information =========")
        logging.info(layer_info)

        # [https://nvbugs/3965323] Need to delete the engine inspector to release the refcount
        del engine_inspector

        buf = engine.serialize()
        with open(engine_name, 'wb') as f:
            f.write(buf)

    def calibrate(self):
        """Generate a new calibration cache."""

        self.need_calibration = True
        self.calibrator.clear_cache()
        self.initialize()
        # Generate a dummy engine to generate a new calibration cache.
        if self.network.has_implicit_batch_dimension:
            self.builder.max_batch_size = 1
        else:
            for input_idx in range(self.network.num_inputs):
                input_shape = self.network.get_input(input_idx).shape
                input_shape[0] = 1
                self.network.get_input(input_idx).shape = input_shape
        engine = self.builder.build_engine(self.network, self.builder_config)


class MultiBuilder(AbstractBuilder):
    """
    MultiBuilder allows for building multiple engines sequentially. As an example, RNN-T has multiple components, each of
    which have separate engines, which we would like to abstract away.
    """

    def __init__(self, builders, args):
        """
        MultiBuilder takes in a list of Builder classes and args to be passed to these Builders.
        """
        self.builders = list(builders)
        self.args = args

    def build_engines(self):
        for b in self.builders:
            b(self.args).build_engines()

    def calibrate(self):
        for b in self.builders:
            b(self.args).calibrate()
