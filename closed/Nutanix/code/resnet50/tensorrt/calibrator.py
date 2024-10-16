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

import tensorrt as trt
import os
from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoprimaryctx

IMAGE_C = 3
IMAGE_H = 224
IMAGE_W = 224


class RN50Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator for the RN50 benchmark."""

    def __init__(self, calib_batch_size=1, calib_max_batches=500, force_calibration=False,
                 cache_file="code/resnet50/tensorrt/calibrator.cache",
                 image_dir="build/preprocessed_data/imagenet/ResNet50/fp32", calib_data_map="data_maps/imagenet/cal_map.txt"):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.calib_batch_size = calib_batch_size
        self.calib_max_batches = calib_max_batches
        self.force_calibration = force_calibration
        self.current_idx = 0
        self.cache_file = cache_file

        # Read image files to use for calibration
        image_list = []
        with open(calib_data_map) as f:
            for line in f:
                image_list.append(line.split()[0])

        self.batches = np.stack([np.load(os.path.join(image_dir, file_name + ".npy")) for file_name in image_list])

        self.device_input = cuda.mem_alloc(self.calib_batch_size * IMAGE_C * IMAGE_H * IMAGE_W * 4)

        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if not self.force_calibration and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache = f.read()
        else:
            self.cache = None

    def get_batch_size(self):
        return self.calib_batch_size

    def get_batch(self, names):
        """
        Acquire a single batch 

        Arguments:
        names (string): names of the engine bindings from TensorRT. Useful to understand the order of inputs.
        """
        if self.current_idx < self.calib_max_batches:
            cuda.memcpy_htod(self.device_input, np.ascontiguousarray(self.batches[self.current_idx: self.current_idx + self.calib_batch_size]))
            self.current_idx += 1
            return [int(self.device_input)]
        else:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        return self.cache

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def clear_cache(self):
        self.cache = None

    def __del__(self):
        self.device_input.free()
