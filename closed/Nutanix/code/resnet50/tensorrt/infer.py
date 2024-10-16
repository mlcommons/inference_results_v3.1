#! /usr/bin/env python3
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
import platform

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import tensorrt as trt

import code.common.arguments as common_args
from code.common.runner import EngineRunner, get_input_format
from code.common.accuracy import ImageNetAccuracyRunner
from code.common import logging


def run_ResNet50_accuracy(engine_file, batch_size, num_images, verbose=False):
    """
    A local runner to test Engine accuracy on a random subset of validation images.
    This will run accuracy tests *WITHOUT* Loadgen.
    """
    if verbose:
        logging.info("Running ResNet50 accuracy test with:")
        logging.info("    engine_file: {:}".format(engine_file))
        logging.info("    batch_size: {:}".format(batch_size))
        logging.info("    num_images: {:}".format(num_images))

    runner = EngineRunner(engine_file, verbose=verbose)
    input_dtype, input_format = get_input_format(runner.engine)
    if input_dtype == trt.DataType.FLOAT:
        format_string = "fp32"
    elif input_dtype == trt.DataType.INT8:
        if input_format == trt.TensorFormat.LINEAR:
            format_string = "int8_linear"
        elif input_format in [trt.TensorFormat.CHW4, trt.TensorFormat.DLA_HWC4]:
            format_string = "int8_chw4"
    image_dir = os.path.join(os.getenv("PREPROCESSED_DATA_DIR", "build/preprocessed_data"),
                             "imagenet/ResNet50", format_string)

    accuracy_runner = ImageNetAccuracyRunner(runner, batch_size, image_dir, num_images,
                                             verbose=verbose)
    return accuracy_runner.run()


def main():
    args = common_args.parse_args(common_args.ACCURACY_ARGS)
    logging.info("Running accuracy test...")
    acc = run_ResNet50_accuracy(args["engine_file"], args["batch_size"], args["num_samples"],
                                verbose=args["verbose"])

    logging.info("Accuracy: {:}".format(acc))


if __name__ == "__main__":
    main()
