#!/usr/bin/env python3
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


__doc__ = """Scripts that take a retinanet engine and openImage input, infer the output and test the accuracy
"""

import argparse
import json
import os
import sys
import glob
import random
import time
from PIL import Image
from importlib import import_module
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch  # Retinanet model source requires GPU installation of PyTorch 1.10
from torchvision.transforms import functional as F
import onnx
import tensorrt as trt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from code.common import logging
from code.common.constants import TRT_LOGGER, Scenario
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.runner import EngineRunner, get_input_format
from code.common.systems.system_list import SystemClassifications
from code.plugin import load_trt_plugin_by_network

G_GPTJ6B_MAX_SEQLEN = 2048
G_CNNDAILYMAIL_CALSET_PATH = None
G_CNNDAILYMAIL_CALMAP_PATH = None
G_CNNDAILYMAIL_VALSET_PATH = None
G_CNNDAILYMAIL_VALMAP_PATH = None
G_CNNDAILYMAIL_ANNO_PATH = None
G_CNNDAILYMAIL_PREPROCESSED_INT8_PATH = None
G_CNNDAILYMAIL_CALIBRATION_CACHE_PATH = None

# Debug flags
G_DEBUG_LAYER_ON = False


class TRTTester:
    """
    Wrapper class to encapsulate the TRT tester util functions.
    """

    def __init__(self, engine_files, batch_size, precision,
                 num_opt_profiles, use_dla=False,
                 skip_engine_build=False, verbose=False,
                 ):
        """
        Test GPT model through the TRT path.
        """
        self.batch_size = batch_size
        self.verbose = verbose
        self.engine_file = engine_file
        self.cache_file = G_CNNDAILYMAIL_CALIBRATION_CACHE_PATH
        self.precision = precision
        self.num_opt_profiles = num_opt_profiles

        # TensorRT engine related fields
        if use_dla:
            self.dla_core = 0
        else:
            self.dla_core = None

        # Initiate the plugin and logger
        self.logger = TRT_LOGGER  # Use the global singleton, which is required by TRT.
        self.logger.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO

        # load_trt_plugin_by_network("gptj-6b")
        trt.init_libnvinfer_plugins(self.logger, "")

        if not skip_engine_build:
            # TODO: Build the engine
        else:
            if not os.path.exists(engine_file):
                raise RuntimeError(f"Cannot find engine file {engine_file}. Please supply the onnx file or engine file.")

        self.runner = EngineRunner(self.engine_file, verbose=verbose)

        # TODO: read CNN dailymail related field

    def apply_flag(self, flag):
        """Apply a TRT builder flag."""
        self.builder_config.flags = (self.builder_config.flags) | (1 << int(flag))

    def clear_flag(self, flag):
        """Clear a TRT builder flag."""
        self.builder_config.flags = (self.builder_config.flags) & ~(1 << int(flag))

    # Helper function to build a TRT engine
    # TODO: This function will likely call Tekit API to build the engine.
    def create_trt_engine(self):
        return

    def run_rouge(self, num_samples=8):
        return 0


class PytorchTester:
    """
    The reference implementation of the retinanet from the mlcommon-training repo, from:
    https://github.com/mlcommons/training/tree/master/single_stage_detector/ssd/model

    To run this tester, you would need to clone the repo, and mount it to the container.
    """

    def __init__(self, pyt_ckpt_path, batch_size=8, output_file="build/retinanet_pytorch.out"):
        # TODO: Will need to load the torch GPT model here
        self.device = torch.device("cuda:0")
        self.batch_size = batch_size
        self.output_file = output_file

    def run_rouge(self, num_samples=8):
        """
        """
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine_file",
                        help="Specify where the GPTJ6B engine file is",
                        required=False)
    parser.add_argument("--pyt_ckpt_path",
                        help="Specify where the PyTorch checkpoint path is",
                        default="build/models/GPTJ-6B/04252023-GPTJ6B-ckpt")
    parser.add_argument("--batch_size",
                        help="batch size",
                        type=int,
                        default=8)
    parser.add_argument("--num_samples",
                        help="Number of samples to run. We have 114514 in total for cnn-dailymail",
                        type=int,
                        default=114514)
    parser.add_argument("--num_opt_profiles",
                        help="Number of TRT optimization profiles to create",
                        type=int,
                        default=1)
    parser.add_argument("--trt_precision",
                        help="Run TensorRT in the specified precision",
                        choices=("fp32", "fp16", "int8", "fp8"),
                        default="fp32")
    parser.add_argument("--use_dla",
                        help="Use DLA instead of gpu",
                        action="store_true")
    parser.add_argument("--skip_engine_build",
                        help="Skip the TRT engine build phase if possible.",
                        action="store_true")
    parser.add_argument("--pytorch",
                        help="whether to run pytorch inference",
                        action="store_true")
    parser.add_argument("--verbose",
                        help="verbose output",
                        action="store_true")
    args = parser.parse_args()

    # Pytorch Tester
    if args.pytorch:
        logging.info(f"Running Accuracy test for Pytorch reference implementation.")
        if not os.path.exists(args.pyt_ckpt_path):
            raise RuntimeError(f"Cannot access {args.pyt_ckpt_path}. Please download the model or mount the scratch path.")
        pt_tester = PytorchTester(args.pyt_ckpt_path, args.batch_size)
        rouge = pt_tester.run_rouge(args.num_samples)
        logging.info(f"Pytorch ROUGE Score: {rouge}, Reference: ???.  % of ref: {0}")
    else:
        # TRT Tester
        logging.info(f"Running accuracy test for GPTJ6B using {args.engine_file} ...")
        tester = TRTTester(args.engine_file, args.batch_size, args.trt_precision, args.num_opt_profiles,
                           args.use_dla, args.skip_engine_build, args.verbose)
        rouge = tester.run_rouge(args.num_samples)
        logging.info(f"TRT ROUGE Score: {rouge}, Reference: ???.  % of ref: {0}")

    # To run the TRT tester:
    # python3 -m code.gptj6b.tensorrt.infer --engine_file /tmp/gptj6b.b8.int8.engine --num_samples=1200 --batch_size=8
    # To run the pytorch tester:
    # python3 -m code.gptj6b.tensorrt.infer --pytorch --num_samples=1200 --batch_size=8


if __name__ == "__main__":
    main()
