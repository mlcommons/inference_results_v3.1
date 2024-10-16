#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to preprocess data for Resnet50 benchmark."""

import argparse
import math
import os
import shutil

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
    import cv2

from code.common import logging
from code.common.image_preprocessor import ImagePreprocessor, center_crop, resize_with_aspectratio


def preprocess_imagenet_for_resnet50(data_dir, preprocessed_data_dir, formats, overwrite=False, cal_only=False, val_only=False):
    """Proprocess the raw images for inference."""

    def loader(file):
        """Resize and crop image to required dims and return as FP32 array."""

        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w, h = (224, 224)
        image = resize_with_aspectratio(image, h, w)
        image = center_crop(image, h, w)
        image = np.asarray(image, dtype='float32')
        # Normalize image.
        means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
        image -= means
        # Transpose.
        image = image.transpose([2, 0, 1])
        return image

    def quantizer(image):
        """Return quantized INT8 image of input FP32 image."""
        return np.clip(image, -128.0, 127.0).astype(dtype=np.int8, order='C')

    preprocessor = ImagePreprocessor(loader, quantizer)
    if not val_only:
        # Preprocess calibration set. FP32 only because calibrator always takes FP32 input.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "ResNet50"),
                         "data_maps/imagenet/cal_map.txt", ["fp32"], overwrite)
    if not cal_only:
        # Preprocess validation set.
        preprocessor.run(os.path.join(data_dir, "imagenet"), os.path.join(preprocessed_data_dir, "imagenet", "ResNet50"),
                         "data_maps/imagenet/val_map.txt", formats, overwrite)


def main():
    """
    Parse arguments to identify the data directory with the input images
      and the output directory for the preprocessed images.
    The data directory is assumed to have the following structure:
    <data_dir>
     └── imagenet
    And the output directory will have the following structure:
    <preprocessed_data_dir>
     └── imagenet
         └── ResNet50
             ├── fp32
             └── int8_linear
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-d",
        help="Directory containing the input images.",
        default="build/data"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data"
    )
    parser.add_argument(
        "--formats", "-t",
        help="Comma-separated list of formats. Choices: fp32, int8_linear, int8_chw4.",
        default="default"
    )
    parser.add_argument(
        "--overwrite", "-f",
        help="Overwrite existing files.",
        action="store_true"
    )
    parser.add_argument(
        "--cal_only",
        help="Only preprocess calibration set.",
        action="store_true"
    )
    parser.add_argument(
        "--val_only",
        help="Only preprocess validation set.",
        action="store_true"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    preprocessed_data_dir = args.preprocessed_data_dir
    formats = args.formats.split(",")
    overwrite = args.overwrite
    cal_only = args.cal_only
    val_only = args.val_only
    default_formats = ["int8_linear"]

    # Now, actually preprocess the input images
    logging.info("Loading and preprocessing images. This might take a while...")
    if args.formats == "default":
        formats = default_formats
    preprocess_imagenet_for_resnet50(data_dir, preprocessed_data_dir, formats, overwrite, cal_only, val_only)

    logging.info("Preprocessing done.")


if __name__ == '__main__':
    main()
