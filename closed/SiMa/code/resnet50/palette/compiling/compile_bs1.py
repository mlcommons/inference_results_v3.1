import argparse
import os
import numpy as np
import logging
import dataclasses

from afe.apis.defines import default_quantization, CalibrationMethod
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.defines import InputName
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import onnx_source, keras_source, tflite_source, pytorch_source
from sima_utils.data.data_generator import DataGenerator
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.backends.mpk.interface import L2CachingMode


"""
Script for quantizing and compiling
"""

models = ["resnet50_v1_opt.onnx",
          ]

sample_start = 10
max_calib_samples = 35

# calib_method = CalibrationMethod.MIN_MAX
calib_method = CalibrationMethod.HISTOGRAM_MSE
# calib_method = CalibrationMethod.MOVING_AVERAGE_MIN_MAX

quant_configs = dataclasses.replace(default_quantization, calibration_method=calib_method)

MODEL_DIR = './'
OUTPUT_DIR = f'./output_bs1_{calib_method.value}_final'


def compile_model(model_name: str, arm_only: bool):

    # Uncomment the following line to enable verbose error messages.
    enable_verbose_error_messages()
    
    print(f"Compiling model {model_name} with batch_size=1 and arm_only={arm_only}")

    # Models importer parameters
    # input shape in format NCHW with N (the batchsize) = 1
    input_name, input_shape, input_type = ("input_tensor:0", (1,3,224,224), ScalarType.float32)

    input_shapes_dict = {input_name: input_shape}
    input_types_dict = {input_name: input_type}

    model_path = os.path.join(MODEL_DIR, f"{model_name}")

    # refer to the SDK User Guide for the specific format 
    importer_params = onnx_source(model_path, input_shapes_dict, input_types_dict)
   
    model_prefix = os.path.splitext(model_path)[0]
    output_dir = os.path.join(OUTPUT_DIR, f"{model_prefix}")
    os.makedirs(output_dir, exist_ok=True)
    loaded_net = load_model(importer_params)

    # Read images
    cal_data_path = os.path.join(MODEL_DIR, "calibration/mlperf_resnet50_cal_NCHW.dat")
    cal_label_path = os.path.join(MODEL_DIR, "calibration/mlperf_resnet50_cal_labels_int32.dat")
    cal_dat = np.fromfile(cal_data_path, dtype=np.float32).reshape(500, 3, 224, 224)
    cal_labels = np.fromfile(cal_label_path, dtype=np.int32)
    
    # Tranpose images from NCHW to NHWC
    cal_dat_NHWC = cal_dat.transpose(0, 2, 3, 1)

    dg = DataGenerator({input_name: cal_dat_NHWC[sample_start:sample_start + max_calib_samples]})
    calibration_data = convert_data_generator_to_iterable(dg)

    model_sdk_net = loaded_net.quantize(calibration_data,
                                        quant_configs,
                                        model_name=model_prefix,
                                        arm_only=arm_only)

    saved_model_directory = "sdk_bs1"
    model_sdk_net.save(model_name=model_name, output_directory=saved_model_directory)

    model_sdk_net.compile(output_path=output_dir,
                          batch_size=1,
                          log_level=logging.INFO,
                          l2_caching_mode=L2CachingMode.SINGLE_MODEL,
                          tessellate_parameters={"MLA_0/placeholder_0": [[], [224], [3]]})

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for model in models:
        compile_model(model, False)

if __name__ == "__main__":
    main()
