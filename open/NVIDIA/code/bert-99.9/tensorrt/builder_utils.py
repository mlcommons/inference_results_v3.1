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

import numpy as np
import onnx
import logging
import tensorrt as trt
import json
import torch
from typing import List, Dict, Any, Union, Optional
import os


def onnx_to_tf_name(onnx_name):
    """
    Converting variables in the onnx checkpoint to names corresponding to the naming convention used in the TF version, expected by the builder
    """
    onnx_name = onnx_name.lower()
    toks = [t.strip('_') for t in onnx_name.split('.')]
    if toks[0] == 'bert':  # embeddings or encoder
        if toks[1] == 'encoder':  # transformer

            if toks[-2] == 'layernorm':  # bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            elif (toks[-2] == 'dense' or toks[-2] in {'key', 'value', 'query'}) and toks[-1] == 'weight':
                toks[-1] = 'kernel'
            elif (toks[-3] == 'dense' or toks[-3] in {'key', 'value', 'query'}) and toks[-1] == 'amax':
                if toks[-2] == 'weight_quantizer':
                    toks[-2] = 'kernel'
                elif toks[-2] == 'input_quantizer':
                    toks[-2] = 'input'

            if 'final_input_quantizer' not in toks[2]:
                toks = toks[3:]
                toks[0] = 'l{}'.format(int(toks[0]))
        else:
            if toks[-2] == 'layernorm':  # bias->beta, weight->gamma
                toks[-1] = 'beta' if toks[-1] == 'bias' else 'gamma'
            else:  # embeddings: drop "_weight" suffix
                if toks[-1] == 'amax':
                    toks[-2] = 'amax'
                toks = toks[:-1]
    elif 'qa' in onnx_name:
        name = 'cls_squad_output_bias' if toks[-1] == 'bias' else 'cls_squad_output_weights'
        return name
    else:
        print("Encountered unknown case:", onnx_name)
        assert (False)
    parsed = '_'.join(toks)
    return parsed


"""
Attentions Keys
"""
WQ = "self_query_kernel"
BQ = "self_query_bias"
WK = "self_key_kernel"
BK = "self_key_bias"
WV = "self_value_kernel"
BV = "self_value_bias"


def convert_onnx_weight_dict(tensor_dict, use_int8=False, interleaved=False):

    weights_dict = dict()
    for outname, tensor in tensor_dict.items():

        if outname.find("_amax") != -1:
            weights_dict[outname] = tensor
        elif outname.find(BQ) != -1:
            prefix = outname[:outname.find(BQ)]

            Wq = tensor_dict[prefix + WQ]
            Wk = tensor_dict[prefix + WK]
            Wv = tensor_dict[prefix + WV]
            Bq = tensor
            Bk = tensor_dict[prefix + BK]
            Bv = tensor_dict[prefix + BV]

            weights_dict[prefix + WQ] = Wq
            weights_dict[prefix + WK] = Wk
            weights_dict[prefix + WV] = Wv
            weights_dict[prefix + BQ] = Bq
            weights_dict[prefix + BK] = Bk
            weights_dict[prefix + BV] = Bv

        elif outname.find(BK) != -1 or outname.find(BV) != -1 or outname.find(WQ) != -1 or outname.find(WK) != -1 or outname.find(WV) != -1:
            pass
        else:
            flat_tensor = np.ascontiguousarray(tensor).flatten()
            weights_dict[outname] = flat_tensor

            if outname.find("kernel") != -1:
                tensor = np.transpose(tensor)
                weights_dict[outname + "_notrans"] = np.ascontiguousarray(tensor).flatten()

    logging.info(f"Found {len(weights_dict)} entries in weight map")
    return weights_dict


def get_pytorch_fake_quant_weights(path: Union[str, os.PathLike]):
    """
    Load the weights from a pytorch state dict
    """
    state_dict = torch.load(path, map_location='cpu')
    # pytorch and onnx has the same naming convention because the onnx was exported from pytorch
    tensor_dict = {onnx_to_tf_name(name): val.numpy() for name, val in state_dict.items()}
    return convert_onnx_weight_dict(tensor_dict, use_int8=True)


def get_onnx_fake_quant_weights(path):
    """Return weights from ONNX model file."""
    model = onnx.load(path)
    weights = model.graph.initializer
    weights_dict = dict()
    for w in weights:
        tf_name = onnx_to_tf_name(w.name)
        if tf_name is not None:
            weights_dict[tf_name] = np.frombuffer(w.raw_data, np.float32).reshape(w.dims)
    return weights_dict


def mark(network, tensor, dtype):
    """Set input dtype on tensor and mark it as an output tensor."""
    # NOTE: TRT9 may allow setting the DTYPE only when the tensor is marked input/output
    network.mark_output(tensor)
    tensor.dtype = dtype


def add_gelu(network, input_tensor):
    """This will trigger FC+GELU fusion in TRT"""
    shape = (1, ) * len(input_tensor.shape)
    POW = network.add_constant(shape, trt.Weights(np.ascontiguousarray([3.0], dtype=np.float32)))
    MULTIPLY = network.add_constant(shape, trt.Weights(np.ascontiguousarray([0.044715], dtype=np.float32)))
    SQRT = network.add_constant(shape, trt.Weights((np.ascontiguousarray([0.79788456080286535587989211986876], dtype=np.float32))))
    ONE = network.add_constant(shape, trt.Weights((np.ascontiguousarray([1.0], dtype=np.float32))))
    HALF = network.add_constant(shape, trt.Weights((np.ascontiguousarray([0.5], dtype=np.float32))))
    X_pow = network.add_elementwise(input_tensor, POW.get_output(0), trt.ElementWiseOperation.POW)
    X_pow_t = X_pow.get_output(0)
    X_mul = network.add_elementwise(X_pow_t, MULTIPLY.get_output(0), trt.ElementWiseOperation.PROD)
    X_add = network.add_elementwise(input_tensor, X_mul.get_output(0), trt.ElementWiseOperation.SUM)
    X_sqrt = network.add_elementwise(X_add.get_output(0), SQRT.get_output(0), trt.ElementWiseOperation.PROD)
    X_sqrt_tensor = X_sqrt.get_output(0)
    X_tanh = network.add_activation(X_sqrt_tensor, trt.ActivationType.TANH)
    X_tanh_tensor = X_tanh.get_output(0)
    X_one = network.add_elementwise(X_tanh_tensor, ONE.get_output(0), trt.ElementWiseOperation.SUM)
    CDF = network.add_elementwise(X_one.get_output(0), HALF.get_output(0), trt.ElementWiseOperation.PROD)
    gelu_layer = network.add_elementwise(CDF.get_output(0), input_tensor, trt.ElementWiseOperation.PROD)

    # enable elementwise fusing for int8 && fp16
    POW.precision = trt.DataType.FLOAT
    MULTIPLY.precision = trt.DataType.FLOAT
    SQRT.precision = trt.DataType.FLOAT
    ONE.precision = trt.DataType.FLOAT
    HALF.precision = trt.DataType.FLOAT
    X_pow.precision = trt.DataType.FLOAT
    X_mul.precision = trt.DataType.FLOAT
    X_add.precision = trt.DataType.FLOAT
    X_sqrt.precision = trt.DataType.FLOAT
    X_tanh.precision = trt.DataType.FLOAT
    X_one.precision = trt.DataType.FLOAT
    CDF.precision = trt.DataType.FLOAT
    gelu_layer.precision = trt.DataType.FLOAT
    return gelu_layer
