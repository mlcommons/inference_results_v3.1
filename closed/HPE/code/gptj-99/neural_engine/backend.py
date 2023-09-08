import time
import os
import sys
import logging
from pathlib import Path 
from ctypes import *

import torch
import transformers
import numpy as np

from typing import Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BACKEND")

class Backend(object):

    def __init__(self, library_path="./build/lib/libGptjPyBind.so", model_path="./ne-q4_0.bin", proc_idx=0):
        lib = cdll.LoadLibrary(library_path)
        init_gptj = lib.init_gptj
        init_gptj.argtypes = [c_int, c_int, c_int, c_int, c_float, c_float, c_float, c_bool, c_int, c_char_p, c_bool, c_int, c_int]
        init_gptj.restype = c_void_p
        eval_gptj_ids = lib.eval_gptj_ids
        eval_gptj_ids.argtypes = [c_void_p, np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'), c_int, c_int, c_int, c_float, c_float, c_int]
        eval_gptj_ids.restype = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
        exit_gptj = lib.exit_gptj
        exit_gptj.argtypes = [c_void_p]
        exit_gptj.restype = None
        self.init_gptj = init_gptj
        self.eval_gptj = eval_gptj_ids
        self.exit_gptj = exit_gptj

        self.model_path = model_path
        self.proc_idx = proc_idx

    def loadModel(self):
        log.info("\nmodel path: " + self.model_path + "\n")
        self.gptj_in_all = self.init_gptj(1234, 128, 2048, 40, 1.0, 0.8, 1.5, False, 2048, (self.model_path + str(self.proc_idx)).encode('utf-8'), True, 4, 1)

    def predict(self, input_batch, attention_mask=None):
        """ Runs inference on 'input_batch' """
        input_batch_array = input_batch.view(-1).numpy().astype('int32')
        output_batch_array = self.eval_gptj(self.gptj_in_all, input_batch_array, input_batch_array.size, 128, 40, 1.0, 0.8, 2048)
        output_batch_array_ptr = cast(output_batch_array, POINTER(c_int))
        output_batch_array = np.ctypeslib.as_array(output_batch_array_ptr, shape=(128,))
        output_token_num = output_batch_array[0]
        output_batch_array = output_batch_array[1:output_token_num+1]
        return torch.tensor(output_batch_array)



