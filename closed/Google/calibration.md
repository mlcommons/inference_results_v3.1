# MLPerf Inference Calibration and Quantization Details

## Quantization steps

All Post training quantization methods we’ve applied for this submission contain these quantization steps. All steps below starts from the pre-trained FP32 model:

*   Calibration step: Run forward pass with the calibration dataset and collect some statistics to quantize the model.
*   Materialization: Quantize model weights and rewrite the model graph to quantize activations be quantized using the statistics collected from the calibration steps.
*   Inference: Run the quantized model for the inference with the updated model graph and weights.


## Calibration and Quantization Details

We use the following formulas for quantization and dequantization: Each quantized value has a “scale” and “zero\_point”. Original float values are quantized linearly (aka. uniform quantization.) For quantization, `Q = round(F / scale + zero_point)`. For dequantization, `F = (Q - zero_point) * scale`. Note that scale and zero\_point are floating values.


### Weight Quantization

We collect weights min/max value during the calibration step. This weight statistic doesn’t use any information from the calibration dataset. It can be calculated from pre-trained float weights. By default we collect w\_min/w\_max for each channel of the weights for channel-wise quantization.

During the materialization step, we compute scale and zero\_point using the w\_min/w\_max range from the calibration dataset. We use 8-bits quantization which Q\_MIN = -128 and Q\_MAX = 127. Here, we only use -127 ~ 127 for symmetric quantization for the weights. The scale is computed using the w\_min and w\_max values. The zero\_point is always zero due to the symmetry. We also quantize the weights using a computed scale, as the quantization scheme above cast the type as int8.


### Weight Quantization with Calibration Dataset

We follow the [GPTQ] (https://arxiv.org/abs/2210.17323) paper scheme to use the calibration data and apply PTQ to the weights.

Extending the quantization algorithm above, we also use activation values to adjust the quantized value during the calibration step. This scheme does not require training data and extends weight quantization by considering a layer's activation values. The difference is the method to update the quantized weights: We compute the hessian matrix of each row (H = 2XX^T) of weights using activation value during forward pass, and use that hessian matrix to adjust quantized value. The scale and zero\_point is same as weight only quantization. 


### Activation Quantization with Calibration Dataset


#### Dynamic Range

Weights are quantized using weights only quantization algorithms. The Calibration step and Materialization step here is same as weight quantization. 

For activations, during the Inference step, we quantize the input of einsum/matmul op by compute scale and zero\_point during the inference forward pass for each inference. And quantize the input value run-time. Here, we use the tensor-wise / asymmetric quantization.

To compute scale, we compute min/max value from the activation, and adjust the range to include the 0 value.

For target op, each input and weights are quantized values. We call the native int op instead of float op. After we get the result from the int op, we dequantize the output using scale/zero\_point for the input and weights. The result would be the same as we put dequantized input and dequantized weight to float op.

Except for matmul(input\_q, weight\_q), all other parts are pre-computed during materialization steps.


#### Static Range

On top of dynamic range quantization, We compute the scale/zero\_point for each input of the target op using calibration dataset. During the inference, we quantize the input of the target op using the pre-computed scale/zero\_point from the calibration steps.
