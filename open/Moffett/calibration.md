# MLPerf Inference Calibration and Quantization Details - GPTJ

## Moffett MLPerf Quantization
Post-training quantization and static range of each weight and activation tensor. Quantization is symmetric for both.

SmoothQuant is used in our quantization method.

According to SmoothQuant, activations are harder to quantize than weights, outliers persist in fixed channels.

So we should “smooth” the input activation by dividing it by a per-channel smoothing factor s.

```math
Y = (Xdiag(s)^{−1}) \cdot (diag(s)W) = \hat{X}\hat{W}
```

```math
s_j = \textrm{max}(|X_j|)^\alpha / \textrm{max}(|W_j|)^{1−\alpha}
```

### Weights

After "smooth", the weights will multiply s per-channel

$$ \hat{W_j} = W_j * s_j $$ 

because we also quantize output to int8, so we wil calculate output channel scales.

$$ \hat{W_k} = \hat{W_k} / s_k $$ 

then, we will do per-channel weight quantization to int8 range [-128, 127].

### Activations

After "smooth", there will be a multiply operator before matmul.

$$ \hat{X_j} = X_j / s_j $$

then, we wil do per-tensor activation quantization to int8 range [-128, 127].

### Outputs
We also quantize the output to int8 range [-128, 127]. And add a multily operator after matmul.

$$ S_k = \textrm{max}(|Y_k|) / max(|\hat{W_k}|)^\alpha $$

$$ Y_k = \hat{Y_k} \cdot S_k $$

## Calibration

We pass a set of calibration samples through the neural network executed in bfloat16 point to obtain a profile of every activation tensor of the network. The profile consists of maximum absolute values over all samples. And min-max strategy is used in calibration.