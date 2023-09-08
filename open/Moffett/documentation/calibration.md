MLPerf Inference Calibration and Quantization Details
---

## Moffett Quantization Rules

- Post-training 8-bit quantization with static range on weight and activation tensor are used. Quantization is symmetric for both.

- Quantization technique close to the SmoothQuant [1] is used, so a per-channel constant scale-vector is factorized out of each weight tensor, for both its input channels and output channels.

- The SmoothQuant technqiue is applied to every Linear Operator.

## Reference

[1] Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023, July). Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning (pp. 38087-38099). PMLR.