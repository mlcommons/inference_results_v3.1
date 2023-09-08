# MLPerf Inference v3.1 - Calibration


## For Results using Nvidia Original Implementation

For the results taken using Nvidia original implementation we are following the same calibration procedure detailed by [Nvidia for their MLPerf Inference v3.0 submissions](https://github.com/mlcommons/inference_results_v3.0/blob/master/closed/NVIDIA/documentation/calibration.md)

## For Results using TFLite C++ Implementation

For the mobilenet and efficientnet submissions, we use quantized models from [TensorFlow Hub](https://tfhub.dev/). Details about the post-training quantization done for these models can be seen [here](
https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization_of_weights_and_activations)

## For Results using Deepsparse backend

For the models run via deepsparse backend, we are following the same calibration detailed by [Neural Magic for their MLPerf Inference v3.0 submissions](https://github.com/mlcommons/inference_results_v3.0/blob/main/open/NeuralMagic/calibration.md)
