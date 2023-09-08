## Preprocess all data
#make preprocess_data

## Preprocess specific data
#make preprocess_data BENCHMARKS="resnet50 retinanet bert rnnt 3d-unet gptj6b" dlrm_v2 gpt175b
make preprocess_data BENCHMARKS="resnet50"
make preprocess_data BENCHMARKS="retinanet"
make preprocess_data BENCHMARKS="bert"
make preprocess_data BENCHMARKS="rnnt"
make preprocess_data BENCHMARKS="3d-unet"
make preprocess_data BENCHMARKS="gptj6b"


