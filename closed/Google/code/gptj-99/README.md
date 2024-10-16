# GPT-J Reference Implementation

Below, we describe different code we developed for implementing the GPTJ 6B
benchmark on our target framework:

- convert_gptj_ckpt.py: This file contains the checkpoint converter tool used
  to convert the weights from the pytorch model to the SAX model. The usage is
  shown in the code section below.
- gptj.py: This file contains the implementation of the GPTJ 6B model on the
  SAX framework.
  - [SAX framework](https://github.com/google/saxml) is an OSS framework. 
    SAX framework will support the v5e platform once it is publicly available.
    For the Preview submission, we used an internal build of SAX that supports
    the target platform.
- backend.py: This file contains the backend implementation that uses a SAX
  client to send requests to the model that runs inside SAX.
- Other python files contain the incremental changes to the original benchmark
  files, such as new flags in main.py to control client threads in backend.py.

## Setup TPU

The submission is using SAX on TPU v5e. TPU v5e is currently not publicly 
available and hence the "Preview submission". SAX setup involves the client and
server components. For exhaustive documentation on these components to publicly 
available documentation. The model server setup should be similar to bringing up
SAX on TPUv4.

## Load the Model and Run the Benchmark
Running the model follows the following steps:
1. Use the convert_gptj_ckpt.py tool to convert the GPTJ 6B pytorch checkpoint
to a jax checkpoint.
2. Build and run the SAX admin config, admin server and the SAX model server 
processes on the VM.
3. Run main.py and evaluation.py similar to the reference model.

SAX setup instructions can be found [here.](https://github.com/google/saxml)
For convenience, the following important commands are inlined below.

The submission includes a reference of GPTJ on SAX. We aimed to match the
model definitions to the Hugging face pytorch reference. The submission model 
definition is available at /code/gptj-99/params/gptj.py.

A class must be chosen during the publishing step. For example, the base is the
GPTJ class. One can control the SAX behavior by adding attributes that modifies
the SAX config. This has been done for convenience, and added to the same file. 
We used GPTJ4BS32Int8Opt10Wait40MB6 for Server, and GPTJ4BS32Int8Opt10Early for 
Offline.

```
convert_checkpoint() {
  # Install the latest main branch of huggingface/transformers
  pip3 install git+https://github.com/huggingface/transformers

  # Get a checkpiont from the GPTJ family 
  # (e.g., https://huggingface.co/EleutherAI/gpt-j-6b.) For this benchmark,
  # use the fine-tuned final checkpoint available as the reference model.

  # This points to
  # https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/gptj/modeling_flax_gptj.py
  # and in the default config, use_parallel_residual is true

  python3 -m convert_gptj_ckpt --base $MODEL_PATH --pax pax_gptj_checkpoint
}

run_admin_config() {
  bazel run saxml/bin:admin_config -- \
    --sax_cell=/sax/test \
    --sax_root=gs://${GSBUCKET}/sax-root \
    --fs_root=gs://${GSBUCKET}/sax-fs-root \
    --alsologtostderr
}

run_admin_server() {
  bazel run saxml/bin:admin_server -- \
    --sax_cell=/sax/test \
    --sax_root=gs://${GSBUCKET}/sax-root \
    --port=10000 \
    --alsologtostderr
}

run_model_server() {
SAX_ROOT=gs://${GSBUCKET}/sax-root \
bazel run saxml/server:server -- \
  --sax_cell=/sax/test \
  --port=10001 \
  --platform_chip=v5e-4 \
  --platform_topology=2x2x1 \
  --alsologtostderr
}

MODEL_PATH=/sax/test/gptj4tokenized # Used by backend.py to create a client
publish_model() {
  saxutil publish \
  /sax/test/gptj4tokenized \
  saxml.server.pax.lm.params.GPTJ4BS32Int8Opt10Early \
  ${CONVERTED_CHECKPOINT} \
  1
}

# After running the commands above, use the main.py similar to the reference 
# model. This will run an initial test inference, you should be able to see log 
# activity from model server terminal.
run_benchmark() {
  python main.py \
    --scenario Server \
    --model-path ${MODEL_PATH} \
    --dataset-path ${DATASET_PATH} \
    --accuracy \
    --max-examples 20 \
    --perf-examples 20 \
    --log-interval 5 \
    --log-path /mlperf_inference/language/gpt-j/saxml/test_loadgen_logs
}

```

## License:

Apache License Version 2.0.
