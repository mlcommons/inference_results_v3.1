

# Run MLPerf inference benchmarks out-of-the-box

This documentation will help you run, reproduce and compare [MLPerf inference benchmarks](https://arxiv.org/abs/1911.02549) 
out-of-the-box using the Collective Mind unified interface.

Please check [MLPerf inference submitter orientation slides](https://doi.org/10.5281/zenodo.8144274)
explaining why [MLCommons](https://mlcommons.org) is developing an open-source, technology-agnostic 
and non-intrusive [Collective Mind automation language (CM)](https://github.com/mlcommons/ck)
and [Collective Knowledge Playground (CK)](https://access.cknowledge.org/playground/?action=experiments) 
to make it easier to run, reproduce, optimize and compare MLPerf inference benchmarks 
across diverse software, hardware, models and data sets from different vendors.

Note that this project is under heavy development by the [MLCommons Task Force on Automation and Reproducibility](https://github.com/mlcommons/ck/blob/master/docs/taskforce.md),
[cTuning.org](https://www.linkedin.com/company/ctuning-foundation) and [cKnowledge.org](https://www.linkedin.com/company/cknowledge)
led by [Grigori Fursin](https://cKnowledge.org/gfursin) and [Arjun Suresh](https://www.linkedin.com/in/arjunsuresh).
You can learn more about our plans and long-term vision from our [ACM REP keynote (June 2023)](https://doi.org/10.5281/zenodo.8105339).

Don't hesitate to get in touch with us using this [public Discord server](https://discord.gg/JjWNWXKxwT) 
to get help with MLPerf optimization and submissions, ask questions, provide your feedback, add new benchmark implementations, models, data sets and hardware backends,
and participate in our [reproducibility and optimization challenges](https://access.cknowledge.org/playground/?action=challenges).


## Install CM automation language

Install the [MLCommons CM automation language](https://doi.org/10.5281/zenodo.8105339) as described in this [guide](https://github.com/mlcommons/ck/blob/master/docs/installation.md). 
It is a small Python library with `cm` and `cmr` command line front-ends and minimal dependencies including Python 3+, Git and wget.

If you encounter problems, please report them at [GitHub](https://github.com/mlcommons/ck/issues).


## Install repository with CM automations

Install the MLCommons repository with [reusable and portable automation recipes (CM scripts)](https://github.com/mlcommons/ck/tree/master/cm-mlops/script) via CM.
These scripts are being developed and shared by the community and MLCommons under Apache 2.0 license 
to enable portable, modular, and technology-agnostic benchmarks and applications 
that can automatically run with any software, hardware, models and data sets.

```bash
cm pull repo mlcommons@ck --checkout=v3.1
```



Note that CM will store all such repositories and downloaded/installed data sets, models and tools
in your `$HOME/CM` directory. 

Since MLPerf benchmarks require lots of space (somethings hundreds of Gigabytes), 
you can change the above location to some large scratch disk using `CM_REPOS` 
environment variable as follows:

```bash
export CM_REPOS={new path to CM repositories and data}
echo "CM_REPOS=${CM_REPOS} >> $HOME/.bashrc"
```

You will need to pull above repository with CM automations again to place it in the new directory.



## Setup virtual environment

If you plan to use your native environment to run MLPerf benchmarks, we suggest you to set up
a Python virtual environment via CM to avoid contaminating your existing Python installation:

```bash
cm run script "install python-venv" --name=mlperf
export CM_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf"
```

CM will install a new Python virtual environment in CM cache and will install all Python dependencies there:
```bash
cm show cache
```

Note that you can install multiple virtual environments with different names and python versions (Linux):
```bash
cm run script "install python-venv" --version=3.10.8 --name=mlperf2
export CM_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf2"
```



## Prepare hardware

Read this section if you want to run MLPerf benchmarks in a native environment, i.e. without containers.

### CPUs

If you plan to run MLPerf benchmarks on x64 and/or Arm64 CPUs, no extra setup is necessary.

### CUDA GPUs

If you plan to use CUDA in your native environment, please follow [this guide](../../installation-cuda.md) to set it up on your system.

### Nvidia Jetson AGX Orin

Follow [this guide](../setup/setup-nvidia-jetson-orin.md).



## Run benchmarks and submit results

Please check our [MLPerf inference submitter orientation slides (July 2023)](https://doi.org/10.5281/zenodo.8144274)
explaining why we have developed a common CM interface to run all MLPerf benchmarks.

Note that only official (registered) MLCommons members can submit results to MLPerf inference.
As an alternative, you can also participate in our community submissions to MLPerf
via the [cTuning foundation](https://www.linkedin.com/company/ctuning-foundation) (a founding member of MLCommons).


We provided a unified CM interface to run the following MLPerf inference benchmarks:
1. [Language processing](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference/bert) using Bert-Large model and Squad v1.1 dataset
2. [Language processing](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference/gpt-j) using GPT-J model and CNN Daily Mail dataset
3. [Image Classification](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference/resnet50) using ResNet50 model and Imagenet-2012 dataset
4. [Image Classification](https://github.com/mlcommons/ck/blob/master/cm-mlops/script/run-mlperf-inference-mobilenet-models/README-about.md) using variations of MobileNets and EfficientNets and Imagenet-2012 dataset
5. [Object Detection](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference/retinanet) using Retinanet model and OpenImages dataset
6. [Speech Recognition](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference/rnnt) using RNNT model and LibriSpeech dataset
7. [Medical Imaging](https://github.com/mlcommons/ck/tree/master/docs/mlperf/inference/3d-unet)  using 3d-unet model and KiTS19 dataset
8. Recommendation using DLRMv2 model and Criteo multihot dataset

All seven benchmarks can participate in the datacenter category.
All seven benchmarks except Recommendation can participate in the edge category. 

Note that `language processing` and `medical imaging` benchmarks must achieve a higher accuracy of at least `99.9%` of the FP32 reference model
in comparison with `99%` default accuracy requirement for all other models.

The `recommendation` benchmark has a high-accuracy variant only. Currently, we are not supporting the `recommendation` benchmark in CM 
because we did not have a required high-end server for testing. 




### Measure power

Power measurement is optional for MLPerf inference benchmark submissions and is known to be very difficult to set up and run.
However, if your system have a good power efficiency, it is great to showcase it and compare against other systems.
That's why we fully automated power measurements for MLPerf inference benchmark in CM.

For any above MLPerf inference benchmark, you can turn on power measurements by adding the following flags to the CM command:
```
--power=yes \
--adr.mlperf-power-client.power_server=<Power server IP> \
--adr.mlperf-power-client.port=<Power server port>
```

On the hardware side, you can follow [this tutorial](https://github.com/mlcommons/ck/blob/master/docs/tutorials/mlperf-inference-power-measurement.md) 
to set up your power analyzer and connect it with your host platform.

Note that the [cTuning foundation](https://www.linkedin.com/company/ctuning-foundation)
has several power analyzer and can help you test your MLPerf benchmark implementations
on our system.



## Debug benchmarks


Since CM language uses native OS scripts with python wrappers, it is relatively straightforward to debug it using your existing tools.

The unified CM interface to run MLPerf inference benchmarks out of the box is implemented using these CM scripts:
* [run-mlperf-inference-app](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/run-mlperf-inference-app)
  * [app-mlperf-inference-reference](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/app-mlperf-inference-reference)
  * [app-mlperf-inference-nvidia](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/app-mlperf-inference-nvidia)
  * [app-mlperf-inference-cpp](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/app-mlperf-inference-cpp)
  * [app-mlperf-inference-tflite-cpp](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/app-mlperf-inference-tflite-cpp)

You can add `--debug` flag to your CM command line when running MLPerf benchmarks
to open a shell with all MLPerf environment variables prepared to 
run and debug the final MLPerf loadgen tool manually.

You can also use GDB by adding environment variable `--env.CM_RUN_PREFIX="gdb --args "` to the CM command line.

Please check [this documentation](../../debugging.md) for more details.



## Update CM language and scripts

The concept of CM language is to always keep backward compatibility of the high-level interface 
while improving and extending low-level scripts, CLI and environment variables.

You should be able to update CM language and scripts at any time as follows:
```bash
python3 -m pip install cmind -U
cm pull repo mlcommons@ck
```

However, some local installations and downloads may become outdated in CM cache.
In such case, you can either start from scratch by cleaning all CM cache entries as follows
```bash
cm rm cache -f
```

or by cleaning only entries related to updated components such as MLPerf inference sources and harnesses:
```bash
cm show cache
cm rm cache --tags=inference,src -f
cm rm cache --tags=harness -f
```



## Optimize benchmarks

We are developing `CM experiment automation` to run multiple experiments, automatically explore multiple parameters, 
record results and reproduce them by the workgroup.

Please check this [documentation](../../../cm-mlops/automation/experiment/README-extra.md) for more details.

*This is ongoing development.*






## Questions? Suggestions?

Get in touch via [public Discord server](https://discord.gg/JjWNWXKxwT).
