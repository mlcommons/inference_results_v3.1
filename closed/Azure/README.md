# MLPerf Inference v3.1 by Azure
This is a repository of Azure results for the [MLPerf](https://mlcommons.org/en/) Inference Benchmark.
This README is a quickstart tutorial on how to reproduce our performance as a public / external user.

This repository is based on NVIDIA's code for MLPerf Inference v3.1 but has been optimized for Azure's infrastructure.

---

### MLPerf Inference Policies and Terminology

This is a new-user guide to learn how to use Azure's MLPerf Inference submission repo. **To get started with MLPerf Inference, first familiarize yourself with the [MLPerf Inference Policies, Rules, and Terminology](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)**. This is a document from the MLCommons committee that runs the MLPerf benchmarks, and the rest of all MLPerf Inference guides will assume that you have read and familiarized yourself with its contents. The most important sections of the document to know are:

- [Key terms and definitions](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#11-definitions-read-this-section-carefully)
- [Scenarios](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#3-scenarios)
- [Benchmarks and constraints for the Closed Division](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#411-constraints-for-the-closed-division)
- [LoadGen Operation](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#51-loadgen-operation)

### Azure's Submission

Azure submits with the latest ND H100 v5-series powered by NVIDIA H100 SXM GPUs, in the datacenter category.

Our submission implements several inference harnesses stored under closed/Azure/code/harness:

- What we refer to as "custom harnesses": lightweight, barebones, C++ harnesses
    - GPTJ harness

Benchmarks are stored in `closed/Azure/code`. Each benchmark, as per MLPerf Inference requirements, contains a `README.md` detailing instructions and documentation for that benchmark. **However**, as a rule of thumb, **follow this guide first** from start to finish before moving on to benchmark-specific `README`s, as this guide has many wrapper commands to automate the same steps across multiple benchmarks at the same time.

### Use a non-root user

If you're already a non-root user, simply don't use sudo for any command that is not a package install or a command that specifically has 'sudo' contained in it. Otherwise, create a new user. It is advisable to make this new user a sudoer, but as said before, do not invoke sudo unless necessary.

Make sure that your user is in docker group already. If you get permission issue when running docker commands, please add the user to docker group with `sudo usermod -a -G docker $USER`.

### Software Dependencies

### Datacenter/Desktop based systems

Our submission uses Docker to set up the environment. Requirements are:

- [Docker CE](https://docs.docker.com/engine/install/)
    - If you have issues with running Docker without sudo, follow this [Docker guide from DigitalOcean](https://www.digitalocean.com/community/questions/how-to-fix-docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socket) on how to enable Docker for your new non-root user. Namely, add your new user to the Docker usergroup, and remove ~/.docker or chown it to your new user.
    - You may also have to restart the docker daemon for the changes to take effect:

```
$ sudo systemctl restart docker
```

- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
    - libnvidia-container >= 1.4.0
- NVIDIA Driver Version 510.xx or greater

```
$ sudo systemctl restart docker
```


**Note that once the scratch space is setup and all the data, models, and preprocessed datasets are set up, you do not have to re-run this step.** You will only need to revisit this step if:

- You accidentally corrupted or deleted your scratch space
- You need to redo the steps for a benchmark you previously did not need to set up
- You, Azure, or MLCommons has decided that something in the preprocessing step needed to be altered

Once you have obtained a scratch space, set the `MLPERF_SCRATCH_PATH` environment variable. This is how our code tracks where the data is stored. By default, if this environment variable is not set, we assume the scratch space is located at `/home/mlperf_inference_data`. Because of this, it is highly recommended to mount your scratch space at this location.

```
$ export MLPERF_SCRATCH_PATH=/path/to/scratch/space
```
This `MLPERF_SCRATCH_PATH` will also be mounted inside the docker container at the same path (i.e. if your scratch space is located at `/mnt/some_ssd`, it will be mounted in the container at `/mnt/some_ssd` as well.)

Then create empty directories in your scratch space to house the data:

```
$ mkdir $MLPERF_SCRATCH_PATH/data $MLPERF_SCRATCH_PATH/models $MLPERF_SCRATCH_PATH/preprocessed_data
```
After you have done so, you will need to download the models and datasets, and run the preprocessing scripts on the datasets. **If you are submitting MLPerf Inference with a low-power machine, such as a mobile platform, it is recommended to do these steps on a desktop or server environment with better CPU and memory capacity.**

Enter the container by entering the `closed/Azure` directory and running:

```
$ make prebuild # Builds and launches a docker container
```
Then inside the container, you will need to do the following:

```
$ echo $MLPERF_SCRATCH_PATH  # Make sure that the container has the MLPERF_SCRATCH_PATH set correctly
$ ls -al $MLPERF_SCRATCH_PATH  # Make sure that the container mounted the scratch space correctly
$ make clean  # Make sure that the build/ directory isn't dirty
$ make link_dirs  # Link the build/ directory to the scratch space
$ ls -al build/  # You should see output like the following:
total 8
drwxrwxr-x  2 user group 4096 Jun 24 18:49 .
drwxrwxr-x 15 user group 4096 Jun 24 18:49 ..
lrwxrwxrwx  1 user group   35 Jun 24 18:49 data -> $MLPERF_SCRATCH_PATH/data
lrwxrwxrwx  1 user group   37 Jun 24 18:49 models -> $MLPERF_SCRATCH_PATH/models
lrwxrwxrwx  1 user group   48 Jun 24 18:49 preprocessed_data -> $MLPERF_SCRATCH_PATH/preprocessed_data
```
Once you have verified that the `build/data`, `build/models/`, and `build/preprocessed_data` point to the correct directories in your scratch space, you can continue.

### Download the dataset and the model

GPT-J benchmark contains a `README.md` (located at `closed/Azure/code/gptj/tensorrt/README.md`) that explains how to download and set up the dataset and model files for that benchmark manually. 
You must follow the same README.md file to preprocess the data.


## Running your first benchmark

**First, enter closed/Azure**. From now on, all of the commands detailed in this guide should be executed from this directory. This directory contains our submission code for the [MLPerf Inference Closed Division](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#61-closed-division). 

***IMPORTANT**:* **Do not run any commands as root (Do not use sudo).** Running under root messes up a lot of permissions, and has caused many headaches in the past. If for some reason you missed the part in the beginning of the guide that warned to not use root, you may run into one of the following problems:

- Your non-root account cannot use Docker.
    - See the 'Use a non-root user' section at the beginning of the guide for instructions on how to fix this.
- You cloned the repo as root, now you have a bunch of file permission errors where you cannot write to some directories.
    - It is highly recommended to chown the entire repo to the new non-root user, or better yet to re-clone the repo with the new user.
    - You will likely also need to re-run the 'git config' and 'Docker login' steps in the 'Cloning the Repo' Section, as those are user-specific commands, and would only have affected 'root'.
- Make sure that your new user has at least read-access to the scratch spaces. If the scratch space was set up incorrectly, only 'root' will be able to read the scratch spaces. If the scratch spaces are network-based filesystems, check /etc/fstab for the settings as well.

### Launching the environment on datacenter/desktop systems

You will need to launch the Docker container first:

```
$ make prebuild
```
***Important notes:***

- The docker container does not copy the files, and instead **mounts** the working directory (closed/Azure) under /work in the container. This means you can edit files outside the container, and the changes will be reflected inside as well.
- In addition to mounting the working directory, the scratch spaces are also mounted into the container. Likewise, this means if you add files to the scratch spaces outside the container, it will be reflected inside the container and vice versa.
- If you want to mount additional directories/spaces in the container, use `$ make prebuild DOCKER_ARGS="-v <from>:<to> -v <from>:<to>"`
- If you want to expose only a certain number of GPUs in the container, use `$ NVIDIA_VISIBLE_DEVICES=0,2,4... make prebuild`

```



### Building the binaries

```
$ make build
```

This command does several things:

1. Sets up symbolic links to the models, datasets, and preprocessed datasets in the MLPerf Inference scratch space in build/
2. Pulls the specified hashes for the subrepositories in our repo:
    1. MLCommons Inference Repo (Official repository for MLPerf Inference tools, libraries, and references)
    2. NVIDIA Triton Inference Server
3. Builds all necessary binaries for the specific detected system

**Note**: This command does not need to be run every time you enter the container, as build/ is stored in a mounted directory from the host machine. It does, however, need to be re-run if:

- Any changes are made to harness code
- Repository hashes are updated for the subrepositories we use
- You are re-using the repo on a system with a different CPU architecture

### Running the actual benchmark

Our repo has one main command to run any of our benchmarks:

```
$ make run RUN_ARGS=""--benchmarks=gptj --scenarios=offline,server"
```
This command is actually shorthand for a 2-step process of building, then running TensorRT engines:

```
$ make generate_engines RUN_ARGS="..."
$ make run_harness RUN_ARGS="..."
```


### How do I view the logs of my previous runs?

Logs are saved to `build/logs/[timestamp]/ND_H100_v5/...` every time `make run_harness` is called.


To run the benchmarks under the higher accuracy target, specify `--config_ver="high_accuracy"` as part of `RUN_ARGS`:

```
$ make run RUN_ARGS="--benchmarks=gptj --scenarios=offline --config_ver=high_accuracy"
```


### Further reading

More specific documentation and for debugging:

- documentation/performance_tuning_guide.md - Documentation related to tuning and benchmarks via configuration changes
- documentation/commands.md - Documentation on commonly used Make targets and RUN_ARGS options
- documentation/FAQ.md - An FAQ on common errors or issues that have popped up in the past
- documentation/submission_guide.md - Documentation on officially submitting our repo to MLPerf Inference
- documentation/calibration.md - Documentation on how we use calibration and quantization for MLPerf Inference
- documentation/heterogeneous_mig.md - Documentation on the HeteroMIG harness and implementation
