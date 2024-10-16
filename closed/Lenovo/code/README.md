# Lenovo code submission structure
Lenovo submissions to MLPerf inference v3.1 consists of results performed on the following configurations

* Systems that utilize NVidia GPUs for accelerating computations
* Systems that utilize QualComm accelerators for AI workloads

Each of these may have their own code implementations for the same benchmark. For this reason there are
subdirectories within each individual benchmark subdirectory in this section

## NVidia GPU accelerator submissions
The benchmarks are executed using the NVidia supplied container environment released in 
[the NGC catalog](https://catalog.ngc.nvidia.com/) MLPerf inference tagged
mlpinf-v3.1.4-cuda12.2-cudnn8.9-x86_64-ubuntu20.04-l4-partner

The full code can be found by accessing this container, or can be directly viewed in the NVIDIA submission within this
MLPerf submission iteration.

For convenience there is also a subdirectory `NVIDIA` within each code directory containing this information and
any benchmark particular comments referring to the Lenovo submission to MLPerf.

## Qualcomm AI accelerator submissions
The benchmarks are executed using code developed by Qualcomm and for each benchmark the pertinent code
is documented in a subdirectory with a name ending with "qaic", *eg* `bert_squad_kilt_loadgen_qaic`.
