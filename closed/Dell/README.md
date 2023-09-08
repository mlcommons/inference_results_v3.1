# MLPerf Inference v3.1 Implementations
This is a repository of Dell Technologies servers using optimized implementations for [MLPerf Inference Benchmark v3.1](https://www.mlperf.org/inference-overview/).

# Implementations
## Benchmarks
**Please refer to /closed/Dell for detailed instructions for NVIDIA GPU & Triton submissions, including performance guides, and instructions on how to run with new systems.** 

**Please refer to /closed/Qualcomm for detailed instructions for Qualcomm Cloud AI 100 submissions.**

**Please refer to /closed/Intel for detailed instructions for Intel CPU submissions.**
  
The following benchmarks are part of our submission for MLPerf Inference v3.1:
- [3d-unet](code/3d-unet/tensorrt/README.md)
- [bert](code/bert/tensorrt/README.md)
- [dlrmv2](code/dlrm-v2/tensorrt/README.md)
- [gptj](code/gptj/tensorrt/README.md)
- [rnnt](code/rnnt/tensorrt/README.md)
- [retinanet](code/retinanet/README.md)
- [resnet50](code/resnet50/tensorrt/README.md)

# Dell Technologies Submission Systems

The closed systems that Dell has submitted are:
- Datacenter Systems
  - Dell PowerEdge R750xa
    - NVIDIA A100-PCIe-80GB
    - NVIDIA H100-PCIe-80GB
    - NVIDIA H100-80C (virtualized)
  - Dell PowerEdge R760xa
    - NVIDIA H100-PCIe-80GB
    - NVIDIA L40
  - Dell PowerEdge XE8640
    - NVIDIA H100-SXM-80GB
  - Dell PowerEdge XE9640
    - NVIDIA H100-SXM-80GB
  - Dell PowerEdge XE9680
    - NVIDIA A100-SXM-80GB / 500W
    - NVIDIA H100-SXM-80GB
  - Dell PowerEdge R760
    - Intel Platinum 8480+  
- Edge Systems
  - Dell PowerEdge R760xa
    - NVIDIA L40
  - Dell PowerEdge XE8640
    - NVIDIA H100-SXM-80GB
  - Dell PowerEdge XR4520c
    - NVIDIA L4
    - Qualcomm Cloud AI 100 Standard
  - Dell PowerEdge XR5610
    - NVIDIA L4
  - Dell PowerEdge XR7620
    - NVIDIA L4


