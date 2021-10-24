# SPARSEDNN

**If you want to use this repo, please send me an email: zihengw@stanford.edu, or raise a Github issue. 
**

**Sparse INT8 knerels are here.
**

Fast sparse deep learning on CPUs. This is the kernel library generator described in the paper: https://arxiv.org/abs/2101.07948
My other repo on sparse deep learning on GPUs: https://github.com/marsupialtail/gpu-sparsert. Will merge at some point when I'm feeling less lazy.

Python API: python fastsparse.py. Minimal required dependencies. Should work anywhere.

C++ API: check out driver_cpu.cpp, or run autotune_cpu_random.sh 128 128 128 0. This requires cnpy to read numpy files, so make sure that you can link to cnpy.
C++ API only: for block sparse int8 matrix multiply, run autotune_cpu_random_int8.cpu 512 512 128.

Python API has some bad overhead due to using ctypes. This is noticeable for smaller matrices but not really noticeable for large matrices. The benchmarkings done in the Arxiv paper was all done with the C++ API. 

**Work that is not yet open sourced: kernel generator for sparse convolutions (as described in the Arxiv paper) using implicit convolution, lightweight inference engine to get end-to-end results. If interested in any of this please email.**

FAQs:
1) How does this compare to Neuralmagic? Last time I checked the deepsparse library does not allow you to run kernel-level benchmarks. If you care about end to end neural network acceleration, you should definitely go with Neuralmagic if they happen to support your model.
2) Future work? This is not exactly along the lines of my PhD thesis so I work on this sparingly. If you want to contribute to this repo you could make a Pytorch or Tensorflow custom op with the Python or C++ API. However it's unclear how gradients would work, and you will have to compile this op with the fixed sparsity pattern, something that the current Pytorch/Tensorflow frameworks might not support that well. 
