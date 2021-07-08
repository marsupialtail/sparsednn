# SPARSEDNN

**If you want to use this repo, please send me an email: zihengw@stanford.edu, or raise a Github issue. 
**

Fast sparse deep learning on CPUs. This is the kernel library generator described in the paper: https://arxiv.org/abs/2101.07948

Python API: python fastsparse.py. Minimal required dependencies. Should work anywhere.

C++ API: check out driver_cpu.cpp, or run autotune_cpu_random.sh 128 128 128 0. This requires cnpy to read numpy files, so make sure that you can link to cnpy.

Python API has some bad overhead due to using ctypes. This is noticeable for smaller matrices but not really noticeable for large matrices. The benchmarkings done in the Arxiv paper was all done with the C++ API. 

FAQs:
1) How does this compare to Neuralmagic? Last time I checked the deepsparse library does not allow you to run kernel-level benchmarks. If you care about end to end neural network acceleration, you should definitely go with Neuralmagic if they happen to support your model.
2) Future work? This is not exactly along the lines of my PhD thesis so I work on this sparingly. If you want to contribute to this repo you could make a Pytorch or Tensorflow custom op with the Python or C++ API. However it's unclear how gradients would work, and you will have to compile this op with the fixed sparsity pattern, something that the current Pytorch/Tensorflow frameworks might not support that well. 
