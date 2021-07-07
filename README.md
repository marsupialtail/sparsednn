# SPARSEDNN

**If you want to use this repo, please send me an email: zihengw@stanford.edu, or raise a Github issue. 
**

Fast sparse deep learning on CPUs. This is the kernel library generator described in the paper: https://arxiv.org/abs/2101.07948

Python API: python fastsparse.py. Minimal required dependencies. Should work anywhere.

C++ API: check out driver_cpu.cpp, or run autotune_cpu_random.sh 128 128 128 0. This requires cnpy to read numpy files, so make sure that you can link to cnpy.

Python API has some bad overhead due to using ctypes. This is noticeable for smaller matrices but not really noticeable for large matrices. The benchmarkings done in the Arxiv paper was all done with the C++ API. 

If you want to use this repo, please send me an email: zihengw@stanford.edu, or raise a Github issue. 
