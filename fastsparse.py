import numpy as np
import os
from ctypes import *
import time


class Input(Structure):
    _fields_ = [
        ("AB_vals",c_void_p),
        ("AB_bias",c_void_p),
        ("BC",c_void_p),
        ("AC",c_void_p),
        ("start",c_int32),
        ("end",c_int32)
    ]


class SpMM:

    # takes in a numpy sparse matrix in the dense array format with 0s, and the C_dimension of the dense matrix
    def __init__(self, matrix, C_dim, bias=None):
        self.A_dim = matrix.shape[0]
        self.B_dim = matrix.shape[1]
        self.matrix = matrix
        self.C_dim = C_dim
        self.bias = bias
        if self.bias is not None:
            assert len(self.bias) == self.A_dim

    def compile(self,name = "spmm", val_name = "vals.npy", bias_name = "bias.npy", AT = 6, CT = 2, B_blocks = 1, C_blocks = 1, no_relu=True,epi="NONE"):

        import code_gen_cpu
        if not "avx2" in open("/proc/cpuinfo","r").read():
            print("We need at least AVX2.")
            raise Exception
        if "avx512" in open("/proc/cpuinfo","r").read():
            code_gen_cpu.AVX512 = True
            code_gen_cpu.VEC = 16
        else:
            code_gen_cpu.AVX512 = False
            code_gen_cpu.VEC = 8

        code_gen_cpu.FUNC_NAME = name
        code_gen_cpu.EPI = epi
        code_gen_cpu.IN_FORMAT = "NCHW"
        code_gen_cpu.OUT_FORMAT = "NCHW"
        code_gen_cpu.GY = 1
        code_gen_cpu.FUSE_END = False
        code_gen_cpu.NO_RELU = no_relu
        code_gen_cpu.A_dim = self.A_dim
        code_gen_cpu.B_dim = self.B_dim
        code_gen_cpu.C_dim = self.C_dim
        code_gen_cpu.AT = AT
        code_gen_cpu.CT = CT
        code_gen_cpu.B_blocks = B_blocks
        code_gen_cpu.C_blocks = C_blocks
        code_gen_cpu.outfile = "out.cpp"
        code_gen_cpu.outfile_asm = "out.s"
        code_gen_cpu.bias = self.bias
        assert self.C_dim % C_blocks == 0

        code_gen_cpu.TSZ = self.C_dim // C_blocks if self.C_dim % C_blocks == 0 else self.C_dim // C_blocks + 1
        code_gen_cpu.X86 = True
        code_gen_cpu.ARM = False
        NRS = False

        BA = self.matrix.transpose()
        #print(BA.shape)
        BA = BA.squeeze()

        code_gen_cpu.AB_vals = []
        code_gen_cpu.A_idx = []
        code_gen_cpu.B_idx = []
        code_gen_cpu.AB_block_offs = [0]
        #global off
        code_gen_cpu.off = 0

        """
        We are going to redo BA here to remove some empty rows
        """

        nnz_cols = np.unique(np.where(BA)[1])
        code_gen_cpu.mapping = {i : nnz_cols[i] for i in range(len(nnz_cols))}
        #print(mapping)
        BA = BA[:,nnz_cols]
        code_gen_cpu.A_dim = len(nnz_cols)

        if code_gen_cpu.A_dim % AT == 0:
            A_blocks = code_gen_cpu.A_dim // AT
        else:
            A_blocks = code_gen_cpu.A_dim // AT + 1

        code_gen_cpu.gencode(BA,self.C_dim,A_blocks,C_blocks,name="bump")

        self.AB_vals = np.array(code_gen_cpu.AB_vals)
        np.save(val_name,np.array(self.AB_vals))
        if self.bias is not None:
            np.save(bias_name,np.array(self.bias))
        else:
            self.bias = np.ones((self.A_dim))
            #np.save(bias_name,np.array(self.bias))
        os.system("gcc -c out.s")
        os.system("ar rvs " + name + ".a out.o >/dev/null 2>&1")
        os.system("gcc -shared out.s -o " + name + ".so ")
        os.system("rm out.o out.s out.cpp")
        self.libc = CDLL(name + ".so")

    def load(self,sl_name, vec_name, bias_name = None):
        self.libc = CDLL(sl_name)
        self.AB_vals = np.load(vec_name)
        assert self.AB_vals.dtype == np.float32
        if bias_name:
            self.bias = np.load(bias_name)
        else:
            # we will not be using the values in the kernel anyways
            self.bias = np.ones((self.A_dim))
        assert len(self.bias) == self.A_dim



    def run(self,BC):
        self.AC = np.empty((self.A_dim,self.C_dim),dtype=np.float32)
        w = self.AC.ctypes.data
        z = BC.ctypes.data
        x = self.AB_vals.ctypes.data
        AB_bias = self.bias
        y = AB_bias.ctypes.data
        self.arg = pointer(Input(x,y,z,w,0,1))
        self.libc._spmm(self.arg)
        return self.AC

    def ref_run(self,BC):
        return np.dot(self.matrix,BC).astype(np.float32)


a = np.load("matrix.npy")
b = SpMM(a,128)
b.compile()
test_input = np.random.normal(size=(128,128)).astype(np.float32)
b.run(test_input)
reference = b.ref_run(test_input)
assert np.abs(np.sum(np.sum(b.AC-reference))) < 0.1
#
start = time.time()
for i in range(1000):
    b.run(test_input)
print((time.time()-start) * 1000)

