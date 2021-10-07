# this program basically does a constexpr and generates cuda code
import textwrap
import numpy as np
from code_fragments import *
from utils import *


import argparse
parser = argparse.ArgumentParser(description='CodeGen V1')

parser.add_argument('--A_dim', type=int, default=12)
parser.add_argument('--B_dim', type=int, default=12)
parser.add_argument('--C_dim', type=int, default=12)
parser.add_argument('--AT', type=int, default=12)
parser.add_argument('--C_blocks', type=int, default=12)
parser.add_argument('--CT',type=int, default=1)
parser.add_argument('--B_blocks',type=int,default=1)
parser.add_argument('--Gy', type=int, default=12)
parser.add_argument('--infile', default=None, type=str)
parser.add_argument('--infile_bias', default=None, type=str)
parser.add_argument('--outfile', default=None, type=str)
parser.add_argument('--outfile_asm', default= None, type = str)
parser.add_argument('--in_format', default="NCHW",type=str)
parser.add_argument('--out_format', default="NCHW",type=str)
parser.add_argument('--Tsb',type=float,default=1)
parser.add_argument('--fuse',default=False,action='store_true')
parser.add_argument('--x86',default=False,action='store_true')
parser.add_argument('--arm',default=False,action='store_true')
parser.add_argument('--threads',type = int, default=4)
parser.add_argument('--relu',default=False,action='store_true')
parser.add_argument('--no_row_skip',default=False,action='store_true')
args = parser.parse_args()
GY = args.Gy
FUSE_END = args.fuse
RELU = args.relu
print(FUSE_END)
TSB_MULT = args.Tsb
A_dim = args.A_dim
B_dim = args.B_dim
C_dim = args.C_dim
THREADS = args.threads
AT = args.AT
C_blocks = args.C_blocks

input_file = args.infile
outfile = args.outfile
outfile_asm = args.outfile_asm
#assert C_dim % C_blocks == 0
GSY = C_dim // C_blocks
TSB =int( GSY * TSB_MULT)
TSZ = C_dim // C_blocks if C_dim % C_blocks == 0 else C_dim // C_blocks + 1

X86 = args.x86
ARM = args.arm
NRS = args.no_row_skip
CT = args.CT
B_blocks = args.B_blocks
BLOCK = AT

assert not (X86 and ARM)
if X86:
    print("Generating X86 vector intrinsics")
elif ARM:
    print("Generating Arm intrinsics")
else:
    assert False

VEC=16

IN_FORMAT = args.in_format
OUT_FORMAT = args.out_format

if IN_FORMAT == "NHWC" or OUT_FORMAT == "NHWC":
    assert False

input_file_bias = args.infile_bias
if input_file_bias:
    bias = np.load(input_file_bias)

#global AB_vals
AB_vals = []
A_idx = []
B_idx = []
AB_block_offs = [0]
#global off
off = 0

if X86:
    LOAD_CACHE_ASM = """
        vmovdqu8    IDX1(%r8,%r11,1), %xmmNUM;                  
        vmovdqu8     IDX3(%r8,%r11,1), %xmmDA;                 
        vbroadcasti32x4  IDX2(%r8,%r11,1),%ymmNUM {%k1};        
        vbroadcasti32x4  IDX4(%r8,%r11,1), %ymmDA {%k1};       
        vpermt2d    %zmmDA, %zmm25, %zmmNUM;          
        vpshufb     %zmm26, %zmmNUM, %zmmNUM;         
        """
elif ARM:
    LOAD_CACHE = """
    RC = vld1q_f32(&BC[IDX + C_offset + lane]);
    """


if X86:

    LOAD_WEIGHT_ASM = """vpbroadcastd OFF(%rcx), %zmmIDX;
    """
    MAIN_PROGRAM_ASM="""vpdpbusd %zmmNUM,%zmmIDX,%zmmTAR;
    """

elif ARM:
    MAIN_PROGRAM ="""
    val = vdupq_n_f32(VAL);
    ACC[IDX_1] = vmlaq_f32(ACC[IDX_1], RC, val);
    """

def emit_load_block(index, currloadreg):

    return LOAD_CACHE_ASM.replace("IDX1",str(index[0])).replace("IDX2",str(index[1])). \
        replace("IDX3",str(index[2])).replace("IDX4",str(index[3])).replace("NUM",str(currloadreg)).replace("DA",str(17))

def emit_compute_block(Ny_idx,vals,currloadreg, virg=False):
    global off
    
    new_block_asm = ""
    for i in range(BLOCK):
        new_block_asm += LOAD_WEIGHT_ASM.replace("OFF",str(off * 4 + i * 4)).replace("IDX",str(31-i))
    
    for i in range(CT):
        for j in range(BLOCK):
            new_block_asm += MAIN_PROGRAM_ASM.replace("NUM",str(currloadreg - i)).replace("IDX",str(31-j)).replace("TAR",str(i * BLOCK + j))
    global AB_vals
    AB_vals.extend(vals)
    global A_idx
    A_idx.extend([Ny_idx] * 4)
    off += BLOCK
    return new_block_asm


def ny_to_a(ny_idx,groupId,blockId, A_dim = None, A_offset = None):
    if A_offset is None:
        A_offset = blockId * (AT)
    return A_offset + ny_idx


def generate_from_B(Ny_indices, B_indices,BA,block,NY,BB_offset, GY = None,A_offset=None):

    program = ""

    asm = """
    ..B1.NUM1:
    xorl    %r10d, %r10d;
    ..B1.NUM2:
    imul      $16,  %r10d, %r11d;
    add       %r9d, %r11d;
    movslq  %r11d, %r11;
    add     $CT,    %r10d;
    
    """.replace("NUM1",str(BB_offset + block*2+2)).replace("NUM2",str(BB_offset + block * 2 + 3)).replace("STRIDE",str(8)).replace("CT",str(CT))


    #print(A_offset)
    if input_file_bias is not None:
        for i in range(NY):
            for j in range(CT):
                if BB_offset > 0:
                    asm += "\t\tvmovdqu32 " + str(mapping[A_offset + i] * C_dim * 4 + j * VEC * 4) + "(%rdx,%r11,4) ,%zmm" + str(i + j * AT) + ";\n"
                else:
                    asm += "\tvpbroadcastd " + str(mapping[A_offset+i] * 4) + "(%rsi), %zmm" + str(i + AT * j) + ";\n"

    else:
        for i in range(NY):
            for j in range(CT):
                if BB_offset > 0:
                    asm += "\t\tvmovups " + str(mapping[A_offset + i] * C_dim * 4 + j * VEC * 4) + "(%rdx,%r11,4), %zmm" + str(i + j * AT) + ";\n"
                else:
                    asm += "\tvxorps " + "%zmm"  + str(i + AT * j) + ",%zmm" + str(i + AT * j) + ",%zmm" + str(i + AT * j) + ";\n"

    done = set()
    loads = ""
    computes = ""

    TOK = 24
    currloadreg = TOK
    # pad end of the zipped list

    padded_Ny_indices = []
    padded_B_indices = []

    counter = 0
    old_B_idx = -1
    for ny_idx, b_idx in zip(Ny_indices[0],B_indices[0]):
        #assert ny_idx == 0 # for now. We are going to handle AT for quantized at a later date if at all.
        if ny_idx != 0 :
            continue # we are going to just process the first element in each A tile. 
        #print(ny_idx, b_idx)
        padded_Ny_indices.append(ny_idx)
        padded_B_indices.append(b_idx)
        counter += 1
    pad_len = ((counter - 1) // 4 + 1 ) * 4 - counter
    padded_Ny_indices.extend([-1] * pad_len)
    padded_B_indices.extend([-1] * pad_len)
    #print(padded_B_indices,len(padded_B_indices))
    #for ny_idx, b_idx in zip(padded_Ny_indices[0],padded_B_indices[0]):
    for pos in range(0,len(padded_Ny_indices),4):
        b_indices = padded_B_indices[pos:pos+4]
        currloadreg = TOK #(currloadreg - TOK + 1) % 6 + TOK
        asm += loads
        asm += computes
        loads = ""
        computes = ""
        ny_idx = 0
        a_idx = ny_to_a(ny_idx,0,block,A_dim = A_dim, A_offset=A_offset)
        global B_idx

        if -1 in b_indices:
            assert(b_indices[0] != -1)
            for i in range(CT):
                load_block_asm = """
        vxorps      %zmm29, %zmm29, % zmm29;
        vxorps      %zmmNUM, %zmmNUM, %zmmNUM;
        vmovdqu8    IDX1(%r8,%r11,1), %xmmNUM; 
                """.replace("IDX1",str(b_indices[0] * C_dim + i * VEC)).replace("NUM",str(currloadreg-i))
                if b_indices[2] != -1:
                    load_block_asm += """
        vmovdqu8     IDX3(%r8,%r11,1), %xmm29;                  
                    """.replace("IDX3",str(b_indices[2] * C_dim + i * VEC))
                if b_indices[1] != -1:
                    load_block_asm += """
        vbroadcasti32x4  IDX2(%r8,%r11,1),%ymmNUM {%k1};               
                    """.replace("IDX2",str(b_indices[1] * C_dim + i * VEC)).replace("NUM",str(currloadreg-i))
                if b_indices[3] != -1:
                    load_block_asm += """
        vbroadcasti32x4  IDX3(%r8,%r11,1),%ymm29 {%k1};               
                    """.replace("IDX3",str(b_indices[3] * C_dim + i * VEC))
                load_block_asm += """
        vpermt2d    %zmm29, %zmm25, %zmmNUM;          
        vpshufb     %zmm26, %zmmNUM, %zmmNUM; 
                """.replace("NUM",str(currloadreg-i))
                loads += load_block_asm
            #print(b_indices)
            num_vals = np.where(np.array(b_indices) == -1)[0][0]
            #print(num_vals)
            values = []
            for k in range(BLOCK):
                values.append( np.array([BA[b_indices[i],a_idx + k] for i in range(num_vals)] + [0 for j in range(4-num_vals)]).astype(np.int8))
            values = np.hstack(values)
            B_idx.extend(b_indices)
            compute_block_asm = emit_compute_block(ny_idx ,  values, currloadreg , virg = ny_idx not in done)
            computes += compute_block_asm


        else:
            for i in range(CT):
                load_block_asm = emit_load_block([k * C_dim + i * VEC for k in b_indices], currloadreg - i)
                loads += load_block_asm

            values = []
            for i in range(BLOCK):
                values.append(BA[b_indices,a_idx + i])
            values = np.hstack(values)
            
            B_idx.extend(b_indices)
            compute_block_asm = emit_compute_block(ny_idx ,  values, currloadreg , virg = ny_idx not in done)
            computes += compute_block_asm

            done.add(ny_idx)

    asm += loads
    asm += computes





    #print(block,group)
    #program += GROUP_CONTROL_END + "\n"
    global AB_block_offs
    AB_block_offs.append(len(AB_vals))

    return program, asm, done


def get_idx_balanced(block,BA,A_offset,block_NY,B_bounds = [0,B_dim], GY=None):
    #print(block_NY)
    BA = BA[B_bounds[0]:B_bounds[1]]
    Ny_indices = [[] for i in range(GY)]
    B_indices = [[] for i in range(GY)]
    nnz = np.sum(np.abs(BA[:,A_offset:A_offset + block_NY]) > EPS )
    nnz_per_group = nnz // GY
    curr_group = 0
    curr_nnz = 0
    for B_idx in range(B_dim // B_blocks):
        for ny in range(block_NY):
            assert curr_group < GY
            A_idx = ny_to_a(ny,curr_group,block,A_dim = A_dim, A_offset=A_offset)
            if np.abs(BA[B_idx,A_idx]) > EPS:
                B_indices[curr_group].append(B_idx + B_bounds[0])
                Ny_indices[curr_group].append(ny)
                curr_nnz += 1
            if curr_nnz > nnz_per_group:
                curr_group += 1
                curr_nnz = 0

    return Ny_indices, B_indices

def no_load_balance(BA):

    #assert A_dim % A_blocks == 0
    interval = AT

    bounds = [interval * i for i in range(A_blocks)] + [A_dim]

    return bounds , interval

def load_balancer2(BA):

    total_nnz = (np.abs(BA) > EPS).sum()
    nnz_per_block = total_nnz / A_blocks
    sums = np.sum(np.abs(BA) > EPS, axis = 0)
    cs = np.cumsum(sums)
    bounds = [np.argmax(cs > nnz_per_block * i) for i in range(A_blocks)]
    bounds = bounds + [A_dim]
    nnzs = np.diff(bounds)
    NY = np.max(nnzs)
    return bounds, NY


# name is the name of the numpy file
def gencode(BA,outfile,C_dim,A_blocks,C_blocks,GY,name=None):
    program = ""
    asm_program = """
# -- Begin  _spmm
        .text
# mark_begin;
       .align    16,0x90
        .globl _spmm
# --- mm(void *)
_spmm:
# parameter 1: %rdi
..B1.1:                         # Preds ..B1.0
                                # Execution count [9.00e-01]
        .cfi_startproc
..___tag_value__spmm.1:
..L2:
                                                          #45.1
        pushq     %rbp                                          #45.1
        .cfi_def_cfa_offset 16
        movq      %rsp, %rbp                                    #45.1
        .cfi_def_cfa 6, 16
        .cfi_offset 6, -16
        andq      $-32, %rsp                                    #45.1
        subq      $96, %rsp                                     #45.1
        mov         $0xf0 , %ebx;               
        kmovb       %ebx, %k1
        movq      (%rdi), %rcx                                  #47.38
        movq      8(%rdi), %rsi                                 #48.46
        movq      16(%rdi), %r8                                 #49.41
        movq      24(%rdi), %rdx                                #50.22
        movq      32(%rdi), %rbx
        movl      44(%rdi), %eax
        movl      40(%rdi), %edi                                #51.21
        decl    %eax
        decl    %edi
        imul     $TSZ, %eax, %r9d
        
        vpmovzxbd   vpermt2d_control(%rip), % zmm25;
        vbroadcasti32x4   vpshufb_control(%rip), % zmm26;
        



    """.replace("BOUND",str(C_blocks//THREADS)).replace("TSZ",str(TSZ))

    #assert A_dim % A_blocks == 0
    #assert C_dim % C_blocks == 0
    B_dim = BA.shape[0]

    # if IN_FORMAT == "NCHW" and OUT_FORMAT == "NCHW":
    #     bounds, NY = load_balancer2(BA)
    # else:
    bounds, NY = no_load_balance(BA)

    program += START_NONFUSED.replace("OUTPUT_FORMAT",OUT_FORMAT).replace("INPUT_FORMAT",IN_FORMAT).replace("Ny",str(NY)).replace("GY",str(GY)).replace("A_dim",str(A_dim)).replace(
        "C_dim",str(C_dim)).replace("B_dim",str(B_dim)).replace("A_BLOCKS",str(A_blocks)).replace("C_BLOCKS",str(C_blocks)).replace("BOUND",str(C_blocks//4)).replace("X86_DEF",str(int(X86))).replace("ARM_DEF",str(int(ARM))) + "\n"

    assert B_dim % B_blocks == 0
    block_size = B_dim // B_blocks
    for b_block in range(B_blocks):
        bb_offset = b_block * A_blocks * 2
        for block in range(A_blocks):
            A_offset = bounds[block]
            block_NY = bounds[block+1] - A_offset
            program += BLOCK_CONTROL_START.replace("BLOCK", str(block)).replace("Ny",str(block_NY)) + "\n"


            Ny_indices, B_indices = get_idx_balanced(block,BA,A_offset,block_NY,B_bounds = [b_block * block_size, (b_block + 1) * block_size],GY=GY)
            #import pdb;pdb.set_trace()
            ccode, asm, done = generate_from_B(Ny_indices,B_indices,BA,block,block_NY,bb_offset, GY=GY,A_offset=A_offset)
            #ccode = generate_c_stem(block_NY)

            program += textwrap.indent(ccode,"\t") + "\n"
            asm_program += textwrap.indent(asm,"\t") + "\n"
               
            if b_block == B_blocks - 1:
                for i in range(block_NY):
                    asm_program += "\t\tvbroadcastss " + str(mapping[A_offset + i] * 4) + "(%rbx), %zmm20;\n"
                    for j in range(CT):
                        if RELU:
                            asm_program += "\t\tvmaxsb %zmm" + str(i + j * AT) + ", %zmm27, %zmm" + str(i + j * AT) + ";\n"
                        asm_program += "\t\tvcvtdq2ps {rn-sae}, %zmm" + str(i + j * AT) + ",%zmm" + str(i + j * AT) + ";\n"
                        asm_program += "\t\tvmulps %zmm" + str(i + j * AT) + ",%zmm20, %zmm" + str(i + j * AT) + ";\n"
                        asm_program += "\t\tvcvtps2udq {rn-sae}, %zmm" + str(i + j * AT) + ",%zmm" + str(i + j * AT) + ";\n"
                        asm_program += "\t\tvpmovusdb %zmm" + str(i + j * AT) + ",%xmm" + str(i + j * AT) + ";\n"

                    asm_program += """
                    vinserti32x4 $1,%xmmONE,%zmmZERO,%zmmZERO;
                    vinserti32x4 $2,%xmmTWO,%zmmZERO,%zmmZERO;
                    vinserti32x4 $3,%xmmTHREE,%zmmZERO,%zmmZERO;
                    """.replace("ZERO",str(i)).replace("ONE",str(i + AT)).replace("TWO",str(i + 2 * AT)).replace("THREE",str(i + 3 * AT))
                    asm_program += "vmovdqu32 %zmm" + str(i) + ", " + str(mapping[A_offset + i] * C_dim ) + "(%rdx,%r11,1);\n"
            else:
                for i in range(block_NY):
                    for j in range(CT):
                        #print(str(mapping[A_offset + i] * C_dim * 4 + j * VEC * 4))
                        
                        asm_program += "\t\tvmovdqu32 %zmm" + str(i + j * AT) + ", " + str(mapping[A_offset + i] * C_dim * 4 + j * VEC * 4) + "(%rdx,%r11,4);\n"


            asm_program += """
            cmpl      $END, %r10d;
            jb  ..B1.NUM;
            """.replace("NUM",str(bb_offset + block * 2 + 3)).replace("END",str(TSZ // VEC))
            

    program += END_NONFUSED.replace("AB_sparse_tidy.npy",name)
    open(outfile,"w").write(program.replace("B_dim",str(B_dim)))
    asm_program += """
    ..B1.NUM1:                        # Preds ..B1.17
                                # Execution count [2.80e+01]
        decl      %eax                                           #44.37
        subl      $TSZ, %r9d                                    #44.37
        cmpl      %eax, %edi                                      #44.33
        jl        ..B1.2        # Prob 96%                      #44.33
                                # LOE rcx rbx rbp rsi rdi r12 r13 r14 r15 eax dl ymm15
..B1.NUM2:                        # Preds ..B1.18
                                # Execution count [1.00e+00]
        vzeroupper                                              #2398.1
        movq %rbp, %rsp
        popq    %rbp        
        #call      pthread_exit@PLT                              #2416.1
        ret
..___tag_value__spmm.13:
        .align    16,0x90
                                # LOE
        .cfi_endproc
# mark_end;
        .type   _spmm,@function
        #.size   _spmm,-_spmm
        ..LN_spmm.0:
        .section .rodata
        .balign 32
        vpermt2d_control: .byte 0,4,16,20, 1,5,17,21, 2, 6, 18, 22,3,7,19,23 
        vpshufb_control:  .byte 0,4,8,12,  1,5,9,13, 2,6,10,14, 3,7,11,15  
# -- End  _spmm



    """.replace("TSZ",str(TSZ)).replace("CBLOCKS",str(C_blocks)).replace("NUM1",str(B_blocks * A_blocks *2 + 2)).replace("NUM2",str(B_blocks * A_blocks * 2 + 3))

    open(outfile_asm,"w").write(asm_program)



BA = np.load(input_file)
print(BA.shape)
BA = BA.squeeze()

"""
We are going to redo BA here to remove some empty rows
"""
if NRS:
    A_dim = BA.shape[1]
    mapping = {i:i for i in range(A_dim)}
else:
    nnz_cols = np.unique(np.where(BA)[1])
    mapping = {i : nnz_cols[i] for i in range(len(nnz_cols))}
    #print(mapping)
    BA = BA[:,nnz_cols]
    A_dim = len(nnz_cols)
if A_dim % AT == 0:
    A_blocks = A_dim // AT
else:
    A_blocks = A_dim // AT + 1


print("Reduced A dimension " + str(A_dim))
gencode(BA,outfile,C_dim,A_blocks,C_blocks,GY,name=input_file)
np.save("AB_vals.npy",np.array(AB_vals))
np.save("AB_block_off.npy",np.array(AB_block_offs).astype(np.int32))
np.save("A_idx.npy",np.array(A_idx).astype(np.int32))
np.save("B_idx.npy",np.array(B_idx).astype(np.int32))

if input_file_bias:
    np.save("bias.npy",bias.squeeze())
