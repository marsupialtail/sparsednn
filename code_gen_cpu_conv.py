# this program basically does a constexpr and generates cuda code
import textwrap
import numpy as np
from code_fragments import *
from utils import *


import argparse
parser = argparse.ArgumentParser(description='CodeGen V1')

parser.add_argument('--OC', type=int, default=12)
parser.add_argument('--IC', type=int, default=12)
parser.add_argument('--image_x', type=int, default=12)
parser.add_argument('--image_y', type=int, default=12)
parser.add_argument('--B_blocks', type=int, default=1)
parser.add_argument('--C_blocks', type=int, default=1)
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
parser.add_argument('--avx512',default=False,action='store_true')
parser.add_argument('--no_relu',default=False,action='store_true')
parser.add_argument('--no_row_skip',default=False,action='store_true')
args = parser.parse_args()
FILTER_X = 3
FILTER_Y = 3
GY = args.Gy
FUSE_END = args.fuse
NO_RELU = args.no_relu
print(FUSE_END)
TSB_MULT = args.Tsb
A_dim = args.OC
OC = args.OC
B_dim = args.IC * FILTER_X * FILTER_Y
IC = args.IC
IMAGE_X = args.image_x
IMAGE_Y = args.image_y
AVX512 = False
if args.avx512:
    AVX512 = True

if AVX512:
    VEC = 16
else:
    VEC = 8

if IMAGE_Y == 56:
    if AVX512:
        Y_PAD = 8
    else:
        Y_PAD = 2
elif IMAGE_Y == 28:
    Y_PAD = 4
elif IMAGE_Y == 14:
    Y_PAD = 2
elif IMAGE_Y == 7:
    Y_PAD = 1

if IMAGE_Y % VEC != 0:
    C_dim = (IMAGE_X ) * (IMAGE_Y + Y_PAD)
else:
    C_dim = (IMAGE_X) * IMAGE_Y

THREADS = args.threads
C_blocks = args.C_blocks
B_blocks = args.B_blocks

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

if AVX512:
    if IMAGE_Y == 56:
        AT = 6
        CT = 4
    elif IMAGE_Y == 28:
        AT = 12
        CT = 2
    elif IMAGE_Y == 14:
        AT = 24
        CT = 1
    elif IMAGE_Y == 7:
        print("Not supported")
    else:
        print("Not supported")
        exit()
else:
    if IMAGE_Y == 56:
        AT = 1
        CT = 7
    elif IMAGE_Y == 28:
        AT = 2
        CT = 4
    elif IMAGE_Y == 14:
        AT = 6
        CT = 2
    elif IMAGE_Y == 7:
        AT = 12
        CT = 1
    else:
        print("Not supported")
        exit()


assert not (X86 and ARM)
if X86:
    print("Generating X86 vector intrinsics")
elif ARM:
    print("Generating Arm intrinsics")
else:
    assert False

IN_FORMAT = args.in_format
OUT_FORMAT = args.out_format

if IN_FORMAT == "NHWC" or OUT_FORMAT == "NHWC":
    assert False

input_file_bias = args.infile_bias
if input_file_bias:
    bias = np.load(input_file_bias)

#global AB_vals
AB_vals = []
#global off
off = 0

if X86:
    if AVX512:
        LOAD_CACHE = """
        RC = _mm512_load_ps(&BC[IDX + C_offset + lane ]);
        
        //RC = _mm256_load_ps(&BC[(C_offset + lane) * B_dim + IDX]);
        """
        LOAD_CACHE_ASM = """vmovups IDX(%r8,%r11,4), %zmmNUM;
        """
    else:
        LOAD_CACHE = """
        RC = _mm256_load_ps(&BC[IDX + C_offset + lane + lane / 28 * 2]);
        
        //RC = _mm256_load_ps(&BC[(C_offset + lane) * B_dim + IDX]);
        """
        LOAD_CACHE_ASM = """vmovups IDX(%r8,%r11,4), %ymmNUM;
        """
elif ARM:
    LOAD_CACHE = """
    RC = vld1q_f32(&BC[IDX + C_offset + lane]);
    """


if X86:
    if AVX512:
        LOAD_WEIGHT="""
        val = _mm512_broadcast_ss(AB_val + OFF);
        """
        MAIN_PROGRAM ="""
        //val = _mm256_set1_ps(VAL);
        ACC[IDX_1] = _mm512_fmadd_ps(RC, val, ACC[IDX_1]);
        """
        LOAD_WEIGHT_ASM = """vbroadcastss OFF(%rcx), %zmm31;
        """
        MAIN_PROGRAM_ASM="""vfmadd231ps %zmmNUM, %zmm31, %zmmIDX_1;
        """

    else:
        LOAD_WEIGHT = """
        val = _mm256_broadcast_ss(AB_val + OFF);
        """
        MAIN_PROGRAM ="""
        ACC[IDX_1] = _mm256_fmadd_ps(RC, val, ACC[IDX_1]);
        """
        LOAD_WEIGHT_ASM = """vbroadcastss OFF(%rcx), %ymm15;
        """
        MAIN_PROGRAM_ASM="""vfmadd231ps %ymmNUM, %ymm15, %ymmIDX_1;
        """
        MAIN_PROGRAM_ASM_VIRG="""vbroadcastss OFF(%rcx), %ymm15;
    vmul231ps %ymmNUM, %ymm15, %ymmIDX_1;
        """

elif ARM:
    MAIN_PROGRAM ="""
    val = vdupq_n_f32(VAL);
    ACC[IDX_1] = vmlaq_f32(ACC[IDX_1], RC, val);
    """


if IN_FORMAT == "NCHW":
    def emit_load_block(index, currloadreg):
        new_block = LOAD_CACHE.replace("IDX",str(index))
        new_block_asm = LOAD_CACHE_ASM.replace("IDX",str(index * 4)).replace("NUM",str(currloadreg))
        #new_block = LOAD_CACHE.replace("IDX",str(B_idx * 8))
        return new_block, new_block_asm
else:
    def emit_load_block(B_idx,B_offset):
        new_block = LOAD_CACHE.replace("IDX",str(B_idx - B_offset))
        return new_block

def emit_load_smem_block(local_TSB, tile_id):
    return LOAD_SHARED.replace("TSB",str(local_TSB)).replace("TILE",str(tile_id * TSB))

def emit_compute_block(Ny_idx,val,currloadreg, virg=False):
    global off
    new_block = LOAD_WEIGHT.replace("OFF",str(off ))
    new_block_asm = LOAD_WEIGHT_ASM.replace("OFF",str(off * 4 ))
    for i in range(CT):
        new_block += MAIN_PROGRAM.replace("IDX_1",str(Ny_idx+ i * AT))
        new_block_asm += MAIN_PROGRAM_ASM.replace("IDX_1",str(Ny_idx +i * AT)).replace("NUM",str(currloadreg - i))
    global AB_vals
    AB_vals.append(val)
    off += 1
    return new_block, new_block_asm


def ny_to_a(ny_idx,groupId,blockId, A_dim = None, A_offset = None):
    if A_offset is None:
        A_offset = blockId * (AT)
    return A_offset + ny_idx


def generate_from_B(Ny_indices, B_indices,BA,block,NY,BB_offset, GY = None,A_offset=None):

    program = ""
    asm = ""

    assert GY == 1
    for group in range(GY):
        #program += GROUP_CONTROL_START.replace("GROUP",str(group)) + "\n"

        next_tile_start = 0
        old_b_idx = -1

        if AVX512:
            asm += """
        ..B1.NUM1:
        xorl    %r10d, %r10d;
        #xorl    %r13d, %r13d;
        ..B1.NUM2:
        imul      $16,  %r10d, %r11d;
        add       %r9d, %r11d;
        movslq  %r11d, %r11;
        #add     $CT,    %r10d;
        """.replace("NUM1",str(BB_offset + block*2+2)).replace("NUM2",str(BB_offset + block * 2 + 3))
        else:
            asm += """
        ..B1.NUM1:
        #xorl    %r13d, %r13d;
        xorl    %r10d, %r10d;
        ..B1.NUM2:
        lea       (%r9,%r10,8), %r11d;
        movslq  %r11d, %r11;
        """.replace("NUM1",str(BB_offset + block*2 + 2)).replace("NUM2",str(BB_offset + block * 2 + 3)).replace("CT",str(CT))

        if not AVX512 and IMAGE_Y == 56:
            asm += """
        mov     %r10d, %r13d;
        mov     %r10d, %r14d;
        imul    $613566757, %r13, %r13;
        shr     $32, %r13;
        sub     %r13d, %r14d;
        shr     %r14d;
        add     %r14d, %r13d;
        shr     $2, %r13d;
        shl     $1, %r13d;
        add     %r13d, %r11d;
            """

        #print(A_offset)

        if AVX512:
            if IMAGE_Y == 28 or IMAGE_Y == 14 or IMAGE_Y == 56:
                asm += """sub         %r10d, %r11d;
                            sub         %r10d, %r11d;
        """
        else:
            if IMAGE_Y == 28 or IMAGE_Y == 14 or IMAGE_Y == 7:
                asm += """sub         %r10d, %r11d;
        """
        for i in range(NY):
            for j in range(CT):
                if BB_offset > 0:
                    asm += "\tvmovups " + str(mapping[A_offset + i] * IMAGE_Y * IMAGE_Y * 4 + j * VEC * 4) + "(%rdx,%r11,4)" + "," + "%ymm" + str(i + AT * j) + ";\n"
                else:
                    asm += "\tvbroadcastss " + str(mapping[A_offset+i] * 4) + ("(%rsi), %zmm" if AVX512 else "(%rsi), %ymm") + str(i + AT * j) + ";\n"
        if AVX512:
            if IMAGE_Y == 28 or IMAGE_Y == 14 or IMAGE_Y == 56:
                asm += """add         %r10d, %r11d;
                add         %r10d, %r11d;
            """
        else:

            if IMAGE_Y == 28 or IMAGE_Y == 14 or IMAGE_Y == 7:
                asm += """add         %r10d, %r11d;
            """
        done = set()
        loads = ""
        computes = ""

        if AVX512:
            TOK = 29
        else:
            TOK = 13
        currloadreg = TOK
        for ny_idx, b_idx in zip(Ny_indices[group],B_indices[group]):

            if IN_FORMAT == "NHWC":
                if old_b_idx < next_tile_start and b_idx >= next_tile_start:
                    smem_block = emit_load_smem_block(min(TSB,B_dim - next_tile_start),next_tile_start // TSB)
                    program += textwrap.indent(smem_block,"\t")
                    next_tile_start += TSB

            if b_idx != old_b_idx:
                if IN_FORMAT == "NCHW":
                    currloadreg = TOK #(currloadreg - TOK + 1) % 6 + TOK
                    if currloadreg == TOK:
                        asm += loads
                        asm += computes
                        loads = ""
                        computes = ""

                    channel = b_idx // (FILTER_X * FILTER_Y)
                    x = (b_idx // FILTER_Y) % FILTER_X
                    y = b_idx % FILTER_Y
                    channel_offset = channel * (IMAGE_X+1) * (IMAGE_Y+Y_PAD) - IMAGE_Y -Y_PAD - 1
                    for i in range(CT):
                        load_block_cuda, load_block_asm = emit_load_block(channel_offset + x * (IMAGE_Y+Y_PAD) + y + i * VEC, currloadreg - i)
                        loads += textwrap.indent(load_block_asm,"\t")
                        program += textwrap.indent(load_block_cuda,"\t")
                else:
                    load_block_cuda, load_block_asm = emit_load_block(b_idx,next_tile_start - TSB)



                old_b_idx = b_idx

            a_idx = ny_to_a(ny_idx,group,block,A_dim = A_dim, A_offset=A_offset)
            value = BA[b_idx,a_idx]

            compute_block_cuda, compute_block_asm = emit_compute_block(ny_idx ,  value, currloadreg , virg = ny_idx not in done)
            computes += textwrap.indent(compute_block_asm, "\t")
            program += textwrap.indent(compute_block_cuda, "\t")

            done.add(ny_idx)



        asm += loads
        asm += computes
        #print(block,group)
        #program += GROUP_CONTROL_END + "\n"

    return program, asm, done


def get_idx_balanced(block,BA,A_offset,block_NY,B_bounds = [0,B_dim], GY=None):

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
# -- Begin  _Z2mmPv
        .text
# mark_begin;
       .align    16,0x90
        .globl _Z2mmPv
# --- mm(void *)
_Z2mmPv:
# parameter 1: %rdi
..B1.1:                         # Preds ..B1.0
                                # Execution count [9.00e-01]
        .cfi_startproc
..___tag_value__Z2mmPv.1:
..L2:
                                                          #45.1
        pushq     %rbp                                          #45.1
        .cfi_def_cfa_offset 16
        movq      %rsp, %rbp                                    #45.1
        .cfi_def_cfa 6, 16
        .cfi_offset 6, -16
        andq      $-32, %rsp                                    #45.1
        subq      $96, %rsp                                     #45.1
        movq      (%rdi), %rcx                                  #47.38
        movq      8(%rdi), %rsi                                 #48.46
        movq      16(%rdi), %r8                                 #49.41
        movq      24(%rdi), %rdx                                #50.22
        movl      36(%rdi), %eax
        movl      32(%rdi), %edi                                #51.21
        vxorps    ZERO, ZERO, ZERO                           #59.19
        decl    %eax
        decl    %edi
        imul     $TSZ, %eax, %r9d



    """.replace("BOUND",str(C_blocks//THREADS)).replace("TSZ",str(TSZ)).replace("ZERO","%zmm30" if AVX512 else "%ymm14")

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


            Ny_indices, B_indices = get_idx_balanced(block,BA,A_offset,block_NY,B_bounds=[b_block * block_size, (b_block+1) * block_size ],GY=GY)
            #import pdb;pdb.set_trace()
            ccode, asm, done = generate_from_B(Ny_indices,B_indices,BA,block,block_NY,bb_offset,GY=GY,A_offset=A_offset)
            #ccode = generate_c_stem(block_NY)

            program += textwrap.indent(ccode,"\t") + "\n"
            asm_program += textwrap.indent(asm,"\t") + "\n"
            if OUT_FORMAT == "NCHW":
                if FUSE_END:

                    if AVX512:
                        if IMAGE_Y == 28 or IMAGE_Y == 14 or IMAGE_Y == 56:
                            asm_program += "sub         %r10d, %r11d;\n\tsub         %r10d, %r11d;"
                    else:
                        if IMAGE_Y == 28 or IMAGE_Y == 14 or IMAGE_Y == 7:
                            asm_program += "sub         %r10d, %r11d;"

                    asm_program += """
                add         $CT, %r10d; 
   
                        """.replace("CT",str(CT))
                    if IMAGE_Y % VEC != 0:
                        if AVX512:
                            asm_program += " mov msg(%rip), %ebx;\n kmovw %ebx, %k1;"
                        else:
                            asm_program += " vmovdqu msg(%rip), %ymm13;\n"
                    else:
                        asm_program += " sub     %r13d, %r11d;\n"

                    if GY > 1:
                        print("End fusion strategy not valid.")
                    for i in range(block_NY):
                        program += BLOCK_END_REDUCTION.replace("OFFSET",str(mapping[A_offset + i] * C_dim)).replace("IDX",str(i)).replace("BIAS",str(A_offset+i))

                        if not NO_RELU and b_block == B_blocks - 1:
                            for j in range(CT):
                                asm_program += "\t\tvmaxps %ymm" + str(i + j * AT) + (", %zmm30," if AVX512 else ", %ymm14,") + "%ymm" + str(i + j * AT) + ";\n"
                        for j in range(CT):
                            if j == CT - 1 and IMAGE_Y % VEC != 0:
                                if AVX512:
                                    asm_program += "\t\tvmovups %zmm" + str(i + j * AT) + ", " + str(mapping[A_offset + i] * IMAGE_Y * IMAGE_Y * 4 + j * VEC * 4) + "(%rdx,%r11,4) {%k1} ; \n"
                                else:
                                    asm_program += "\t\tvpmaskmovd %ymm" + str(i + j * AT) + ",%ymm13," + str(mapping[A_offset + i] * IMAGE_Y * IMAGE_Y * 4 + j * VEC * 4) + "(%rdx,%r11,4);\n"
                            else:
                                asm_program += "\t\tvmovups %ymm" + str(i + j * AT) + ", " + str(mapping[A_offset + i] * IMAGE_Y * IMAGE_Y * 4 + j * VEC * 4) + "(%rdx,%r11,4);\n"
                    asm_program += """
                    cmp         $END, %r10d;
                    jb  ..B1.NUM;
                    """.replace("NUM",str(bb_offset + block * 2 + 3)).replace("END",str(TSZ // VEC))

                    program += "\t}"
                else:
                    program += BLOCK_END.replace("A_offset",str(A_offset)).replace("Ny",str(block_NY)).replace("A_BLOCKS",str(A_blocks)).replace(
                "C_BLOCKS", str(C_blocks)).replace("A_dim",str(A_dim)).replace("C_dim",str(C_dim)).replace("B_dim",str(B_dim)) + "\n"
            else:
                program += BLOCK_END_NHWC.replace("A_offset",str(A_offset)).replace("Ny",str(block_NY)).replace("A_BLOCKS",str(A_blocks)).replace(
                    "C_BLOCKS", str(C_blocks)).replace("A_dim",str(A_dim)).replace("C_dim",str(C_dim)).replace("B_dim",str(B_dim)) + "\n"
                # program += BLOCK_CONTROL_END

    program += END_NONFUSED.replace("AB_sparse_tidy.npy",name)
    open(outfile,"w").write(program.replace("B_dim",str(B_dim)))
    if AVX512:
        if IMAGE_Y == 28:
            MSG = ".short 0x0fff"
        elif IMAGE_Y == 14:
            MSG = ".short 0x3fff"
        elif IMAGE_Y == 56:
            MSG = ".short 0x00ff"
        else:
            MSG = ".short 0xffff"

    else:
        if IMAGE_Y == 28:
            MSG = ".int 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000"
        elif IMAGE_Y == 14:
            MSG = ".int 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0x00000000, 0x00000000"
        elif IMAGE_Y == 7:
            MSG = ".int 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0x00000000"
        else:
            MSG = ".int 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000"


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
..___tag_value__Z2mmPv.13:
        .align    16,0x90
                                # LOE
        .cfi_endproc
# mark_end;
        .type   _Z2mmPv,@function
        .size   _Z2mmPv,.-_Z2mmPv
..LN_Z2mmPv.0:
        .section .rodata
        .balign 32
        msg:
MSG

# -- End  _Z2mmPv
    """.replace("MSG",MSG).replace("TSZ",str(TSZ)).replace("CBLOCKS",str(C_blocks)).replace("NUM1",str(A_blocks *2 * B_blocks + 2)).replace("NUM2",str(A_blocks * 2 * B_blocks + 3))

    if AVX512:
        asm_program = asm_program.replace("ymm","zmm")

    open(outfile_asm,"w").write(asm_program)



BA = np.load(input_file)
BA =BA
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
if input_file_bias:
    np.save("AB_bias.npy",bias.squeeze())
