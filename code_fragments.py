START_NONFUSED="""
#include <cnpy.h>
#include "mkl.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <x86intrin.h>
// we are doing AC = AB * BC, reduce across the B dimension
// binding B to the x dimension, A to the y dimension and C to the z dimension

#define Tsy 1
#define Tsz (C_dim / C_BLOCKS)
#define ST 1
#define Fx 1
#define Fy (Tsz/Fx)

//#define 64 (64 / 1 / Tsy)

#define Usy (Tsy * Fy)
#define Gsy Usy

#define Gy 1
#define Block_size (Gy * Gsy)
#define X86 X86_DEF
#define ARM ARM_DEF
#include <pthread.h>
#include <cstdlib>


struct thread_data {
        const float * __restrict__ AB_val;
        const float * __restrict__ AB_bias;
        const float * __restrict__ BC;
        float * AC;
        int start;
        int end;
};

void * mm(void * threadarg)
{
        struct thread_data *my_data = (struct thread_data * ) threadarg;
        const float * __restrict__ AB_val = my_data->AB_val;
        const float * __restrict__ AB_bias = my_data->AB_bias;
        const float * __restrict__ BC = my_data->BC;
        float * AC = my_data->AC;
        int start = my_data->start;
        int end = my_data->end;

#if X86
    __m256 ACC[Ny];
	__m256 RC, val;
#elif ARM
    float32x4_t ACC[Ny];
    float32x4_t RC, val;
#endif
    __m256 zero = _mm256_setzero_ps();
   // #pragma omp parallel for schedule(static) private(ACC,RC,val,zero)

	for(int C_block = start; C_block < end; C_block ++){

	int C_offset = C_block * (C_dim / C_BLOCKS);
	
	
"""



BLOCK_CONTROL_START= """
#if X86
    for(int j=0; j < Ny; j++)
    {
            ACC[j] = _mm256_setzero_ps();
    }

	#pragma vector aligned	
	for(int lane =0; lane < Tsz; lane += 8){
#elif ARM
    for(int j=0; j < Ny; j++)
    {
            ACC[j] = vdupq_n_f32(0.0f);
    }

	for(int lane =0; lane < Tsz; lane += 4){
#endif
"""

BLOCK_END_REDUCTION="""
#if X86

   _mm256_store_ps(&AC[OFFSET + C_offset + lane],_mm256_max_ps(zero,_mm256_add_ps(ACC[IDX] , _mm256_broadcast_ss(AB_bias + BIAS))));
   ACC[IDX] = _mm256_setzero_ps();


#elif ARM

    vst1q_f32(&AC[OFFSET + C_offset + lane], ACC[IDX]); 
   ACC[IDX] = vdupq_n_f32(0.0f);

#endif
"""

BLOCK_END = """
#if X86
#pragma vector aligned
for(int i =0; i < Ny; i++)
{
   _mm256_store_ps(&AC[(A_offset + i) * C_dim + C_offset + lane],ACC[i]);
   ACC[i] = _mm256_setzero_ps();
}
}
#elif ARM
for(int i =0; i < Ny; i++)
{
    vst1q_f32(&AC[(A_offset + i) * C_dim + C_offset + lane], ACC[i]); 
   ACC[i] = vdupq_n_f32(0.0f);
}
}
#endif


"""

END_NONFUSED = """

}
//pthread_exit(NULL);
}

"""
