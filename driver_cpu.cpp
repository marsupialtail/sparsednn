#include <cnpy.h>
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
#include <chrono>
//#define 64 (64 / 1 / Tsy)
#include <cstdlib>
#include <omp.h>
using namespace std;

const int COUNT = 4;
const int WORK = 10'000'000;


#define BOUND ((C_blocks - 1 ) / 4 + 1)

#if INT8
struct thread_data {
        const int8_t * __restrict__ AB_val;
        const int * __restrict__ AB_bias;
        const int8_t * __restrict__ BC;
        int8_t * AC;
        const float * scale;
        int start;
        int end;
};
#else
struct thread_data {
        const float * __restrict__ AB_val;
        const float * __restrict__ AB_bias;
        const float * __restrict__ BC;
        float * AC;
        int start;
        int end;
};
#endif
struct thread_data_reps {
    thread_data * arg;
    int reps;
};


typedef uint16_t offset_t;
#define PTR_OFFSET_SZ sizeof(offset_t)
//taken from https://embeddedartistry.com/blog/2017/02/22/generating-aligned-memory/
#ifndef align_up
#define align_up(num, align) \
    (((num) + ((align) - 1)) & ~((align) - 1))
#endif
void * aligned_malloc(size_t align, size_t size)
{
    void * ptr = NULL;
    assert((align & (align - 1)) == 0);
    if(align && size)
    {
        uint32_t hdr_size = PTR_OFFSET_SZ + (align - 1);
        void * p = malloc(size + hdr_size);

        if(p)
        {
            ptr = (void *) align_up(((uintptr_t)p + PTR_OFFSET_SZ), align);

            *((offset_t *)ptr - 1) = 
                (offset_t)((uintptr_t)ptr - (uintptr_t)p);

        } // else NULL, could not malloc
    } //else NULL, invalid arguments

    return ptr;
}

void aligned_free(void * ptr)
{
    assert(ptr);

    offset_t offset = *((offset_t *)ptr - 1);

    void * p = (void *)((uint8_t *)ptr - offset);
    free(p);
}

#define THREADS 4

#include <algorithm>

void fillvector(float *data, int n) {
    for(int i=0; i<n; i++){
        data[i] = float(rand() % 10 - 5);
    }
}

#define SIZE 100000
void clear_cache()
{
        int a[SIZE];
        for(int i = 0; i < SIZE; i ++)
        {
            a[i] = i;
        }
        std::random_shuffle(a, a + SIZE);

        int b = 0;
        for(int i = 0; i < 100; i ++)
        {
            b = a[b];
        }
}



static void* (*mm)(void*);
#if INT8
static int * ref1;
#else
static float * ref1;
#endif


int main()
{
#if INT8
 ref1 =(int* ) aligned_alloc(128, A_dim * C_dim * 4);
#else
 ref1 =(float* ) aligned_alloc(128,A_dim * C_dim * 4);
#endif

	cnpy::NpyArray arr1 = cnpy::npy_load("BC.npy");
#if INT8
    int8_t * BC_unaligned = arr1.data<int8_t>();
	assert(arr1.word_size == 1);
#else
	float * BC_unaligned = arr1.data<float>();
	assert(arr1.word_size == sizeof(float));
#endif
	std::cout << B_dim << " " << C_dim << std::endl;
	assert(arr1.shape.size()==2 && arr1.shape[0] == B_dim && arr1.shape[1] == C_dim);
	
	cnpy::NpyArray arr2 = cnpy::npy_load("AB_vals.npy");
#if INT8
	int8_t * AB_vals = arr2.data<int8_t>();
	assert(arr2.word_size == 1);
#else
	float * AB_vals = arr2.data<float>();
	assert(arr2.word_size == sizeof(float));
#endif
	assert(arr2.shape.size() ==1);
	int nnzs = arr2.shape[0];

	cnpy::NpyArray arr3 = cnpy::npy_load("bias.npy");
#if INT8
	int * AB_bias = arr3.data<int>();
	assert(arr3.word_size == 4);
#else
	float * AB_bias = arr3.data<float>();
	assert(arr3.word_size == sizeof(float));
	//assert(arr3.shape.size() ==1 && arr3.shape[0] == A_dim);
#endif

#if INT8
	cnpy::NpyArray arr4 = cnpy::npy_load("scale.npy");
	float * scale = arr4.data<float>();
#endif 


    cnpy::NpyArray arr7 = cnpy::npy_load("ref.npy");
#if INT8
    int8_t * ref1_stack = arr7.data<int8_t>();
    std::memcpy(ref1,ref1_stack,A_dim * C_dim);
#else
    float * ref1_stack = arr7.data<float>();
    std::memcpy(ref1,ref1_stack,A_dim * C_dim * 4);
#endif

#if X86
#if INT8
    int8_t * BCs = (int8_t*) aligned_alloc(128, B_dim * C_dim);
    std::memcpy(&BCs[0],BC_unaligned,B_dim * C_dim);
#else
    float* BCs = (float*) aligned_alloc(128, B_dim * C_dim * 4);
    std::memcpy(&BCs[0],BC_unaligned,B_dim * C_dim * 4);

#endif
#elif ARM
	float * BC = (float*) aligned_malloc(128, B_dim * C_dim * 4);
#endif
	

#if X86

#if INT8

	// the intermediate results are in int32 so they need more space
	int8_t * result;
	result = (int8_t *)aligned_alloc(128, A_dim * C_dim *4 );
	memset(result,0,A_dim * C_dim *4 );

#else
	float *result;
    result = (float *)aligned_alloc(128,A_dim * C_dim *sizeof(result));
    memset(result,0,A_dim * C_dim * sizeof(result));

//    for(int i = 0; i < A_dim*C_dim; i ++)
//        {
//        result[i] = 1.0f;
//        }

#endif

#elif ARM
    result = (float *) aligned_malloc(128, A_dim * C_dim * sizeof(result));
    memset(result,0,A_dim * C_dim * sizeof(result));
#endif


    // let's pre-write the bias to the result. this is acceptable.
 /*   for(int i = 0; i < A_dim; i ++)
    {
	    for(int j = 0; j < C_dim; j ++)
	    {
		    result[i * C_dim + j] = AB_bias[i];
	    }
    }
*/
    void *handle;

    char *error_str;

     
   using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    handle = dlopen ("./test.so", RTLD_LAZY);
    if (!handle) {
        fputs (dlerror(), stderr);
        exit(1);
    }

    mm =(void* (*)(void *)) dlsym(handle, "_spmm");
    if ((error_str = dlerror()) != NULL)  {
        fputs(error_str, stderr);
        exit(1);
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    printf (" == Load shared library == \n== at %.5f milliseconds == \n ", ms_double.count() );

        //printf (" Load at %.5f milliseconds == \n\n", (s_elapsed * 1000));


  struct thread_data td[THREADS];
  for(int i = 0; i < THREADS; i ++)
  {
	td[i].AB_val = AB_vals;
	td[i].AB_bias= AB_bias;
	td[i].BC = &BCs[0];
	td[i].AC = result; 
    td[i].scale = scale;
#if MULTI
        td[i].start = i * BOUND;
	td[i].end = min(i * BOUND + BOUND, C_blocks);
#else
	td[i].start = 0;//i * BOUND  ;
        td[i].end = C_blocks;//min(i * BOUND + BOUND, C_blocks);
#endif

  }
#if MULTI
    // warm up omp thread pool
	#pragma omp parallel for
        for(int i = 0; i < THREADS; i ++)
	{
          mm(&td[i]);
	}
#endif
    auto issed = 0;
    void * status;
    t1 = high_resolution_clock::now();
#if MULTI
    while(issed < 10 ) {
	#pragma omp parallel for
        for(int i = 0; i < THREADS; i ++)
	{
          mm(&td[i]);
	}
        issed += 1;
    }
#else
    while(issed < 10 ) {
         mm(&td[0]);
        issed += 1;
    }
#endif

    t2 = high_resolution_clock::now();
    ms_double = t2 - t1;	
	//memset(result,0,A_dim * C_dim * sizeof(result));

	int reps = 20000 / ms_double.count();
//    int reps = 20;

    std::cout << reps << std::endl;
    t1 = high_resolution_clock::now();

    issed = 0;
#if MULTI
    while(issed < reps ) {
	#pragma omp parallel for
        for(int i = 0; i < THREADS; i ++)
	{
          mm(&td[i]);
	}
        issed += 1;
    }
#else
    while(issed < reps ) {
         mm(&td[0]);
        issed += 1;
    }
#endif
   t2 = high_resolution_clock::now();
    ms_double = t2 - t1;
    printf (" == spmm microkernel == \n== at %.5f milliseconds == \n == %d reps == ", (ms_double.count() / reps), reps);

    
//    dlclose(handle);
#if INT8
        cnpy::npy_save("cpu_output.npy",(char*)(&result[0]),{A_dim, C_dim},"w");
#else
    cnpy::npy_save("cpu_output.npy",(float*)(&result[0]),{A_dim, C_dim},"w");

#endif

	std::cout << result[0] << result[1] << result[2] << std::endl;
}
