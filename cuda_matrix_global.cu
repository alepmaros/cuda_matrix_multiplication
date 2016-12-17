/*
 * Alexandre Maros - 2016
 *
 * Cuda Matrix Multiplication with Global Memory.
 *
 * nvcc cuda_matrix_global.cu -o cg.o
 *
 * Implemented by Alexandre Maros for learning purposes.
 * A version of this code using Shared Memory is in here:
 * https://github.com/alepmaros/cuda_matrix_multiplication
 *
 * Distributed under the MIT Lincese.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

//32x32
#define NTHREADS_X 32
#define NTHREADS_Y 32
#define THREADS_PER_BLOCK NTHREADS_X * NTHREADS_Y

/* A macro used for error checking in CUDA function calls
 * Credit to: http://stackoverflow.com/a/14038590 for the gpuErrchk macro.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void matrix_mul(int *a, int *b, int *c, int a_ncolumns, int c_nlines, int c_ncolumns)
{

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int line =  blockIdx.y * blockDim.y + threadIdx.y;

    if (column  >= c_ncolumns || line >= c_nlines)
        return;

    int i, sum = 0;


    int beginA = a_ncolumns * line;
    int beginB = column;

    for (i = 0; i < a_ncolumns; i++)
    {
        sum += a[beginA + i] * b[i * c_ncolumns + beginB];
    }

    c[line * c_ncolumns + column] = sum;
}

int main(){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int a_nlines, a_ncolumns;
    int b_nlines, b_ncolumns;
    int c_nlines, c_ncolumns;

    size_t a_size, b_size, c_size;
    int i, j;

    cudaEvent_t start, stop;
    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    scanf("%d", &a_nlines);
    scanf("%d", &a_ncolumns);
    scanf("%d", &b_nlines);
    scanf("%d", &b_ncolumns);

    c_nlines = a_nlines;
    c_ncolumns = b_ncolumns;

#ifdef __DEBUG
    printf("a_nlines: %d\na_ncolumns: %d\nb_nlines: %d\nb_ncolumns: %d\nc_nlines: %d\nc_ncolumns: %d\n", a_nlines, a_ncolumns, b_nlines, b_ncolumns, c_nlines, c_ncolumns);
#endif

    if ( a_ncolumns != b_nlines )
    {
        printf("Number of columns in Matrix A should be equals to number of lines in Matrix B\n");
        return EXIT_FAILURE;
    }

    a_size = a_nlines * a_ncolumns * sizeof(int);
    b_size = b_nlines * b_ncolumns * sizeof(int);
    c_size = c_nlines * c_ncolumns * sizeof(int);

    gpuErrchk( cudaMalloc((void **) &d_a, a_size) );
    gpuErrchk( cudaMalloc((void **) &d_b, b_size) );
    gpuErrchk( cudaMalloc((void **) &d_c, c_size) );

    a = (int *)malloc(a_size);
    b = (int *)malloc(b_size);
    c = (int *)malloc(c_size);

    memset(c, 0, c_nlines*c_ncolumns*sizeof(int));

    for (i = 0; i < a_nlines; i++)
    {
        for (j = 0; j < a_ncolumns; j++)
        {
            scanf("%d", &a[i * a_ncolumns + j]);
        }
    }

    for (i = 0; i < b_nlines; i++)
    {
        for (j = 0; j < b_ncolumns; j++)
        {
            scanf("%d", &b[i * b_ncolumns + j]);
        }
    }

    gpuErrchk( cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice) );

    dim3 tbloco = dim3(
                    (int) std::ceil( (double) c_ncolumns / NTHREADS_X ),
                    (int) std::ceil ( (double) c_nlines / NTHREADS_Y ),
                    1
                );

    dim3 tthreads = dim3(
                        NTHREADS_X,
                        NTHREADS_Y,
                        1
                    );

#ifdef __DEBUG
    printf("tbloco.x: %d tbloco.y: %d tbloco.z: %d\n", tbloco.x, tbloco.y, tbloco.z);
    printf("tthreads.x: %d tthreads.y: %d\n", tthreads.x, tthreads.y);
#endif

    cudaEventRecord(start);

    // kernel call
    matrix_mul<<<tbloco,tthreads>>>(d_a, d_b, d_c, a_ncolumns, c_nlines, c_ncolumns);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaEventRecord(stop) );
    gpuErrchk( cudaMemcpy(c, d_c, c_size, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaEventSynchronize(stop) );


#ifndef __NO_OUTPUT
    // print Matrix
    for (i = 0; i < c_nlines; i++)
    {
        for (j = 0; j < c_ncolumns; j++)
        {
            printf("%d ", c[i * c_ncolumns + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

#ifdef __TIME
        float milliseconds = 0;
        gpuErrchk( cudaEventElapsedTime(&milliseconds, start, stop) );
        printf("%.5f\n", milliseconds);
#endif

    free(a); free(b); free(c);

    gpuErrchk( cudaFree(d_a) );
    gpuErrchk( cudaFree(d_b) );
    gpuErrchk( cudaFree(d_c) );

    return 0;
}
