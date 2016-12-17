/*
 * Alexandre Maros - 2016
 *
 * Cuda Matrix Multiplication with Shared Memory.
 *
 * nvcc cuda_matrix_shared.cu -o cs.o
 *
 * Implemented by Alexandre Maros for learning purposes.
 * A version of this code using Global Memory is in here:
 * https://github.com/alepmaros/cuda_matrix_multiplication
 *
 * Distributed under the MIT Lincese.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// 32x32 Threads in a block.
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

__global__ void matrix_mul(int *a, int *b, int *c, int a_ncolumns, int c_nlines,
        int c_ncolumns, int nBlocks)
{
    int i, z, sum = 0;

    /* How many multiplications there will be for each value in Matrix C
     * This corresponds to the number of columns in Matrix A (or number of)
     * lines in Matrix B
     */
    int nMultiplications = a_ncolumns;

    /* Each iteration of the block will multiply NTHREADS_Y values. This value
     * Can be less then NTHREADS_Y if the number of a_ncolumns is not divisible
     * by NTHREADS_Y. This value is used to control that.
     */
    int multiplicationsInBlock = NTHREADS_Y;

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int line =  blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int s_a[NTHREADS_Y][NTHREADS_X];
    __shared__ int s_b[NTHREADS_Y][NTHREADS_X];

    /* temporary line and temporary column
     * Each thread is responsible for loading one value in the matrix A and
     * Matrix B. These variables are used to hold which line and column of the
     * original Matrices they are suppose to load. I also need to check if those
     * values that they will load actually correspond to a valid position in the
     * original Matrix.
     */
    int a_tLine, a_tColumn, b_tLine, b_tColumn;

    for (z = 0; z < nBlocks; z++) {

        // Load Matrix A
        a_tLine = (blockIdx.y * NTHREADS_Y + threadIdx.y);
        a_tColumn = (z * NTHREADS_X + threadIdx.x);
        if (a_tLine < c_nlines && a_tColumn < a_ncolumns) {
            s_a[threadIdx.y][threadIdx.x] = a[ (a_ncolumns * a_tLine) + a_tColumn];
        }

        // Load Matrix B
        b_tLine = (z * NTHREADS_Y + threadIdx.y);
        b_tColumn = (blockIdx.x * NTHREADS_X + threadIdx.x);
        if (b_tLine < a_ncolumns && b_tColumn < c_ncolumns) {
            s_b[threadIdx.y][threadIdx.x] = b[ (c_ncolumns * b_tLine) + b_tColumn ];
        }

        __syncthreads();

        /* Checkin to see if that thread actually belongs to a valid position in
         * the Matrix C
         */
        if (column < c_ncolumns && line < c_nlines) {
            if (nMultiplications < NTHREADS_Y) {
                multiplicationsInBlock = nMultiplications;
            }

            for (i = 0; i < multiplicationsInBlock; i++) {
                sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
            }

            nMultiplications -= NTHREADS_Y;
        }

        __syncthreads();
    }

    /* Checkin to see if that thread actually belongs to a valid position in
     * the Matrix C
     */
    if (column < c_ncolumns && line < c_nlines) {
        c[line * c_ncolumns + column] = sum;
    }
}

int main(){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int a_nlines, a_ncolumns;
    int b_nlines, b_ncolumns;
    int c_nlines, c_ncolumns;

    size_t a_size, b_size, c_size;
    int i, j;

    struct timeval timevalA;
    struct timeval timevalB;

    scanf("%d", &a_nlines);
    scanf("%d", &a_ncolumns);
    scanf("%d", &b_nlines);
    scanf("%d", &b_ncolumns);

    c_nlines = a_nlines;
    c_ncolumns = b_ncolumns;

#ifdef __DEBUG
    printf("a_nlines: %d\na_ncolumns: %d\nb_nlines: %d\nb_ncolumns: %d\nc_nlines: %d\nc_ncolumns: %d\n", a_nlines, a_ncolumns, b_nlines, b_ncolumns, c_nlines, c_ncolumns);
#endif

    if ( a_ncolumns != b_nlines ) {
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

    for (i = 0; i < a_nlines; i++) {
        for (j = 0; j < a_ncolumns; j++) {
            scanf("%d", &a[i * a_ncolumns + j]);
        }
    }

    for (i = 0; i < b_nlines; i++) {
        for (j = 0; j < b_ncolumns; j++) {
            scanf("%d", &b[i * b_ncolumns + j]);
        }
    }

    gettimeofday(&timevalA,NULL);

    gpuErrchk( cudaMemcpy(d_a, a, a_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_b, b, b_size, cudaMemcpyHostToDevice) );

    dim3 tbloco = dim3(
                    (int) std::ceil( (double) c_ncolumns / NTHREADS_X ),
                    (int) std::ceil( (double) c_nlines / NTHREADS_Y ),
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

    // kernel call
    matrix_mul<<<tbloco,tthreads>>>(d_a, d_b, d_c, a_ncolumns, c_nlines,
        c_ncolumns, (int) std::ceil( (double) a_ncolumns / NTHREADS_X));
    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk( cudaMemcpy(c, d_c, c_size, cudaMemcpyDeviceToHost) );

    gettimeofday(&timevalB,NULL);

#ifndef __DEBUG
    // print Matrix
    for (i = 0; i < c_nlines; i++) {
        for (j = 0; j < c_ncolumns; j++) {
            printf("%d ", c[i * c_ncolumns + j]);
        }
        printf("\n");
    }
    printf("\n");
#endif

#ifdef __TIME
    printf("%.5lf\n", timevalB.tv_sec-timevalA.tv_sec+(timevalB.tv_usec-timevalA.tv_usec)/(double)1000000);
#endif

    free(a); free(b); free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
