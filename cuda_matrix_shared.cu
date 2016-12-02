#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define NLINES 6144
#define NCOLUMNS 6144
#define THREADS_PER_BLOCK 1024
//32x32
#define NTHREADS 32
#define NCOLUMNSPERBLOCK 32
#define NLINESPERBLOCK 32

__global__ void vector_mul(int *a, int *b, int *c) {
    int i, z, sum = 0;

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int line =  blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ int s_a[NLINESPERBLOCK][NCOLUMNSPERBLOCK];
    __shared__ int s_b[NLINESPERBLOCK][NCOLUMNSPERBLOCK];

	for (z = 0; z < gridDim.x; z++) {
        s_a[threadIdx.y][threadIdx.x] = a[ (NCOLUMNS * (blockIdx.y * NLINESPERBLOCK + threadIdx.y)) + (z * NCOLUMNSPERBLOCK + threadIdx.x) ];
        s_b[threadIdx.y][threadIdx.x] = b[ (NCOLUMNS * (z * NLINESPERBLOCK + threadIdx.y)) + blockIdx.x * NCOLUMNSPERBLOCK + threadIdx.x ];

        __syncthreads();

		for (i = 0; i < NLINESPERBLOCK; i++) {
	    	sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
	    }

        __syncthreads();
	}

    //printf("%d %d\n", line, column);
    c[line * NLINES + column] = sum;
}

int main(){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = NLINES * NCOLUMNS * sizeof(int);
    int i, j, n;

    struct timeval timevalA;
	struct timeval timevalB;

    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    for(i = 0; i < NLINES*NCOLUMNS; i++){
        c[i] = 0;
    }

    scanf("%d", &n);

    for (int i = 0; i < NLINES; i++) {
        for (j = 0; j < NLINES; j++) {
            scanf("%d", &a[i * NLINES + j]);
        }
    }

    for (int i = 0; i < NLINES; i++) {
        for (j = 0; j < NLINES; j++) {
            scanf("%d", &b[i * NLINES + j]);
        }
    }

    gettimeofday(&timevalA,NULL);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 tbloco = dim3(NLINES/NTHREADS, NLINES/NTHREADS,1);
    dim3 tthreads = dim3(NTHREADS, NTHREADS, 1);
    vector_mul<<<tbloco,tthreads>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    gettimeofday(&timevalB,NULL);

    // print Matrix
    // for (i = 0; i < NLINES; i++) {
    //     for (j = 0; j < NLINES; j++) {
    //         printf("%d ", c[i * NLINES + j]);
    //     }
    //     printf("\n");
    // }
    //printf("\n");

    printf("%.5lf\n", timevalB.tv_sec-timevalA.tv_sec+(timevalB.tv_usec-timevalA.tv_usec)/(double)1000000);

    free(a); free(b); free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}