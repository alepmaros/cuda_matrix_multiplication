# Matrix Multiplication using GPU (CUDA)

Cuda Matrix Implementation using Global and Shared memory.

The input follows this pattern:

1. The number of lines of Matrix A
2. The number of columns of Matrix A
3. The number of lines of Matrix B
4. The number of columns of Matrix B
5. The values of Matrix A
6. The values of Matrix B

# Tests

Here is a few tests with a GeForce GTX 960M. The times are in seconds.

| Method                        | 1024 Lines | 2048 Lines | 4096 Lines | 6144 Lines |
|-------------------------------|------------|------------|------------|------------|
| Serial                        | 8.61027    | 115.55706  | ---        | ---        |
| CUDA w/ Global Memory         | 0.02713    | 0.20379    | 1.83966    | 6.40315    |
| CUDA w/ Shared Memory         | 0.01364    | 0.09895    | 0.76345    | 2.54037    |

And here is a plot comparing the two methods

![plot](https://raw.githubusercontent.com/Pridexs/cuda_matrix_multiplication/master/plot.png)

# Usage

0. (optional) Use the script generate_input.py to generate a random Matrix
