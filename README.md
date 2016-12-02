# Matrix Multiplication using GPU (CUDA)

Cuda Matrix Implementation using Global and Shared memory.

It is really restrict at the moment, here is is the requirements for the Matrices:
1. The number of lines needs to be equal to the number of columns
2. The number of lines (and columns) needs to be divisible by 32

# Tests

Here is a few tests with a GeForce GTX 960M. The times are in seconds.

| Method                        | 1024 Lines | 2048 Lines | 4096 Lines | 6144 Lines |
|-------------------------------|------------|------------|------------|------------|
| Serial                        | 8.61027    | 115.55706  | ---        | ---        |
| CUDA w/ Global Memory         | 0.02713    | 0.20379    | 1.83966    | 6.40315    |
| CUDA w/ Shared Memory         | 0.01364    | 0.09895    | 0.76345    | 2.54037    |

And here is a plot comparing the two methods

![plot][plot.png]

# Usage

0. (optional) Use the script generate_input.py to generate a random Matrix
1. Set the #DEFINE NLINES and NCOLUMNS to the size of your Matrix.
2. Compile with nvcc
