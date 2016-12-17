# Matrix Multiplication using GPU (CUDA)

Cuda Matrix Implementation using Global and Shared memory.

The input follows this pattern:

1. The number of lines of Matrix A
2. The number of columns of Matrix A
3. The number of lines of Matrix B
4. The number of columns of Matrix B
5. The values of Matrix A
6. The values of Matrix B

The Shared method was implemented by dividing the Matrices into blocks.

# Tests

Here is a few tests with a GeForce GTX 960M. The times are in milliseconds.

![plot1](https://raw.githubusercontent.com/alepmaros/cuda_matrix_multiplication/master/plot1.png)
![plot2](https://raw.githubusercontent.com/alepmaros/cuda_matrix_multiplication/master/plot2.png)

# Usage

0. (optional) Use the script generate_input.py to generate a random Matrix
1. Compile it & Run it
