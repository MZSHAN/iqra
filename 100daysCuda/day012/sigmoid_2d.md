#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void sigmoid_kernel_2d(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;  // 2D to 1D index mapping
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}