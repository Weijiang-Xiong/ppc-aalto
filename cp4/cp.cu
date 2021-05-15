// /*
// This is the function you need to implement. Quick reference:
// - input rows: 0 <= y < ny
// - input columns: 0 <= x < nx
// - element at row y and column x is stored in data[x + y*nx]
// - correlation between rows i and row j has to be stored in result[i + j*ny]
// - only parts with 0 <= j <= i < ny need to be filled
// */
// void correlate(int ny, int nx, const float *data, float *result) {
// }

#include "math.h"
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#define KERNEL_SIZE 16 // tuned a the submission portal

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

// transposed version 
__global__ void mykernel(int ny, int nx, float* transposed, float* result) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if (row >= ny || col >= ny)
        return;
    float inner_prod = 0;
    for (int i = 0; i < nx; i++)
    {   
        inner_prod +=  transposed[i*ny + row] * transposed[i*ny + col];
    }
    result[row*ny+col] = inner_prod;
}

// untransposed version 
// __global__ void mykernel(int ny, int nx, float* normalized, float* result) {
//     int row = threadIdx.x + blockIdx.x * blockDim.x;
//     int col = threadIdx.y + blockIdx.y * blockDim.y;
//     if (row >= ny or col >= ny)
//         return;
//     float inner_prod = 0;
//     for (int i = 0; i < nx; i++)
//     {   
//         // inner_prod +=  normalized[i + row*nx] * normalized[i + col*nx];
//     }
//     result[row*ny+col] = inner_prod;
// }

void correlate(int ny, int nx, const float* data, float* result) {

    const int pr_step = 20;                           // prefetch step
    float *normalized = (float *)malloc(sizeof(float)*ny*nx);
    float *transposed = (float *)malloc(sizeof(float)*ny*nx);
    // each row
    #pragma omp parallel for schedule(static,1)// need -fopenmp as compiler input
    for (int y = 0; y < ny; y++)
    {   
        // First normalize the input rows so that each row has the arithmetic mean of 0  be careful to do the normalization so that you do not change pairwise correlations.
        float row_sum = 0.0;
        float row_square_sum = 0.0;
        // iterate over columns
        for (int x = 0; x < nx; x++)
        {   
            __builtin_prefetch(&data[x + y*nx + pr_step]);
            row_sum += data[x + y*nx];
        }
        float rwo_avg = row_sum/nx;
        for (int x = 0; x < nx; x++)
        {   
            float item = data[x + y*nx]-rwo_avg;
            normalized[x + y*nx] = item;
            row_square_sum += pow(item, 2);
        }
        // Then normalize the input rows so that for each row the sum of the squares of the elements is 1 — again, be careful to do the normalization so that you do not change pairwise correlations.
        float root_square_sum = sqrt(row_square_sum);
        for (int x = 0; x < nx; x++)
        {
            __builtin_prefetch(&normalized[x + y*nx + pr_step]);
            // transposed of shape (nx, ny) normalized of shape (ny, nx)
            // transposed[x, y] = normalized[y, x]
            // transposed[y + x*ny] = normalized[x + y*nx]
            transposed[y + x*ny] = normalized[x + y*nx] / root_square_sum;
            // normalized[x + y*nx] /= root_square_sum;
            // transposed[y + x*ny] = normalized[x + y*nx];
        }
    }

    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nx * ny * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, transposed, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(KERNEL_SIZE, KERNEL_SIZE);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(ny, nx, dGPU, rGPU);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
    free(normalized);
    free(transposed);
}