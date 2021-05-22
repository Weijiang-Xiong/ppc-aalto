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
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <iostream>

#define BLOCK_SZ 64
#define THREAD_SZ 8

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

// almost the same as the one in slides
__global__ void gpu_cov_mxt(int nx, int ny, int nx_p, int ny_p, float* transposed, float* result) {

    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    // we don't need to calculate the values for the blocks below the diagonal
    // just assign 0 to the values
    // instead of doing nothing we have to assign 0 to them
    if (ic > jc)
    {
        for (int ib = 0; ib < THREAD_SZ; ib++)
        {
            for (int jb = 0; jb < THREAD_SZ; jb++)
            {
                int i = ic * BLOCK_SZ + ib * THREAD_SZ + ia;
                int j = jc * BLOCK_SZ + jb * THREAD_SZ + ja;
                if (i < ny && j < ny) result[j + i*ny] = 0.0;
            }
        }
    } else {
        // almost the same as the example here https://ppc.cs.aalto.fi/ch4/v2/
        float v[THREAD_SZ][THREAD_SZ];
        for (int ib = 0; ib < THREAD_SZ; ib++)
        {
            for (int jb = 0; jb < THREAD_SZ; jb++)
            {
                v[ib][jb] = 0;
            }
        }
        for (int k = 0; k < nx; k++)
        {
            float x[THREAD_SZ];
            float y[THREAD_SZ];
            for (int ib = 0; ib < THREAD_SZ; ib++)
            {
                int i = ic * BLOCK_SZ + ib * THREAD_SZ + ia;
                // use the transposed matrix, so threads in a warp has consecutive 
                // index i values, and they will access continuous memory location
                x[ib] = transposed[ny_p * k + i];
            }
            for (int jb = 0; jb < THREAD_SZ; jb++)
            {
                int j = jc * BLOCK_SZ + jb * THREAD_SZ + ja;
                y[jb] = transposed[ny_p * k + j];
            }
            // do 8*8 useful calculation after reading 8+8 numbers
            for (int ib = 0; ib < THREAD_SZ; ib++)
            {
                for (int jb = 0; jb < THREAD_SZ; jb++)
                {
                    v[ib][jb] += x[ib] * y[jb];
                }
            }
        }
        for (int ib = 0; ib < THREAD_SZ; ib++)
        {
            for (int jb = 0; jb < THREAD_SZ; jb++)
            {
                int i = ic * BLOCK_SZ + ib * THREAD_SZ + ia;
                int j = jc * BLOCK_SZ + jb * THREAD_SZ + ja;
                if (i < ny && j < ny) result[j + i*ny] = v[ib][jb];
            }
        }
    }
}

__global__ void gpu_mtx_transpose(int nx, int ny, int nx_p, int ny_p, float* transposed, float* normalized) {
    int ja = threadIdx.x; // 0, 1, ..., 63
    int i = blockIdx.y; // 0, 1, ..., ny_p

    for (int jb = 0; jb < nx_p; jb += BLOCK_SZ) // a row contains nx_p elements after padding
    {
        int j = jb + ja;
        // just switch i and j, remember the transposed matrix is padded so use ny_p
        transposed[ny_p * j + i] = (i < ny && j < nx) ? normalized[nx * i + j] : 0;
    }
}

__global__ void gpu_normalize(int nx, int ny, int nx_p, int ny_p, float* normalized, float* data) {
    int ja = threadIdx.y; // 0, 1, ..., 63
    int i = blockIdx.y; // 0, 1, ..., ny_p/BLOCK_SZ

    int y = i*BLOCK_SZ + ja;
    // if we don't write y<ny here, there would be memory access errors
    // the actual content is basically the same as CPU code. 
    if (y < ny)
    {
        float row_sum = 0.0;
        float row_square_sum = 0.0;
        // iterate over columns
        for (int x = 0; x < nx; x++)
        {   
            row_sum += data[x + y*nx];
        }
        float rwo_avg = row_sum/nx;
        for (int x = 0; x < nx; x++)
        {   
            float item = data[x + y*nx]-rwo_avg;
            normalized[x + y*nx] = item;
            row_square_sum += pow(item, 2);
        }
        float root_square_sum = sqrt(row_square_sum);
        for (int x = 0; x < nx; x++)
        {
            normalized[x + y*nx] /= root_square_sum;         
        }
    }
}


void correlate(int ny, int nx, const float* data, float* result) {
    
    int nx_p = roundup(nx, BLOCK_SZ); // how many blocks in x directon
    int ny_p = roundup(ny, BLOCK_SZ); // how many blocks in y directon

    // initialize pointers to null
    float* dGPU = NULL;
    float* dGPU_raw = NULL;
    float* dGPU_norm = NULL;
    float* rGPU = NULL;

    // calculate matrix sizes
    const int pad_sz = nx_p * ny_p * sizeof(float);
    const int mtx_sz = nx * ny * sizeof(float);
    const int out_sz = ny * ny * sizeof(float);

    // allocate memory on GPU and copy original data to GPU
    CHECK(cudaMalloc((void**)&dGPU, pad_sz));
    CHECK(cudaMalloc((void**)&dGPU_raw, mtx_sz));
    CHECK(cudaMalloc((void**)&dGPU_norm, mtx_sz));
    CHECK(cudaMalloc((void**)&rGPU, out_sz));
    CHECK(cudaMemcpy(dGPU_raw, data, mtx_sz, cudaMemcpyHostToDevice));

    // GPU matrix normalization
    {
        dim3 dimBlock(1, BLOCK_SZ);   // a thread normalizes a row 
        dim3 dimGrid(1, ny_p/BLOCK_SZ);  // ny_p/BLOCK_SZ blocks process all ny_p rows
        gpu_normalize<<<dimGrid, dimBlock>>>(nx, ny, nx_p, ny_p, dGPU_norm, dGPU_raw);
        CHECK(cudaGetLastError());
    }

    // GPU matrix transpose  
    {
        dim3 dimBlock(BLOCK_SZ, 1);   // a thread process 1/64 row 
        dim3 dimGrid(1, ny_p);  // a block process a row
        gpu_mtx_transpose<<<dimGrid, dimBlock>>>(nx, ny, nx_p, ny_p, dGPU, dGPU_norm);
        CHECK(cudaGetLastError());
    }

    // Run kernel  GPU multiplication 
    {
        // a block contains 8*8 thread, and each calculates 8*8 output
        // as a result a block calculates 64 * 64 of the final output
        dim3 dimBlock(THREAD_SZ, THREAD_SZ); 
        // so we need ny_p/64 blocks for the whole output 
        dim3 dimGrid(ny_p / BLOCK_SZ, ny_p / BLOCK_SZ); 
        gpu_cov_mxt<<<dimGrid, dimBlock>>>(nx, ny, nx_p, ny_p, dGPU, rGPU);
        CHECK(cudaGetLastError());
    }

    // Copy data back to CPU & release memory 
    CHECK(cudaMemcpy(result, rGPU, out_sz, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(dGPU_raw));
    CHECK(cudaFree(dGPU_norm));
    CHECK(cudaFree(rGPU));
}