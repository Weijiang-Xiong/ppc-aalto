#include <cmath>
#include <vector>
#include <random>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <algorithm>

#define HUGE_NEGATIVE -10000
#define BLOCK_x 32
#define BLOCK_y 2

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

constexpr float PAD_VALUE_F = 0.0;
constexpr float TWO = float(2.0);

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/

__global__ void GPU_best_location(int nx, int ny, float* img_sum_gpu, float* window_loss_gpu, int* window_best_gpu){
    int w = threadIdx.x + blockIdx.x * blockDim.x + 1; // w = 1, 2, ..., nx, nx different values in total
    int h = threadIdx.y + blockIdx.y * blockDim.y + 1; // h = 1, 2, ..., ny, ny different values in total
    if (w>nx or h>ny) return; 

    float sum_all = img_sum_gpu[nx + (nx+1)*ny];

    float thread_best{-1.0};
    int x0_thr{0}, y0_thr{0}, x1_thr{0}, y1_thr{0}; // best location for a window with size (w, h)

    int numel = nx * ny;
    int numel_in = h*w;
    int numel_out = numel - numel_in;
    float inv_in = static_cast<float>(1) / numel_in;
    float inv_out = static_cast<float>(1) / numel_out;

    for (int y0 = 0; y0 <= ny-h; y0++)
    {
        // put x in the inner most, so memory access will be continuous
        for (int x0 = 0; x0 <= nx-w; x0++) 
        {
            int x1 = x0 + w;
            int y1 = y0 + h;
            // S_in = S(x1, y1) - S(x0, y1) - S(x1, y0) + S(x0, y0)
            float sum_in_vec = img_sum_gpu[x1+(nx+1)*y1] - img_sum_gpu[x0+(nx+1)*y1] 
                             - img_sum_gpu[x1+(nx+1)*y0] + img_sum_gpu[x0+(nx+1)*y0];
            // have a look at https://en.wikipedia.org/wiki/Horner%27s_method
            // gcc will prepare the coefficient outside the 2 inner most loops
            float neg_loss = inv_out*sum_all*sum_all + sum_in_vec*(sum_in_vec*(inv_in+inv_out) - TWO*inv_out*sum_all);
            if (neg_loss > thread_best)
            {
                thread_best = neg_loss;
                x0_thr = x0;
                y0_thr = y0;
                x1_thr = x1;
                y1_thr = y1;
            }
        }
    }
    // window_loss_gpu of size (ny, nx) window_best_gpu both of size (ny, nx, 4)
    int w_idx = w-1;
    int h_idx = h-1;
    window_loss_gpu[w_idx + nx*h_idx] = thread_best;
    window_best_gpu[4*(w_idx + nx*h_idx) + 0] =  x0_thr;
    window_best_gpu[4*(w_idx + nx*h_idx) + 1] =  y0_thr;
    window_best_gpu[4*(w_idx + nx*h_idx) + 2] =  x1_thr;
    window_best_gpu[4*(w_idx + nx*h_idx) + 3] =  y1_thr;
}

Result segment(int ny, int nx, const float *data) {
    // calculate the sum of pixel values 
    int numel = nx * ny;
    int numel_p = (nx+1) * (ny+1);
    std::vector<float> img_sum(numel_p, 0);
    std::vector<float> window_loss(numel, 0); // best loss for each window
    std::vector<int> window_best(4*numel, 0); // best location (x0, y0) for each window
    
    // #pragma omp parallel for
    for (int y = 0; y <= ny; y++)
    {
        img_sum[0 + (nx+1)*y] = PAD_VALUE_F; // set the first column to 0
    }

    // #pragma omp parallel for
    for (int x = 0; x <= nx; x++)
    {
        img_sum[x + (nx+1)*0] = PAD_VALUE_F; // set the first row to 0
    }
    // do not use omp here! it's inherently sequential 
    for (int y = 1; y <= ny; y++) // start from item (1,1) and propagate
    {
        for (int x = 1; x <= nx; x++)
        {
            // S(i, j) = S(i-1, j) + S(i, j-1) - S(i-1, j_1) + A(i-1, j-1)
            img_sum[x + (nx+1)*y] = img_sum[x-1 + (nx+1)*y] + img_sum[x + (nx+1)*(y-1)] 
                                  - img_sum[(x-1)+(nx+1)*(y-1)] + data[3*((x-1) + nx*(y-1))];
        }
    }
    float sum_all = img_sum[nx + (nx+1)*ny];

    float* img_sum_gpu = NULL;
    float* window_loss_gpu = NULL; // best loss for each window
    int* window_best_gpu = NULL; // best location (x0, y0) for each window

    CHECK(cudaMalloc((void**)&img_sum_gpu, numel_p*sizeof(float)));
    CHECK(cudaMalloc((void**)&window_loss_gpu, numel*sizeof(float)));
    CHECK(cudaMalloc((void**)&window_best_gpu, 4*numel*sizeof(int)));
    CHECK(cudaMemcpy(img_sum_gpu, img_sum.data(), numel_p*sizeof(float), cudaMemcpyHostToDevice));

    // run kernel
    {
        // window size w = 1, 2, ..., nx , h = 1, 2, ..., ny 
        dim3 dimBlock(BLOCK_x,BLOCK_y); // each thread takes care of a window size (w, h)
        dim3 dimGrid(divup(nx, BLOCK_x), divup(ny, BLOCK_y)); // need how many blocks needed
        // std::cout << " nx " << nx << " ny " << ny << std::endl;
        // std::cout << " dimBlock.x " << dimBlock.x << " dimBlock.y " << dimBlock.y << std::endl;
        // std::cout << " dimGrid.x " << dimGrid.x << " dimGrid.y " << dimGrid.y << std::endl; 
        GPU_best_location<<<dimGrid, dimBlock>>>(nx, ny, img_sum_gpu, window_loss_gpu, window_best_gpu);
        CHECK(cudaGetLastError());
    }

    CHECK(cudaMemcpy(window_loss.data(), window_loss_gpu, numel*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(window_best.data(), window_best_gpu, 4*numel*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(img_sum_gpu));
    CHECK(cudaFree(window_loss_gpu));
    CHECK(cudaFree(window_best_gpu));

    int idx = std::distance(window_loss.begin(), std::max_element(window_loss.begin(), window_loss.end()));
    int x0_bst = window_best[4 * idx + 0];
    int y0_bst = window_best[4 * idx + 1];
    int x1_bst = window_best[4 * idx + 2];
    int y1_bst = window_best[4 * idx + 3];

    // calculate final result with best value
    int numel_in = (y1_bst - y0_bst)*(x1_bst - x0_bst);
    int numel_out = numel - numel_in;
    float inv_in = static_cast<float>(1) / numel_in;
    float inv_out = static_cast<float>(1) / numel_out;
    float sum_in_vec = img_sum[x1_bst+(nx+1)*y1_bst] - img_sum[x0_bst+(nx+1)*y1_bst] 
                     - img_sum[x1_bst+(nx+1)*y0_bst] + img_sum[x0_bst+(nx+1)*y0_bst];
    float sum_out_vec = sum_all - sum_in_vec;
    sum_in_vec *= inv_in;
    sum_out_vec *= inv_out;

    Result result{y0_bst, x0_bst,
                  y1_bst, x1_bst,
                 {float(sum_out_vec), float(sum_out_vec), float(sum_out_vec)},
                 {float(sum_in_vec), float(sum_in_vec), float(sum_in_vec)}};

    return result;
}
