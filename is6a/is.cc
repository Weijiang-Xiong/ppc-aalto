#include <cmath>
#include <vector>
#include <random>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <ctime>
#include <immintrin.h>

#define HUGE_NEGATIVE -10000
#define TINY_POSITIVE 1e-15
#define VECTOR_LENGTH 8

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

typedef float float8_t __attribute__((vector_size(VECTOR_LENGTH * sizeof(float))));

constexpr float PAD_VALUE_F = 0.0;

constexpr float8_t fVeczero{
    PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F,
    PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F};

static inline float max_vec(float8_t vv)
{
    float m1 = (vv[0]<=vv[1]) ? vv[1] : vv[0];
    float m2 = (vv[2]<=vv[3]) ? vv[3] : vv[2];
    float m3 = (vv[4]<=vv[5]) ? vv[5] : vv[4];
    float m4 = (vv[6]<=vv[7]) ? vv[7] : vv[6];
    float m11 = (m1 < m2) ? m2 : m1;
    float m22 = (m3 < m4) ? m4 : m3;
    float v = (m11 < m22) ? m22 : m11;
    return v;
};

static float8_t *float8_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(float8_t), sizeof(float8_t) * n))
    {
        throw std::bad_alloc();
    }
    return (float8_t *)tmp;
};

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
- input: data[c + 3 * x + 3 * nx * y], this is a binary image with elements (0,0,0) or (1,1,1)
therefore 1 channel is enough to represent the whole binary image

The overall process:
    1. extract one channel from the original data
    2. calculate integral image (sum over all items on the upper left corner of a pixel)
    3. copy the integral image by a factor VECTOR_LENGTH
    4. use vectorized calculation to find best window size (4 nested for loop)
    5. find best location for that best window

The formula for step 2 is 
    S(i, j) = S(i-1, j) + S(i, j-1) - S(i-1, j_1) + A(i-1, j-1)
    where S represents the integral image and A is the extracted channel in step 1

The reason for step 3:
    the core formula for S_in = S(x1, y1) - S(x0, y1) - S(x1, y0) + S(x0, y0)
    if we let x0 coordinates vary from x0+0 to x0+7 (8 elements) we have 
    S_in0 = S(x1+0, y1) - S(x0+0, y1) - S(x1+0, y0) + S(x0+0, y0)
    S_in1 = S(x1+1, y1) - S(x0+1, y1) - S(x1+1, y0) + S(x0+1, y0)
    ... ... 
    S_in7 = S(x1+7, y1) - S(x0+7, y1) - S(x1+7, y0) + S(x0+7, y0)
    we can calculate [S_in0, ..., S_in7] in a vector, if we have 
    [S(x+0, y), S(x+1, y), ..., S(x+7,y)] for every position (x,y)
    it make sense to use edge padding for the iamge sum, because normally
    we would use zero padding for the original image, so the sum does not 
    change after exceeding the boundary

Before vectorizing step 4, the time sonsumption of the scalar version is 
benchmarks/4.txt                   6.017s  pass
Standard output:
getting one channel from original: 0.000591s
calculating image sum for one channel image: 0.0018046s
finding best window size: 6.01388s
finding best location for best window: 0.0001902s

After vectorization 
benchmarks/4.txt                   3.685s  pass
Standard output:
calculating image sum for one channel image: 0.0024294s
stacked img_sum into a data vector: 0.0023033s
finding best window size: 3.67949s
finding best location for best window: 0.0002392s

get 4 points, so difficult 
*/
Result segment(int ny, int nx, const float *data) {
    // calculate the sum of pixel values 
    int numel = nx * ny;
    float* img_sum = (float*) malloc(sizeof(float) * (nx+1)*(ny+1)); // size (ny+1, nx+1)
    int nx_p = nx+VECTOR_LENGTH;
    float* img_sum_pad = (float*) malloc(sizeof(float) * nx_p *(ny+1));
    // float8_t* img_sum_mask = float8_alloc((nx+1)*(ny+1));
    // float8_t* img_sum_vec = float8_alloc((nx+1)*(ny+1)); // stacked sum_img

    // auto start = std::chrono::system_clock::now();
    
    #pragma omp parallel for
    for (int y = 0; y <= ny; y++)
    {
        img_sum[0 + (nx+1)*y] = PAD_VALUE_F; // set the first column to 0
    }

    #pragma omp parallel for
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
    
    #pragma omp parallel for
    for (int y = 0; y <= ny; y++) // start from item (1,1) and propagate
    {
        for (int x = 0; x < nx_p; x++)
        {
            // S(i, j) = S(i-1, j) + S(i, j-1) - S(i-1, j_1) + A(i-1, j-1)
            img_sum_pad[x + nx_p*y] = x<=nx ? img_sum[x + (nx+1)*y] : std::sqrt(-1);
        }
    }

    float sum_all = img_sum[nx + (nx+1)*ny];
    // printMatrix(img_sum, ny+1, nx+1);

    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end-start;
    // std::cout << "calculating image sum for one channel image: " << elapsed_seconds.count() << "s\n";

    // ====================================================================//

    // start = std::chrono::system_clock::now();
    // for (int y = 0; y <= ny; y++)
    // {
    //     for (int x = 0; x <= nx; x++)
    //     {
    //         for (int idx = 0; idx < VECTOR_LENGTH; idx++)
    //         {   
    //             img_sum_vec[x + (nx+1)*y][idx] = img_sum[std::min(x+idx,nx) + (nx+1)*y];
    //         }
    //     }
    // }

    // end = std::chrono::system_clock::now();
    // elapsed_seconds = end-start;
    // std::cout << "stacked img_sum into a data vector: " << elapsed_seconds.count() << "s\n";

    // start = std::chrono::system_clock::now();

    float overall_best{HUGE_NEGATIVE};
    int x0_bst{0}, y0_bst{0}, x1_bst{0}, y1_bst{0}, w_bst{0}, h_bst{0}; //best coordinate in a thread
    #pragma omp parallel
    {   
        // local initialization keep track of the smallest loss, or largest negative loss. 
        float thread_best{HUGE_NEGATIVE}; // the best loss of this thread
        float8_t window_best_vec = fVeczero; // the window size of the best loss in this thread
        float window_best = 0; // the window size of the best loss in this thread
        int w_win{0}, h_win{0}; // best h and w for a window
        // individual jobs for each thread
        // use dynamic because different window size means different amount of work
        #pragma omp for schedule(dynamic, 1) 
        for (int h = 1; h <= ny; h++)  // start with a fixed window
        {
            for (int y0 = 0; y0 <= ny-h; y0++)
            {
                
                for (int w = 1; w <= nx; w++)
                {
                    // then some more data reuse here
                    int numel_in = h*w;
                    int numel_out = numel - numel_in;
                    float inv_in = static_cast<float>(1) / numel_in;
                    float inv_out = static_cast<float>(1) / numel_out;
                    // vector version
                    for (int x0 = 0; x0 <= nx-w; x0+=VECTOR_LENGTH)
                    {
                        int x1 = x0 + w;
                        int y1 = y0 + h;
                        // S_in = S(x1, y1) - S(x0, y1) - S(x1, y0) + S(x0, y0)
                        __m256 S_x1y1 = _mm256_loadu_ps(&img_sum_pad[x1 + nx_p*y1]);
                        __m256 S_x0y1 = _mm256_loadu_ps(&img_sum_pad[x0 + nx_p*y1]);
                        __m256 S_x1y0 = _mm256_loadu_ps(&img_sum_pad[x1 + nx_p*y0]);
                        __m256 S_x0y0 = _mm256_loadu_ps(&img_sum_pad[x0 + nx_p*y0]);
                        float8_t sum_in_vec = S_x1y1 - S_x1y0 - S_x0y1 + S_x0y0;
                        // float8_t sum_in_vec = img_sum_vec[x1+(nx+1)*y1] - img_sum_vec[x0+(nx+1)*y1] 
                        //                     - img_sum_vec[x1+(nx+1)*y0] + img_sum_vec[x0+(nx+1)*y0];
                        // have a look at https://en.wikipedia.org/wiki/Horner%27s_method
                        // gcc will prepare the coefficients outside the 2 inner most loops, actual loop work reduced
                        float8_t neg_loss_vec = inv_out * sum_all * sum_all + sum_in_vec * (sum_in_vec * (inv_in+inv_out) - float(2.0) * inv_out * sum_all);
                        window_best_vec = neg_loss_vec > window_best_vec ? neg_loss_vec : window_best_vec;
                        // float error = 0;
                        // for (int idx = 0; idx < VECTOR_LENGTH; idx++) {
                        //     if (idx+x1>nx) break;
                        //     error = std::max(neg_loss_vec[idx], error);
                        // }
                        // window_best = error < window_best ? window_best : error;
                    }
                    window_best = max_vec(window_best_vec);
                    if (window_best > thread_best)
                    {
                        thread_best = window_best;
                        w_win = w;
                        h_win = h;
                    }
                }
            }
        }
        #pragma omp critical // update global data 
        {
            if (thread_best > overall_best)
            {
                overall_best = thread_best;
                w_bst = w_win;
                h_bst = h_win;
            }
        }
    }
    // end = std::chrono::system_clock::now();
    // elapsed_seconds = end-start;
    // std::cout << "finding best window size: " << elapsed_seconds.count() << "s\n";

    // start = std::chrono::system_clock::now();
    int numel_in = h_bst*w_bst;
    int numel_out = numel - numel_in;
    float inv_in = static_cast<float>(1) / numel_in;
    float inv_out = static_cast<float>(1) / numel_out;
    for (int y0 = 0; y0 <= ny-h_bst; y0++)
    {
        // put x in the inner most, so memory access will be continuous
        for (int x0 = 0; x0 <= nx-w_bst; x0++) 
        {
            int x1 = x0 + w_bst;
            int y1 = y0 + h_bst;
            // S_in = S(x1, y1) - S(x0, y1) - S(x1, y0) + S(x0, y0)
            float sum_in = img_sum[x1+(nx+1)*y1] - img_sum[x0+(nx+1)*y1] - img_sum[x1+(nx+1)*y0] + img_sum[x0+(nx+1)*y0];
            float neg_loss = inv_out * sum_all * sum_all + sum_in * (sum_in * (inv_in+inv_out) - float(2.0) * inv_out * sum_all);
            if (neg_loss >= overall_best)
            {
                x0_bst = x0;
                y0_bst = y0;
                x1_bst = x1;
                y1_bst = y1;
            }
        }
    }

    // calculate final result with best value
    numel_in = (y1_bst - y0_bst)*(x1_bst - x0_bst);
    numel_out = numel - numel_in;
    inv_in = static_cast<float>(1) / numel_in;
    inv_out = static_cast<float>(1) / numel_out;
    float sum_in = img_sum[x1_bst+(nx+1)*y1_bst] - img_sum[x0_bst+(nx+1)*y1_bst] 
                         - img_sum[x1_bst+(nx+1)*y0_bst] + img_sum[x0_bst+(nx+1)*y0_bst];
    float sum_out = sum_all - sum_in;
    sum_in *= inv_in;
    sum_out *= inv_out;

    Result result{y0_bst, x0_bst,
                  y1_bst, x1_bst,
                 {float(sum_out), float(sum_out), float(sum_out)},
                 {float(sum_in), float(sum_in), float(sum_in)}};

    // end = std::chrono::system_clock::now();
    // elapsed_seconds = end-start;
    // std::cout << "finding best location for best window: " << elapsed_seconds.count() << "s\n";

    free(img_sum);
    // free(img_sum_mask);
    // free(img_sum_vec);
    free(img_sum_pad);

    return result;
}
