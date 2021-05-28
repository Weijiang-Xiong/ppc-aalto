#include <cmath>
#include <vector>
#include <random>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <ctime>

#define HUGE_NEGATIVE -10000
#define VECTOR_LENGTH 4

typedef float float4_t __attribute__((vector_size(VECTOR_LENGTH * sizeof(float))));

constexpr float PAD_VALUE_F = 0.0;

constexpr float4_t fVeczero{
    PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F};

static inline float min_vec4(float4_t vv)
{
    float v = std::min(
            std::min(vv[0], vv[1]),
            std::min(vv[2], vv[3]));
    return v;
};

static float4_t *float4_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(float4_t), sizeof(float4_t) * n))
    {
        throw std::bad_alloc();
    }
    return (float4_t *)tmp;
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
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    // calculate the sum of pixel values 
    int numel = nx * ny;
    int prfetch = 20; 
    int num_vec = (nx + VECTOR_LENGTH - 1) / VECTOR_LENGTH;
    int nx_p = num_vec * VECTOR_LENGTH;
    float* one_channel = (float*) malloc(sizeof(float) * numel); // the input will be a binary mask, so only 1 channel needed
    float* img_sum = (float*) malloc(sizeof(float) * (nx+1)*(ny+1)); // size (ny+1, nx+1)
    // float4_t* img_sum_vec = float4_alloc(VECTOR_LENGTH * (nx_p+1)*(ny+1)); // stacked sum_img

    // auto start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {   
            int position_1d = x + nx*y;
            __builtin_prefetch(&data[3*(position_1d + prfetch)+ 0]);
            one_channel[position_1d] = data[3*position_1d + 0];
        }
    }

    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end-start;
    // std::cout << "padding and vectorize: " << elapsed_seconds.count() << "s\n";

    // start = std::chrono::system_clock::now();
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
            __builtin_prefetch(&img_sum[x + prfetch + (nx+1)*(y-1)]);
            __builtin_prefetch(&img_sum[(x-1) + prfetch+(nx+1)*(y-1)]);
            img_sum[x + (nx+1)*y] = img_sum[x-1 + (nx+1)*y] + img_sum[x + (nx+1)*(y-1)] - img_sum[(x-1)+(nx+1)*(y-1)] + one_channel[(x-1) + nx*(y-1)];
        }
    }

    float sum_all = img_sum[nx + (nx+1)*ny];

    // end = std::chrono::system_clock::now();
    // elapsed_seconds = end-start;
    // std::cout << "calculating sum: " << elapsed_seconds.count() << "s\n";

    // start = std::chrono::system_clock::now();

    float overall_best{HUGE_NEGATIVE};
    int x0_bst{0}, y0_bst{0}, x1_bst{0}, y1_bst{0}, w_bst{0}, h_bst{0}; //best coordinate in a thread
    #pragma omp parallel
    {   
        // local initialization keep track of the smallest loss, or largest negative loss. 
        float thread_best{HUGE_NEGATIVE}; // the best loss of this thread
        float window_best{HUGE_NEGATIVE}; // the window size of the best loss in this thread
        int w_win{0}, h_win{0}; // best h and w for a window
        // individual jobs for each thread
        // use dynamic because different window size means different amount of work
        #pragma omp for schedule(dynamic, 1) 
        for (int w = 1; w <= nx; w++) // start with a fixed window
        {
            for (int h = 1; h <= ny; h++) 
            {
                // then some more data reuse here
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
                        float sum_in_vec = img_sum[x1+(nx+1)*y1] - img_sum[x0+(nx+1)*y1] - img_sum[x1+(nx+1)*y0] + img_sum[x0+(nx+1)*y0];
                        // have a look at https://en.wikipedia.org/wiki/Horner%27s_method
                        // gcc will prepare the coefficient outside the 2 inner most loops
                        float neg_loss = inv_out * sum_all * sum_all + sum_in_vec * (sum_in_vec * (inv_in+inv_out) - float(2.0) * inv_out * sum_all);
                        if (neg_loss > window_best)
                        {
                            window_best = neg_loss;
                        }
                    }
                }
                if (window_best > thread_best)
                {
                    thread_best = window_best;
                    w_win = w;
                    h_win = h;
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
            float sum_in_vec = img_sum[x1+(nx+1)*y1] - img_sum[x0+(nx+1)*y1] - img_sum[x1+(nx+1)*y0] + img_sum[x0+(nx+1)*y0];
            float neg_loss = inv_out * sum_all * sum_all + sum_in_vec * (sum_in_vec * (inv_in+inv_out) - float(2.0) * inv_out * sum_all);
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
    float sum_in_vec = img_sum[x1_bst+(nx+1)*y1_bst] - img_sum[x0_bst+(nx+1)*y1_bst] 
                         - img_sum[x1_bst+(nx+1)*y0_bst] + img_sum[x0_bst+(nx+1)*y0_bst];
    float sum_out_vec = sum_all - sum_in_vec;
    sum_in_vec *= inv_in;
    sum_out_vec *= inv_out;

    Result result{y0_bst, x0_bst,
                  y1_bst, x1_bst,
                 {float(sum_out_vec), float(sum_out_vec), float(sum_out_vec)},
                 {float(sum_in_vec), float(sum_in_vec), float(sum_in_vec)}};

    // end = std::chrono::system_clock::now();
    // elapsed_seconds = end-start;
    // std::cout << "iterating over all possible places: " << elapsed_seconds.count() << "s\n";

    free(img_sum);
    free(one_channel);

    return result;
}