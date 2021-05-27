#include <cmath>
#include <vector>
#include <random>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <ctime>

#define HUGE_NEGATIVE -10000

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

constexpr double PAD_VALUE = 0.0;

constexpr double4_t d4zero {
    PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE
};

static inline double sum_vec(double4_t vv) {
    double v = 0.0;    
    for (int i = 0; i < 4; ++i) {
        v += vv[i];
    }
    return v;
}

static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

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
    double4_t* vectorized = double4_alloc(numel);
    double4_t* img_sum = double4_alloc(numel + nx + ny + 1); // size (ny+1, nx+1)

    auto start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (int y = 0; y < ny; y++)
    {
        for (int x = 0; x < nx; x++)
        {   
            int position_1d = x + nx*y;
            __builtin_prefetch(&data[3*(position_1d + prfetch)+ 0]);
            __builtin_prefetch(&data[3*(position_1d + prfetch)+ 1]);
            __builtin_prefetch(&data[3*(position_1d + prfetch)+ 2]);
            vectorized[position_1d][0] = data[3*position_1d + 0];
            vectorized[position_1d][1] = data[3*position_1d + 1];
            vectorized[position_1d][2] = data[3*position_1d + 2];
        }
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "padding and vectorize: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    #pragma omp parallel for
    for (int y = 0; y <= ny; y++)
    {
        img_sum[0 + (nx+1)*y] = d4zero; // set the first column to 0
    }

    #pragma omp parallel for
    for (int x = 0; x <= nx; x++)
    {
        img_sum[x + (nx+1)*0] = d4zero; // set the first row to 0
    }
    // do not use omp here! it's inherently sequential 
    for (int y = 1; y <= ny; y++) // start from item (1,1) and propagate
    {
        for (int x = 1; x <= nx; x++)
        {
            // S(i, j) = S(i-1, j) + S(i, j-1) - S(i-1, j_1) + A(i-1, j-1)
            __builtin_prefetch(&img_sum[x + prfetch + (nx+1)*(y-1)]);
            __builtin_prefetch(&img_sum[(x-1) + prfetch+(nx+1)*(y-1)]);
            __builtin_prefetch(&vectorized[(x-1) + prfetch + nx*(y-1)]);
            img_sum[x + (nx+1)*y] = img_sum[x-1 + (nx+1)*y] + img_sum[x + (nx+1)*(y-1)] - img_sum[(x-1)+(nx+1)*(y-1)] + vectorized[(x-1) + nx*(y-1)];
        }
    }
    double4_t sum_all = img_sum[nx + (nx+1)*ny];

    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "calculating sum: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();

    double overall_best{HUGE_NEGATIVE};
    int x0_bst{0}, y0_bst{0}, x1_bst{0}, y1_bst{0}; //best coordinate in a thread
    #pragma omp parallel
    {   
        // local initialization keep track of the smallest loss, or largest negative loss. 
        double thread_best{HUGE_NEGATIVE}; // the best loss of this thread
        int x0_thr{0}, y0_thr{0}, x1_thr{0}, y1_thr{0}; //best coordinate in a thread
        // individual jobs for each thread
        // use dynamic because different window size means different amount of work
        #pragma omp for schedule(dynamic) 
        for (int w = 1; w <= nx; w++) // start with a fixed window
        {
            for (int h = 1; h <= ny; h++) 
            {
                // then some more data reuse here
                int numel_in = h*w;
                int numel_out = numel - numel_in;
                double inv_in = static_cast<double>(1) / numel_in;
                double inv_out = static_cast<double>(1) / numel_out;

                for (int y0 = 0; y0 <= ny-h; y0++)
                {
                    // put x in the inner most, so memory access will be continuous
                    for (int x0 = 0; x0 <= nx-w; x0++) 
                    {
                        int x1 = x0 + w;
                        int y1 = y0 + h;
                        // S_in = S(x1, y1) - S(x0, y1) - S(x1, y0) + S(x0, y0)
                        double4_t sum_in_vec = img_sum[x1+(nx+1)*y1] - img_sum[x0+(nx+1)*y1] - img_sum[x1+(nx+1)*y0] + img_sum[x0+(nx+1)*y0];
                        // double4_t sum_out_vec = sum_all - sum_in_vec;
                        // double4_t neg_loss_vec = inv_in * sum_in_vec *sum_in_vec + inv_out * sum_out_vec * sum_out_vec;
                        // double4_t neg_loss_vec = (inv_in+inv_out) * sum_in_vec * sum_in_vec - 2.0 * inv_out * sum_all * sum_in_vec + inv_out * sum_all * sum_all;
                        // have a look at https://en.wikipedia.org/wiki/Horner%27s_method
                        // gcc will prepare the coefficient outside the 2 inner most loops
                        double4_t neg_loss_vec = inv_out * sum_all * sum_all + sum_in_vec * (sum_in_vec * (inv_in+inv_out) - 2.0 * inv_out * sum_all);
                        double neg_loss = neg_loss_vec[0] + neg_loss_vec[1] + neg_loss_vec[2];
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
            }
        }

        #pragma omp critical // update global data 
        {
            if (thread_best > overall_best)
            {
                overall_best = thread_best;
                x0_bst = x0_thr;
                y0_bst = y0_thr;
                x1_bst = x1_thr;
                y1_bst = y1_thr;
            }
        }
    }

    // calculate final result with best value
    int numel_in = (y1_bst - y0_bst)*(x1_bst - x0_bst);
    int numel_out = numel - numel_in;
    double inv_in = static_cast<double>(1) / numel_in;
    double inv_out = static_cast<double>(1) / numel_out;
    double4_t sum_in_vec = img_sum[x1_bst+(nx+1)*y1_bst] - img_sum[x0_bst+(nx+1)*y1_bst] 
                         - img_sum[x1_bst+(nx+1)*y0_bst] + img_sum[x0_bst+(nx+1)*y0_bst];
    double4_t sum_out_vec = sum_all - sum_in_vec;
    sum_in_vec *= inv_in;
    sum_out_vec *= inv_out;

    Result result{y0_bst, x0_bst,
                  y1_bst, x1_bst,
                 {float(sum_out_vec[0]), float(sum_out_vec[1]), float(sum_out_vec[2])},
                 {float(sum_in_vec[0]), float(sum_in_vec[1]), float(sum_in_vec[2])}};

    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "iterating over all possible places: " << elapsed_seconds.count() << "s\n";

    free(vectorized);
    free(img_sum);
    return result;

}
