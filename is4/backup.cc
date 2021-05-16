#include <cmath>
#include <vector>
#include <random>
#include <omp.h>

// void printVector(int A[], int len)
// {
// for (int i = 0; i < len; i++)
//     std::cout << A[i] << " ";
// }

// void printMatrix(int A[], int ny, int nx)
// {   
//     for (int y = 0; y < ny; y++)
//     {
//         for (int x = 0; x < nx; x++)
//         {
//             std::cout << A[y*ny + x] << " ";
//         }
//         std::cout << std::endl;
//     }
// }

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

constexpr float PAD_VALUE = 0.0;

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
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    // calculate the sum of pixel values 
    int numel = nx * ny;
    int prfetch = 20; 
    double4_t* vectorized = double4_alloc(numel);
    double4_t* img_sum = double4_alloc(numel + nx + ny + 1); // size (ny+1, nx+1)

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

    #pragma omp parallel for
    for (int y = 0; y < ny; y++)
    {
        img_sum[0 + (nx+1)*y] = d4zero; // set the first column to 0
    }

    #pragma omp parallel for
    for (int x = 0; x < nx; x++)
    {
        img_sum[x + (nx+1)*0] = d4zero; // set the first row to 0
    }

    #pragma omp parallel for
    for (int y = 1; y <= ny; y++) // start from item (1,1)
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
    double best_loss{-1}, current_loss{0};
    
    #pragma omp parallel
    for (int w = 0; w <= nx; w++) // start with a fixed window
    {
        #pragma omp for schedule(dynamic)
        for (int h = 0; h <= ny; h++) // then a lot more data reuse 
        {
            int numel_in = h*w;
            int numel_out = numel - numel_in;
            double inv_in = static_cast<double>(1) / numel_in;
            double inv_out = static_cast<double>(1) / numel_out;

            for (int y0 = 0; y0 <= ny-h; y0++)
            {
                for (int x0 = 0; x0 <= nx-w; x0++)
                {
                    /* code */
                }
                
            }
            
            
        }
        
    }
    


    


    
    






    return result;
}
