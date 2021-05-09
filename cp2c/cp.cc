/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

/* 
Looking at the recent assignments in a row, I found the results are rather interesting. 
when I use vectorization together with openMP for cp2c, the runtime for benchmark2 is 0.476s on the server,
compared to 0.626s in cp2b (openMP only), and 2.883s on the server with vectorization only (final submission)
*/

#include <cmath>
#include <vector>
#include <random>

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

void correlate(int ny, int nx, const float *data, float *result)
{
    const int vec_len = 4;            // divide a row into some groups
    const int num_vec = (nx + vec_len -1)/vec_len;  // number of vector per row, padded 
    // const int num_vec = nx/vec_len;  // number of vector per row, NOT padded
    // const int outlier = nx % vec_len; // the trailing elements that do not make a full group (if not padded)

    // copy the original data into a vectorized matrix
    double4_t* vectorized = double4_alloc(ny*num_vec);
    // keep the sequential normalization for now 
    std::vector<double> normalized(nx * ny);

    // each row
    #pragma omp parallel for // need -fopenmp as compiler input
    for (int y = 0; y < ny; y++)
    {
        // First normalize the input rows so that each row has the arithmetic mean of 0  be careful to do the normalization so that you do not change pairwise correlations.
        double row_sum = 0.0;
        double row_square_sum = 0.0;
        // iterate over columns
        for (int x = 0; x < nx; x++)
        {
            row_sum += data[x + y * nx];
        }
        double rwo_avg = row_sum / nx;
        for (int x = 0; x < nx; x++)
        {
            double item = data[x + y * nx] - rwo_avg;
            normalized[x + y * nx] = item;
            row_square_sum += pow(item, 2);
        }
        // Then normalize the input rows so that for each row the sum of the squares of the elements is 1 — again, be careful to do the normalization so that you do not change pairwise correlations.
        double root_square_sum = sqrt(row_square_sum);
        for (int x = 0; x < nx; x++)
        {
            normalized[x + y * nx] /= root_square_sum;
        }
        for (int i = 0; i < num_vec; i++)
        {
            for (int j = 0; j < vec_len; j++)
            {
                int real_idx = i*vec_len+j; // the idx of the element in the original row
                // padding with zero does not change the vector product
                vectorized[i+ num_vec*y][j] = real_idx < nx ? normalized[nx*y + real_idx] : 0.0;
            }
        }
    }
    // Let X be the normalized input matrix.
    // Calculate the (upper triangle of the) matrix product Y = XX.T.
    // row id in XX.T
    #pragma omp parallel for // need -fopenmp as compiler input
    for (int row = 0; row < ny; row++)
    {
        // column id in XX.T
        for (int col = row; col < ny; col++)
        {
            // vector product of each element
            // actually the row th row in X times the col th row in X
            // vectorize the inner product
            // split the vector into nx / vec_len blocks, each block contains vec_len items
            // the gid-th item belongs to the gid-th group, and sum the product by group
            double4_t vec_inner_prod = d4zero; // intermediate vector representation
            for (int vid = 0; vid < num_vec; vid++) // block id
            {
                vec_inner_prod += vectorized[vid+num_vec*row] * vectorized[vid+num_vec*col];
            }
            double inner_prod = sum_vec(vec_inner_prod);
            result[row * ny + col] = float(inner_prod);
        }
    }
    std::free(vectorized);
}

// int main()
// {
//     const int nx{400}, ny{100};
//     float data[nx*ny], result[nx*ny];
//     float data1[4] = {1.0, 2.0, 3.0, 4.0};
//     float result1[4] = {0.0, 0.0, 0.0, 0.0};
//     // std::vector<double> test(100, 6.6); size 100, default value 6.6
//     std::default_random_engine rand_eng;
//     std::normal_distribution<double> rand_dist(0.0, 1.0);
//     for (int i = 0; i < ny; i++)
//     {
//         for (int j = 0; j < nx; j++)
//         {
//             data[nx*i + j] = rand_dist(rand_eng);
//             // data[nx*i+j]= 1.0;
//         }
//     }
//     // result1 should be [1 1 0 1]
//     correlate(2, 2, data1, result1);
    
//     correlate(ny, nx, data, result); 
//     return 0;
// }
