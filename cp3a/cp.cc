/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <cmath>
#include <vector>
#include <random>
// #include <iostream>
// #include <chrono>
// #include <ctime>

typedef float float8_t __attribute__((vector_size(8 * sizeof(float))));
typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

constexpr float PAD_VALUE = 0.0;
constexpr float PAD_VALUE_D = 0.0;

constexpr float8_t f8zero{
    PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE,
    PAD_VALUE, PAD_VALUE, PAD_VALUE, PAD_VALUE};

constexpr double4_t d4zero{
    PAD_VALUE_D, PAD_VALUE_D, PAD_VALUE_D, PAD_VALUE_D};

static inline float sum_vec(float8_t vv)
{
    double v = 0.0;
    for (int i = 0; i < 8; ++i)
    {
        v += vv[i];
    }
    return v;
};

static inline double sum_vec(double4_t vv)
{
    double v = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        v += vv[i];
    }
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

static double4_t *double4_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n))
    {
        throw std::bad_alloc();
    }
    return (double4_t *)tmp;
};

void correlate(int ny, int nx, const float *data, float *result)
{
    const int cell_sz = 4;                            // cell size, divide the matrix X @ X.T into cells of kxk, say 4x4
    const int vec_len = 4;                            // number of elements in a vector
    const int pr_step = 15;                           // prefetch step
    const int num_cel = (ny + cell_sz - 1) / cell_sz; // number of kxk cells per column, in x @ X.T
    const int num_vec = (nx + vec_len - 1) / vec_len; // number of vector per row, padded
    const int pad_row = cell_sz * num_cel;            // number of rows after padding 
    
    // copy the original data into a vectorized matrix
    double4_t *vectorized = double4_alloc(pad_row * num_vec);
    // keep the sequential normalization for now
    std::vector<double> normalized(nx * ny);
    // normalization only accounts for a small fraction of total runtime
    // auto start = std::chrono::system_clock::now();
    #pragma omp parallel for // need -fopenmp as compiler input
    for (int y = 0; y < ny; y++)
    {
        // for those within 
        // First normalize the input rows so that each row has the arithmetic mean of 0  
        // be careful to do the normalization so that you do not change pairwise correlations.
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
                int real_idx = i * vec_len + j; // the idx of the element in the original row
                // padding with zero does not change the vector product
                vectorized[i + num_vec * y][j] = real_idx < nx ? normalized[nx * y + real_idx] : 0.0;
            }
        }
    }

    #pragma omp parallel for // need -fopenmp as compiler input
    for (int y = ny; y < pad_row; y++)
    {
        for (int i = 0; i < num_vec; i++)
        {
            vectorized[i + num_vec * y] = d4zero;
        }
    }
    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end-start;
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

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
            double4_t vec_inner_prod = d4zero;      // intermediate vector representation
            for (int vid = 0; vid < num_vec; vid++) // block id
            {   
                __builtin_prefetch(&vectorized[vid + num_vec * row + pr_step]);
                __builtin_prefetch(&vectorized[vid + num_vec * col + pr_step]);
                vec_inner_prod += vectorized[vid + num_vec * row] * vectorized[vid + num_vec * col];
            }
            double inner_prod = sum_vec(vec_inner_prod);
            result[row * ny + col] = float(inner_prod);
        }
    }
    std::free(vectorized);
}
