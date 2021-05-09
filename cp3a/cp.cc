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

// typedef float float8_t __attribute__((vector_size(8 * sizeof(float))));
typedef double double8_t __attribute__((vector_size(8 * sizeof(double))));

// constexpr float PAD_VALUE_F = 0.0;
constexpr double PAD_VALUE_D = 0.0;

// constexpr float8_t f8zero{
//     PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F,
//     PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F};

constexpr double8_t d8zero{
    PAD_VALUE_D, PAD_VALUE_D, PAD_VALUE_D, PAD_VALUE_D,
    PAD_VALUE_D, PAD_VALUE_D, PAD_VALUE_D, PAD_VALUE_D};

// static inline float sum_vec(float8_t vv)
// {
//     double v = 0.0;
//     for (int i = 0; i < 8; ++i)
//     {
//         v += vv[i];
//     }
//     return v;
// };

static inline double sum_vec(double8_t vv)
{
    double v = 0.0;
    for (int i = 0; i < 8; ++i)
    {
        v += vv[i];
    }
    return v;
};

// static float8_t *float8_alloc(std::size_t n)
// {
//     void *tmp = 0;
//     if (posix_memalign(&tmp, sizeof(float8_t), sizeof(float8_t) * n))
//     {
//         throw std::bad_alloc();
//     }
//     return (float8_t *)tmp;
// };

static double8_t *double8_alloc(std::size_t n)
{
    void *tmp = 0;
    if (posix_memalign(&tmp, sizeof(double8_t), sizeof(double8_t) * n))
    {
        throw std::bad_alloc();
    }
    return (double8_t *)tmp;
};

void correlate(int ny, int nx, const float *data, float *result)
{
    const int cell_sz = 8;                            // cell size, divide the matrix X @ X.T into cells of kxk, say 4x4
    const int vec_len = 8;                            // number of elements in a vector
    const int pr_step = 15;                           // prefetch step
    const int vec_per_group = 125;                    // divide a row of vector into several groups, each contains vec_per_group vectors
    const int num_cel = (ny + cell_sz - 1) / cell_sz; // number of kxk cells per column, in x @ X.T
    const int num_vec_ini = (nx + vec_len - 1) / vec_len; // initial number of vector per row, padded
    const int num_group = (num_vec_ini + vec_per_group -1) / vec_per_group; // how many groups of vector in a row 
    const int num_vec = num_group*vec_per_group; // #vector per row after grouping
    const int pad_row = cell_sz * num_cel;            // number of rows after padding 
    
    // copy the original data into a vectorized matrix
    double8_t *vectorized = double8_alloc(pad_row * num_vec);
    // keep the sequential normalization for now
    std::vector<double> normalized(nx * ny);
    // normalization only accounts for a small fraction of total runtime
    // auto start = std::chrono::system_clock::now();
    #pragma omp parallel for schedule(static,1)// need -fopenmp as compiler input
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
                __builtin_prefetch(&normalized[nx * y + real_idx + pr_step]);
                vectorized[i + num_vec * y][j] = real_idx < nx ? normalized[nx * y + real_idx] : 0.0;
            }
        }
    }

    #pragma omp parallel for schedule(static,1)// need -fopenmp as compiler input
    for (int y = ny; y < pad_row; y++)
    {
        for (int i = 0; i < num_vec; i++)
        {
            vectorized[i + num_vec * y] = d8zero;
        }
    }
    // auto end = std::chrono::system_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end-start;
    // std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    // double8_t *zero_block = double4_alloc(cell_sz * cell_sz);      // intermediate vector representation
    // for (int blk_id = 0; blk_id < cell_sz * cell_sz; blk_id++)
    // {
    //     zero_block[blk_id] = d8zero;
    // }

    // Let X be the normalized input matrix.
    // Calculate the (upper triangle of the) matrix product Y = XX.T.
    // row id in XX.T
    for (int group_idx = 0; group_idx < num_group; group_idx++)
    {
        #pragma omp parallel for schedule(static,1) // need -fopenmp as compiler input
        for (int row = 0; row < num_cel; row++)
        {
            // column id in XX.T
            double8_t row_vecs[cell_sz], col_vecs[cell_sz]; //
            for (int col = row; col < num_cel; col++)
            {
                // vector product of each element
                // actually the row th row in X times the col th row in X
                // vectorize the inner product
                // split the vector into nx / vec_len blocks, each block contains vec_len items
                // the gid-th item belongs to the gid-th group, and sum the product by group
                // double block_sumed[cell_sz * cell_sz];
                double8_t block_inner_prod[cell_sz * cell_sz];      // intermediate vector representation
                // block_inner_prod = zero_block;
                for (int blk_id = 0; blk_id < cell_sz * cell_sz; blk_id++)
                {
                    block_inner_prod[blk_id] = d8zero;
                }
                for (int vid = 0; vid < vec_per_group; vid++) // which vector in that group
                {   
                    // get the 2*cell_sz vectors needed for this block 
                    for (int blk_idx = 0; blk_idx < cell_sz; blk_idx++)
                    {   
                        int real_row_id = cell_sz*row+blk_idx;
                        int real_col_id = cell_sz*col+blk_idx;
                        // __builtin_prefetch(&vectorized[(real_row_id + 4)*num_vec + vid]);
                        // __builtin_prefetch(&vectorized[(real_row_id + 4)*num_vec + vid]);
                        row_vecs[blk_idx] = vectorized[real_row_id*num_vec + group_idx*vec_per_group + vid];
                        col_vecs[blk_idx] = vectorized[real_col_id*num_vec + group_idx*vec_per_group + vid];
                    }
                    
                    for (int row_idx = 0; row_idx < cell_sz; row_idx++)
                    {
                        for (int col_idx = 0; col_idx < cell_sz; col_idx++)
                        {
                        block_inner_prod[row_idx*cell_sz + col_idx] += row_vecs[row_idx] * col_vecs[col_idx];
                        }
                    }
                }
                // sum up and assign values to final result 
                for (int row_idx = 0; row_idx < cell_sz; row_idx++)
                {
                    for (int col_idx = 0; col_idx < cell_sz; col_idx++)
                    {
                        int real_row_id = cell_sz*row+row_idx;
                        int real_col_id = cell_sz*col+col_idx;
                        if(real_row_id<ny && real_col_id<ny)
                        {
                            result[real_row_id * ny + real_col_id] += sum_vec(block_inner_prod[row_idx*cell_sz + col_idx]);
                        }
                    }
                }
            }
        }
    }
    std::free(vectorized);
}
