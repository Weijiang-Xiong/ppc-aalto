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
#include <iostream>
#include <chrono>
#include <ctime>
#include <tuple>
#include <algorithm>
#include <immintrin.h>

// typedef float float8_t __attribute__((vector_size(8 * sizeof(float))));
typedef double double4_t __attribute__((vector_size(4 * sizeof(double))));

// constexpr float PAD_VALUE_F = 0.0;
constexpr double PAD_VALUE_D = 0.0;

// constexpr float8_t f8zero{
//     PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F,
//     PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F, PAD_VALUE_F};

constexpr double4_t d4zero{
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

static inline double sum_vec(double4_t vv)
{
    double v = 0.0;
    for (int i = 0; i < 4; ++i)
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
    // int na=10;
    // std::vector<std::tuple<int,int,int>> rows(na*na);

    // for (int ia = 0; ia < na; ++ia) {
    //     for (int ja = 0; ja < na; ++ja) {
    //         int ija = _pdep_u32(ia, 0x55555555) | _pdep_u32(ja, 0xAAAAAAAA);
    //         rows[ia*na + ja] = std::make_tuple(ija, ia, ja);
    //     }
    // }
    // std::sort(rows.begin(), rows.end());
    
    // for (auto &&t : rows)
    // {
    //     std::cout << "(" << std::get<0>(t) << ", " << std::get<1>(t) << ", " << std::get<2>(t) << ")\n";
    // }
    
    const int cell_sz = 8;                            // cell size, divide the matrix X @ X.T into cells of kxk, say 4x4
    const int vec_len = 4;                            // number of elements in a vector
    // const int pr_step = 3;                           // prefetch step, did not work 
    const int num_cel = (ny + cell_sz - 1) / cell_sz; // number of kxk cells per column, in x @ X.T
    const int num_vec = (nx + vec_len - 1) / vec_len; // number of vector per row, padded
    const int pad_row = cell_sz * num_cel;            // number of rows after padding 
    
    // copy the original data into a vectorized matrix
    double4_t *vectorized = double4_alloc(pad_row * num_vec);
    // keep the sequential normalization for now
    std::vector<double> normalized(nx * ny);
    // normalization only accounts for a small fraction of total runtime
    // auto start = std::chrono::system_clock::now();
    #pragma omp parallel for schedule(static,1)// need -fopenmp as compiler input
    for (int y = 0; y < ny; y++)
    {
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
    // for those rows in the padded matrix that are outside the original matrix
    #pragma omp parallel for schedule(static,1)// need -fopenmp as compiler input
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

    // double4_t *zero_block = double4_alloc(cell_sz * cell_sz);      // intermediate vector representation
    // for (int blk_id = 0; blk_id < cell_sz * cell_sz; blk_id++)
    // {
    //     zero_block[blk_id] = d4zero;
    // }

    // Let X be the normalized input matrix.
    // Calculate the (upper triangle of the) matrix product Y = XX.T.
    // row id in XX.T
    #pragma omp parallel for schedule(static,1) // need -fopenmp as compiler input
    for (int row = 0; row < num_cel; row++)
    {
        // column id in XX.T
        double4_t row_vecs[cell_sz], col_vecs[cell_sz]; //
        for (int col = row; col < num_cel; col++)
        {
            // vector product of each element
            // actually the row th row in X times the col th row in X
            // vectorize the inner product
            // split the vector into nx / vec_len blocks, each block contains vec_len items
            // the gid-th item belongs to the gid-th group, and sum the product by group
            // double block_sumed[cell_sz * cell_sz];
            double4_t block_inner_prod[cell_sz * cell_sz];      // intermediate vector representation
            // block_inner_prod = zero_block;
            for (int blk_id = 0; blk_id < cell_sz * cell_sz; blk_id++)
            {
                block_inner_prod[blk_id] = d4zero;
            }
            for (int vid = 0; vid < num_vec; vid++) // block id
            {   
                // get the 2*cell_sz vectors needed for this block 
                for (int blk_idx = 0; blk_idx < cell_sz; blk_idx++)
                {   
                    int real_row_id = cell_sz*row+blk_idx;
                    int real_col_id = cell_sz*col+blk_idx;
                    // __builtin_prefetch(&vectorized[(real_row_id+pr_step)*num_vec + vid]);
                    // __builtin_prefetch(&vectorized[(real_col_id+pr_step)*num_vec + vid]);
                    row_vecs[blk_idx] = vectorized[real_row_id*num_vec + vid];
                    col_vecs[blk_idx] = vectorized[real_col_id*num_vec + vid];
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
                        result[real_row_id * ny + real_col_id]= sum_vec(block_inner_prod[row_idx*cell_sz + col_idx]);
                    }
                }
            }
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