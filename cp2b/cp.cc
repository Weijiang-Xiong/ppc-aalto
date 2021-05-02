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

void correlate(int ny, int nx, const float *data, float *result)
{
    const int num_group = 4;            // divide a row into some groups
    const int outlier = nx % num_group; //the trailing elements that do not make a full group

    std::vector<double> normalized(nx * ny);
    // double normalized[nx * ny]; // can not use normalized[4000*2000] ... don't know why
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
    }
    // Let X be the normalized input matrix.
    // Calculate the (upper triangle of the) matrix product Y = XX.T.
    // row id in XX.T
    #pragma omp parallel for // need -fopenmp as compiler input
    for (int row = 0; row < ny; row++)
    {
        // column id in XX.T
        #pragma omp parallel for // need -fopenmp as compiler input
        for (int col = row; col < ny; col++)
        {
            // vector product of each element
            // actually the row th row in X times the col th row in X
            // Speed up the inner product
            // split the vector into nx / num_group blocks, each block contains num_group items
            // the gid-th item belongs to the gid-th group, and sum the product by group
            double inner_prod{0};
            double group_sum[num_group]{0};

            for (int bid = 0; bid < nx / num_group; bid++) // block id
            {
                for (int gid = 0; gid < num_group; gid++) // group id
                {
                    group_sum[gid] += normalized[bid * num_group + gid + row * nx] * normalized[bid * num_group + gid + col * nx];
                }
            }
            // now sum up the trailing items
            for (int tid = 0; tid < outlier; tid++)
            {
                inner_prod += normalized[(nx - outlier) + tid + row * nx] * normalized[(nx - outlier) + tid + col * nx];
            }
            // sum up the group sums
            for (auto &sum : group_sum)
            {
                inner_prod += sum;
            }
            
            result[row * ny + col] = float(inner_prod);
        }
    }
}


int main()
{
    const int nx{400}, ny{100};
    float data[nx*ny], result[nx*ny];
    float data1[4] = {1.7, 2.5, 3.3, 4.9};
    float result1[4] = {0.0, 0.0, 0.0, 0.0};
    // std::vector<double> test(100, 6.6); size 100, default value 6.6
    std::default_random_engine rand_eng;
    std::normal_distribution<double> rand_dist(0.0, 1.0);
    for (int i = 0; i < ny; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            // data[nx*i + j] = rand_dist(rand_eng);
            data[nx*i+j]= 1.0;
        }
    }
    // result1 should be [1 1 0 1]
    correlate(2, 2, data1, result1);
    
    correlate(ny, nx, data, result); 
    return 0;
}