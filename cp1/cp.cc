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

void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<double> normalized(nx * ny);
    // double normalized[nx * ny]; // can not use normalized[4000*2000] ... don't know why
    // each row
    for (int y = 0; y < ny; y++)
    {   
        // First normalize the input rows so that each row has the arithmetic mean of 0  be careful to do the normalization so that you do not change pairwise correlations.
        double row_sum = 0.0;
        double row_square_sum = 0.0;
        // iterate over columns
        for (int x = 0; x < nx; x++)
        {   
            row_sum += data[x + y*nx];
        }
        double rwo_avg = row_sum/nx;
        for (int x = 0; x < nx; x++)
        {   
            double item = data[x + y*nx]-rwo_avg;
            normalized[x + y*nx] = item;
            row_square_sum += pow(item, 2);
        }
        // Then normalize the input rows so that for each row the sum of the squares of the elements is 1 — again, be careful to do the normalization so that you do not change pairwise correlations.
        double root_square_sum = sqrt(row_square_sum);
        for (int x = 0; x < nx; x++)
        {
            normalized[x + y*nx] /= root_square_sum;
        }

    }
    // Let X be the normalized input matrix.
    // Calculate the (upper triangle of the) matrix product Y = XX.T.
    // row id in XX.T
    for (int row = 0; row < ny; row++)
    {   
        // column id in XX.T
        for (int col = row; col < ny; col++)
        {   
            // vector product of each element 
            // actually the row th row in X times the col th row in X
            double inner_prod = 0;
            for (int i = 0; i < nx; i++)
            {
                inner_prod +=  normalized[i +row*nx] * normalized[i +col*nx];
            }
            result[row*ny+col] = float(inner_prod);
        }
        
    }

}

// #include "cp.h"
// #include "math.h"
// #include <stdio.h>
// #include <vector>
// #include <numeric>

// void correlate(int ny, int nx, const float *data, float *result)
// {
//     std::vector<double> interm_data(nx * ny);
//     for (int row = 0; row < ny; row++)
//     {
//         double sum = 0;
//         for (int column = 0; column < nx; column++)
//         {
//             sum += data[column + row * nx];
//         }
//         double mean = sum / nx;
//         double square_sum = 0;
//         for (int column = 0; column < nx; column++)
//         {
//             double x = data[column + row * nx] - mean;
//             interm_data[column + row * nx] = x;
//             square_sum += x * x;
//         }
//         square_sum = sqrt(square_sum);
//         for (int column = 0; column < nx; column++)
//         {
//             interm_data[column + row * nx] /= square_sum;
//         }
//     }
//     for (int i = 0; i < ny; i++)
//     {
//         for (int j = i; j < ny; j++)
//         {
//             double sumx = 0;
//             for (int k = 0; k < nx; k++)
//             {
//                 sumx += interm_data[k + i * nx] * interm_data[k + j * nx];
//             }
//             result[j + i * ny] = (float)sumx;
//         }
//     }
// }
