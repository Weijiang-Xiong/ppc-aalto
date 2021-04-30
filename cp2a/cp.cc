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


void correlate(int ny, int nx, const float *data, float *result) {
    constexpr float PAD_VALUE = std::numeric_limits<float>::infinity();
}



int main()
{
    const int nx{400}, ny{100};
    float data[nx*ny], result[nx*ny];
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
    
    correlate(ny, nx, data, result);

}
