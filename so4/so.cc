#include <algorithm>
#include <iostream>
#include <omp.h>

#define MIN_LEN 32 // for vectors shorter than this, use stl directly
#define MAX_LEN 100000 // for vectors longer than this, break it down to BRANCH_FACTOR groups
#define BRANCH_FACTOR 8 // it turns out MAX_LEN and BRANCH_FACTOR do not matter ... 

using namespace std;

typedef unsigned long long data_t;

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

// idea from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
// see also https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
int next_power_2 (int x)
{
    x = x - 1;
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16); // for int (up to 32 bit, signed)
    x = x + 1;
    return x;
}

// have a look at http://www.cs.kent.edu/~jbaker/23Workshop/Chesebrough/mergesort/mergesortOMP.cpp
void merge_sort(int vec_len, data_t *data){
    // if the input is small, just call stl (faster than omp), don't set it to a large number, 
    // otherwise the codes will pass the tests but do wrong things in benchmark
    // if (vec_len < MIN_LEN)
    // {
    //     sort(data, data + vec_len);
    //     return;
    // }
    
    int max_thread = omp_get_max_threads();
    int num_thread = next_power_2(max_thread);
    if (num_thread < 2)
    {
        sort(data, data + vec_len);
        return;
    }

    int block_size = vec_len / num_thread;
    if(vec_len % num_thread != 0) block_size++;

    // break down to several parts if the input is too large
    // if (block_size > MAX_LEN)
    // {
    //     for (int k = 0; k < BRANCH_FACTOR; k++)
    //     {
    //         int sub_block_size = block_size / BRANCH_FACTOR;
    //         if(sub_block_size % BRANCH_FACTOR != 0) sub_block_size++;
    //         merge_sort(sub_block_size, data + k * sub_block_size);
    //     }
    // }
    

    #pragma omp parallel num_threads(num_thread)
    {
        int thr_idx = omp_get_thread_num();
        // sort(data + thr_idx * block_size, data + min(next_thr_idx * block_size, vec_len));
        sort(data + min(thr_idx * block_size, vec_len), data + min((thr_idx + 1) * block_size, vec_len));
    }

    // we have 8 threads at first, then merge the threads!
    num_thread /=2;
    while(num_thread > 0)
    {   
        // half the threads than before, each new threads will merge the results from 2 previous thread
        #pragma omp parallel num_threads(num_thread)
        {
            // thread number 0, 1, 2, 3 => 0, 2, 4, 6
            int thr_idx = omp_get_thread_num() * 2;
            // have a look at https://www.cplusplus.com/reference/algorithm/inplace_merge/
            // new thread 0 merges the work by last thread 0 and last thread 1
            // new thread 1 merges the work by last thread 2 and last thread 3 and so on
            inplace_merge(data + min(thr_idx * block_size, vec_len),  // last thread 2 starts here
                          data + min((thr_idx+1) * block_size, vec_len),  // last thread 3 starts here
                          data + min((thr_idx+2) * block_size, vec_len)); // last thread 3 ends here
        }
    block_size *= 2;
    num_thread /= 2;
    };
}

void psort(int n, data_t* data) 
{
    merge_sort(n, data);
}