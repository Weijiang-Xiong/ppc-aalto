#include <algorithm>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <vector>
#include <random>

#define MIN_LEN 100000
#define MEDIAN_CAND 7 // choose an prime number, much much smaller than MIN_LEN

typedef unsigned long long data_t;

// void swap(data_t *a, data_t *b){
//     auto t = *a;
//     *a = *b;
//     *b = t;
// }

// int partition(data_t *data, int low, int high, data_t pivot)
// {
//     // just use the last element as pivot
//     int i = low-1;
//     // #pragma omp parallel for
//     for (int j = low; j < high; j++)
//     {
//         if (data[j] < pivot)
//         {
//             i++;
//             swap(&data[i], &data[j]);
//         }
//     }
//     // put the pivot right after those smaller than it
//     swap(&data[i+1], &data[high]); 
//     return i+1; // return the index of the pivot
// }

data_t choose_pivot(data_t *data_low, data_t *data_high)
{
    unsigned int seed = omp_get_thread_num();
    std::vector<data_t> v(MEDIAN_CAND);
    int step = (data_high - data_low)/MEDIAN_CAND;
    for (int i = 0; i < MEDIAN_CAND; i++)
    {
        // v[i] = *(data_low + step*i + std::rand()%step);
        // https://stackoverflow.com/questions/4234480/concurrent-random-number-generation/4234555
        v[i] = *(data_low + step*i + rand_r(&seed)%step);
        // v[i] = *(data_low + rand_r(&seed)%(data_high - data_low));
    }
    // https://en.cppreference.com/w/cpp/algorithm/nth_element
    std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    return v[v.size()/2];
}

void quick_sort(data_t *data_low, data_t *data_high)
{
    if (data_high - data_low <= MIN_LEN)
    {
        // std::cout << "using stl to sort short array"<< std::endl;
        // the last index is `high`, but std::sort need length, so that will be high+1
        std::sort(data_low, data_high+1); 
        return;
    }

    if (data_low < data_high)
    {
        // quick sort example here 
        // https://en.cppreference.com/w/cpp/algorithm/partition
        data_t pivot = choose_pivot(data_low, data_high);
        data_t* pi1 = std::partition(data_low, data_high+1, [pivot](data_t em){return em < pivot;});
        // no need to recursively sort elements that are equal to pivot
        data_t* pi2 = std::partition(pi1, data_high+1, [pivot](data_t em){return em == pivot;});
        #pragma omp task
        {
            quick_sort(data_low, pi1);
        }
        #pragma omp task
        {
            quick_sort(pi2, data_high);
        }
    }
}

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of quicksort.
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            quick_sort(data+0, data+n-1);
        }
    }
}
