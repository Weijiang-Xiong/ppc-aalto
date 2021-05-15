/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>

void mf(int ny, int nx, int hy, int hx, const float *in, float *out)
{
	// replace a pixel with the average value in a small window
	for (int y = 0; y < ny; y++)
	{
		for (int x = 0; x < nx; x++)
		{
			int a_start = std::max(x - hx, 0);
			int a_end = std::min(x + hx + 1, nx);
			int b_start = std::max(y - hy, 0);
			int b_end = std::min(y + hy + 1, ny);

			// use a vector container, learned from cp1 assignment ... do not use array...
			std::vector<double> window((a_end - a_start) * (b_end - b_start));
			window.clear(); // Erases all the elements. 

			for (int b = b_start; b < b_end; b++)
			{
				for (int a = a_start; a < a_end; a++)
				{
					window.push_back(in[a + b * nx]);
				}
			}

			int median_idx = window.size() / 2; // the k+1 th element no matter n = 2k or n=2k+1
			std::nth_element(window.begin(), window.begin() + median_idx, window.end());
			double median_cand = window[median_idx];

			if (window.size() % 2 == 0) // even number n = 2k, median_idx=k, corresponds to k+1 th element
			{
				std::nth_element(window.begin(), window.begin() + median_idx - 1, window.end());
				out[x + y * nx] = 0.5 * (window[median_idx - 1] + median_cand);
			}
			else // odd number 2k+1, median_idx=k
			{
				out[x + y * nx] = median_cand;
			}
		}
	}
}
