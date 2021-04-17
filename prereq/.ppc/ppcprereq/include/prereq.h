#pragma once

struct Result {
    float avg[3];
};

Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1);
