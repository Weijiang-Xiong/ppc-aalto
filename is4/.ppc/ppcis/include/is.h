#pragma once

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

Result segment(int ny, int nx, const float *data);
