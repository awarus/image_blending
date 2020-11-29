#include <stdio.h>
#include <float.h>

double convolve2d_fl(int part_h, int part_w, float part[][part_w], float kern[][part_w]);

double convolve2d_fl(int part_h, int part_w, float part[][part_w], float kern[][part_w]){
    double sum = 0.0;

    for(int i = 0; i < part_h; ++i)
        for(int j = 0; j < part_w; ++j)
            sum += part[i][j] * kern[i][j];

    return sum;
}