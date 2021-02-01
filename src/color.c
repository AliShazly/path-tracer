#include "color.h"
#include <stdint.h>

void fcolor_scale(fcolor_t ret, const fcolor_t color, const double scalar)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        ret[i] = color[i] * scalar;
}

void fcolor_scale_inv(fcolor_t ret, const fcolor_t color, const double scalar)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        ret[i] = color[i] / scalar;
}

void fcolor_offset(fcolor_t ret, const fcolor_t color, const double offset)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        ret[i] = color[i] + offset;
}

void fcolor_add(fcolor_t ret, const fcolor_t c1, const fcolor_t c2)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        ret[i] = c1[i] + c2[i];
}

void fcolor_sub(fcolor_t ret, const fcolor_t c1, const fcolor_t c2)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        ret[i] = c1[i] - c2[i];
}

void fcolor_mul(fcolor_t ret, const fcolor_t c1, const fcolor_t c2)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        ret[i] = c1[i] * c2[i];
}

void fcolor_divide(fcolor_t ret, const fcolor_t c1, const fcolor_t c2)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        ret[i] = c1[i] / c2[i];
}

float fcolor_sum(const fcolor_t c1)
{
    float sum = 0;
    for (int i = 0; i < COL_NCHANNELS; i++)
        sum += c1[i];
    return sum;
}

void fcolor_norm(fcolor_t ret, const fcolor_t c1,
        const float min, const float max, const float scalar)
{
    float delta = max - min;
    for (int i = 0; i < COL_NCHANNELS; i++)
    {
        ret[i] = ((c1[i] - min) / delta) * scalar;
    }
}
