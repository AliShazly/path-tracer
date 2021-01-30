#include <color.h>
#include <stdint.h>

void fcolor_scale(fcolor_t color, double scalar)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        color[i] *= scalar;
}

void fcolor_scale_inv(fcolor_t color, double scalar)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        color[i] /= scalar;
}

void fcolor_offset(fcolor_t color, double offset)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
        color[i] += offset;
}

void fcolor_add(fcolor_t ret, fcolor_t c1, fcolor_t c2)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
    {
        ret[i] = c1[i] + c2[i];
    }
}

void fcolor_sub(fcolor_t ret, fcolor_t c1, fcolor_t c2)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
    {
        ret[i] = c1[i] - c2[i];
    }
}

void fcolor_mul(fcolor_t ret, fcolor_t c1, fcolor_t c2)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
    {
        ret[i] = c1[i] * c2[i];
    }
}

void fcolor_divide(fcolor_t ret, fcolor_t c1, fcolor_t c2)
{
    for (int i = 0; i < COL_NCHANNELS; i++)
    {
        ret[i] = c1[i] * c2[i];
    }
}
