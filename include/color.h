#ifndef COLOR_H
#define COLOR_H

#include <stdint.h>

#define COL_NCHANNELS 3

typedef float fcolor_t[COL_NCHANNELS];
typedef uint8_t color_t[COL_NCHANNELS];

void fcolor_scale(fcolor_t ret, const fcolor_t color, const double scalar);
void fcolor_scale_inv(fcolor_t ret, const fcolor_t color, const double scalar);
void fcolor_offset(fcolor_t ret, const fcolor_t color, const double offset);
void fcolor_add(fcolor_t ret, const fcolor_t c1, const fcolor_t c2);
void fcolor_sub(fcolor_t ret, const fcolor_t c1, const fcolor_t c2);
void fcolor_mul(fcolor_t ret, const fcolor_t c1, const fcolor_t c2);
void fcolor_divide(fcolor_t ret, const fcolor_t c1, const fcolor_t c2);
float fcolor_sum(const fcolor_t c1);
void fcolor_norm(fcolor_t ret, const fcolor_t c1,
        const float min, const float max, const float scalar);

#endif
