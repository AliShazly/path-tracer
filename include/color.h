#ifndef COLOR_H
#define COLOR_H

#include <stdint.h>

#define COL_NCHANNELS 3
typedef float fcolor_t[COL_NCHANNELS];
typedef uint8_t color_t[COL_NCHANNELS];

void fcolor_scale(fcolor_t color, double scalar);
void fcolor_scale_inv(fcolor_t color, double scalar);
void fcolor_offset(fcolor_t color, double offset);
void fcolor_add(fcolor_t ret, fcolor_t c1, fcolor_t c2);
void fcolor_sub(fcolor_t ret, fcolor_t c1, fcolor_t c2);
void fcolor_mul(fcolor_t ret, fcolor_t c1, fcolor_t c2);
void fcolor_divide(fcolor_t ret, fcolor_t c1, fcolor_t c2);

#endif
