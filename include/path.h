#ifndef PATH_HEADER
#define PATH_HEADER

#include <stddef.h>
#include "color.h"

typedef struct
{
    size_t rows;
    size_t cols;
    fcolor_t *buffer;
}Framebuffer;

#endif
