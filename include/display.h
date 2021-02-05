#ifndef DISPLAY_H
#define DISPLAY_H

#include "path.h"
#include "color.h"
#include <MiniFB.h>
#include <aio.h>

void display_buffer(Framebuffer *fb, pid_t parent_pid, int pipe_write);

#endif
