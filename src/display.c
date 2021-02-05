#include "path.h"
#include "display.h"
#include "color.h"
#include <MiniFB.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <ctype.h>

static void update_ui32_buffer(Framebuffer *fb, uint32_t* buffer);
void resize(struct mfb_window *window, int width, int height);
void keyboard(struct mfb_window *window, mfb_key key, mfb_key_mod mod, bool is_pressed);

typedef struct{
    int cols;
    int rows;
    int pipe_write;
    pid_t parent_pid;
}GlobalData;

void display_buffer(Framebuffer *fb, pid_t parent_pid, int pipe_write)
{
    struct mfb_window *window = mfb_open_ex("Path Tracer",
            fb->cols, fb->rows, WF_RESIZABLE);
    if (!window)
        exit(EXIT_FAILURE);

    GlobalData data = {
        .cols = fb->cols,
        .rows = fb->rows,
        .pipe_write=pipe_write,
        .parent_pid=parent_pid};
    mfb_set_user_data(window, &data);

    mfb_set_resize_callback(window, resize);
    mfb_set_keyboard_callback(window, keyboard);

    uint32_t *ui32_buffer = malloc(fb->rows * fb->cols * sizeof(*ui32_buffer));

    do {
        int state;

        update_ui32_buffer(fb, ui32_buffer);

        state = mfb_update_ex(window, ui32_buffer, fb->cols, fb->rows);

        if (state < 0)
        {
            window = NULL;
            break;
        }
    } while(mfb_wait_sync(window));
}

static void update_ui32_buffer(Framebuffer *fb, uint32_t* buffer)
{
    for (int i = 0; i < fb->rows * fb->cols; i++)
        buffer[i] = MFB_RGB(fb->buffer[i][0], fb->buffer[i][1], fb->buffer[i][2]);
}

void resize(struct mfb_window *window, int width, int height)
{
    GlobalData *data = mfb_get_user_data(window);

    double ratio_x = width / (double) data->cols;
    double ratio_y = height / (double) data->rows;
    double ratio = ratio_x < ratio_y ? ratio_x : ratio_y;

    int new_width = data->cols * ratio;
    int new_height = data->rows * ratio;
    mfb_set_viewport(window, 0, 0, new_width, new_height);
}

void keyboard(struct mfb_window *window, mfb_key key, mfb_key_mod mod, bool is_pressed)
{
    if (key == KB_KEY_ESCAPE)
        mfb_close(window);

    if (!is_pressed)
        return;

    GlobalData *data = mfb_get_user_data(window);

    char keypress = tolower((char)key);
    int ret = write(data->pipe_write, &keypress, sizeof(char));
    assert(ret > 0);

    kill(data->parent_pid, SIGUSR1);
}

