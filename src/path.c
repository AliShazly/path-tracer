#include "linmath_d.h"
#include "list.h"
#include "obj_parser.h"
#include "color.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MAX_LINE_SIZE 256
#define TRI_NPTS 3
#define N_MESHES 8
#define FOV_RAD (1.0472)
#define RAY_BUMP_AMT (0.001)
#define ROWS 256
#define COLS 256
#define MAX_DEPTH 64
#define N_SAMPLES 500
#define CAM_X 0
#define CAM_Y 0
#define CAM_Z 0

typedef struct Ray
{
    vec3 origin;
    vec3 direction;
}Ray;

// assuming lambertian
typedef struct Material
{
    fcolor_t color;
    float reflectance;
    float emittance;
    char *name;
}Material;

typedef struct Mesh
{
    vec3 *verts;
    vec3 *normals;
    size_t size;
    Material material;
}Mesh;

typedef struct Framebuffer
{
    size_t rows;
    size_t cols;
    fcolor_t *buffer;
}Framebuffer;

void print_progress(char* msg, int len, float current, float max);
void print_col(fcolor_t c);
int clamp(int val, int min, int max);
void uniform_sample(vec2 out_offset, int n_samples, int sample_iter);
int coord_to_idx(int x, int y, int rows, int cols);
double frand(double low,double high);
double distance(vec3 a, vec3 b);
void triangle_normal(vec3 dst, vec3 a, vec3 b, vec3 c);
bool point_in_circle(vec2 pt, vec2 center, int radius);
Ray gen_camera_ray(vec2 pixel, const Framebuffer *fb);
void cosine_sample_hemisphere(vec3 ret, vec3 normal);
bool ray_triangle_intersect(Ray ray, vec3 v0, vec3 v1, vec3 v2, vec3 out_inter_pt);
bool cast_ray(Ray ray, Mesh meshes[N_MESHES], vec3 out_hit, vec3 out_norm, Material **out_hit_mat);
void trace_path(fcolor_t out_color, Ray ray, unsigned int depth, Mesh meshes[N_MESHES]);
void read_meshes(Mesh out_mesh_arr[N_MESHES], const char *mesh_list_path);
void free_meshes(Mesh mesh_arr[N_MESHES]);

int main(void)
{
    srand(time(NULL));
    Mesh *cornell_meshes = malloc(sizeof(Mesh) * N_MESHES);
    read_meshes(cornell_meshes, "./objs.txt");

    const fcolor_t p_white = {1.0f, 1.0f, 1.0f};
    const fcolor_t white   = {0.7f, 0.7f, 0.7f};
    const fcolor_t green   = {0, 1.0f, 0};
    const fcolor_t red     = {1.0f, 0, 0};

    const float r = 0.5;
    Material right_mat = {.emittance = 0, .reflectance = r, .name = "RightWall"};
    Material left_mat  = {.emittance = 0, .reflectance = r, .name = "LeftWall"};
    Material back_mat  = {.emittance = 0, .reflectance = r, .name = "BackWall"};
    Material top_mat   = {.emittance = 0, .reflectance = r, .name = "Cieling"};
    Material floor_mat = {.emittance = 0, .reflectance = r, .name = "Floor"};
    Material obj1_mat  = {.emittance = 0, .reflectance = r, .name = "FirstObject"};
    Material obj2_mat  = {.emittance = 0, .reflectance = r, .name = "SecondObject"};
    Material light_mat = {.emittance = 9001, .reflectance = 0, .name = "Light"};

    memcpy(right_mat.color, red, sizeof(fcolor_t));
    memcpy(left_mat.color, green, sizeof(fcolor_t));
    memcpy(back_mat.color, white, sizeof(fcolor_t));
    memcpy(top_mat.color, white, sizeof(fcolor_t));
    memcpy(floor_mat.color, white, sizeof(fcolor_t));
    memcpy(obj1_mat.color, white, sizeof(fcolor_t));
    memcpy(obj2_mat.color, white, sizeof(fcolor_t));
    memcpy(light_mat.color, p_white, sizeof(fcolor_t));

    // ordering is determined in ./objs.txt
    cornell_meshes[0].material = right_mat;
    cornell_meshes[1].material = light_mat;
    cornell_meshes[2].material = left_mat;
    cornell_meshes[3].material = obj1_mat;
    cornell_meshes[4].material = floor_mat;
    cornell_meshes[5].material = top_mat;
    cornell_meshes[6].material = obj2_mat;
    cornell_meshes[7].material = back_mat;

    Framebuffer fb;
    fb.rows = ROWS;
    fb.cols = COLS;
    const int fb_size = fb.rows * fb.cols;
    fb.buffer = malloc(sizeof(fcolor_t) * fb_size);

    omp_lock_t writelock;
    omp_init_lock(&writelock);

    time_t begin = time(NULL);

    int loop_count = 0;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < fb.rows; i++)
    {
        for (int j = 0; j < fb.cols; j++)
        {
            fcolor_t color_avg = {0, 0, 0};
            for (int k = 0; k < N_SAMPLES; k++)
            {
                vec2 pixel = {j, i};
                vec2 pix_sample_offset;
                fcolor_t out_color;

                uniform_sample(pix_sample_offset, N_SAMPLES, k);

                pixel[0] += pix_sample_offset[0];
                pixel[1] += pix_sample_offset[1];

                Ray ray = gen_camera_ray(pixel, &fb);
                trace_path(out_color, ray, 0, cornell_meshes);

                fcolor_add(color_avg, color_avg, out_color);
            }
            fcolor_scale_inv(color_avg, color_avg, N_SAMPLES);
            for (int i = 0; i < COL_NCHANNELS; i++)
            {
                color_avg[i]= clamp(color_avg[i], 0, 255);
            }

            memcpy(fb.buffer[coord_to_idx(j, i, fb.rows, fb.cols)], color_avg, sizeof(fcolor_t));

            omp_set_lock(&writelock);
            print_progress("Rendering... ", 50, loop_count++, fb_size);
            omp_unset_lock(&writelock);
        }
    }
    omp_destroy_lock(&writelock);

    time_t end = time(NULL);
    printf("\nTime elapsed: %zu seconds.\n", (end - begin));

    color_t *write_buf = malloc(fb.rows * fb.cols * sizeof(color_t));
    for (int i = 0; i < fb.rows * fb.cols; i++)
    {
        for (int j = 0; j < COL_NCHANNELS; j++)
        {
            write_buf[i][j] = (uint8_t)(fb.buffer[i][j]);
        }
    }
    stbi_write_jpg("./out.jpg",fb.cols,fb.rows,COL_NCHANNELS,write_buf,100);

    free_meshes(cornell_meshes);
    free(fb.buffer);
    free(write_buf);
}

void print_progress(char* msg, int len, float current, float max)
{
    const float progress_amt = current/max;
    const int n_prog_bars = len * progress_amt;

    printf("\r%s",msg);
    printf("%.2f%% ", progress_amt * 100);
    putchar('[');
    for (int i = 0; i < len; i++)
    {
        if ((i - 1) < n_prog_bars)
            putchar('=');
        else
            putchar(' ');
    }
    putchar(']');
    fflush(stdout);
}

void print_col(fcolor_t c)
{
    printf("%f, %f, %f\n", c[0], c[1], c[2]);
}

int clamp(int val, int min, int max)
{
    if (val > max)
        return max;
    else if (val < min)
        return min;
    return val;
}

void uniform_sample(vec2 out_offset, int n_samples, int sample_iter)
{
    const double step = 1. / n_samples;
    out_offset[0] = step * ((sample_iter % n_samples) + 1);
    out_offset[1] = step * (floor(sample_iter /(double) n_samples) + 1);
}

int coord_to_idx(int x, int y, int rows, int cols)
{
    // flipping vertically
    int row = (rows - 1) - y;
    // 2D idx (row, col) to a 1D index
    return row * cols + x;
}

double frand(double low,double high)
{
    return (rand()/(double)(RAND_MAX))*fabs(low-high)+low;
}

double distance(vec3 a, vec3 b)
{
    return sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2) + pow(b[2] - a[2], 2));
}

void triangle_normal(vec3 dst, vec3 a, vec3 b, vec3 c)
{
    vec3 v;
    vec3 w;
    vec3_sub(v, c, a);
    vec3_sub(w, b, a);
    vec3_mul_cross(dst, v, w);
    vec3_norm(dst, dst);
}

bool point_in_circle(vec2 pt, vec2 center, int radius)
{
    // no need for a square root, square the radius for comparison instead
    return (pow((center[0] - pt[0]), 2) + pow((center[1] - pt[1]), 2)) < pow(radius, 2);
}

// switch rows/cols for a fb index as a pixel, and you're in raster space
Ray gen_camera_ray(vec2 pixel, const Framebuffer *fb)
{
    const double image_aspect = fb->rows / (double) fb->cols;

    const vec2 pixel_screen = {
        pixel[0] / fb->cols,
        1 - (pixel[1] / fb->rows)
    };

    Ray ray;
    ray.origin[0] = CAM_X;
    ray.origin[1] = CAM_Y;
    ray.origin[2] = CAM_Z;

    double t = tan(FOV_RAD / 2);
    ray.direction[0] = (2 * pixel_screen[0] - 1) * image_aspect * t;
    ray.direction[1] = (1 - 2 * pixel_screen[1]) * t;
    ray.direction[2] = -1;

    vec3_norm(ray.direction, ray.direction);
    return ray;
}

// http://www.kevinbeason.com/smallpt/
void cosine_sample_hemisphere(vec3 ret, vec3 normal)
{
    float r1 = 2.0f * M_PI * frand(0, 1);
    float r2 = frand(0, 1);
    float r2s = sqrt(r2);

    vec3 u;
    if (fabs(normal[0]) > 0.1f)
    {
        vec3 c = {0, 1, 0};
        vec3_mul_cross(u, c, normal);
    }
    else
    {
        vec3 c = {1, 0, 0};
        vec3_mul_cross(u, c, normal);
    }

    vec3_norm(u, u);
    vec3 v;
    vec3_mul_cross(v, normal, u);

    vec3 s_u, s_v, s_w;
    vec3_scale(s_u, u, cos(r1));
    vec3_scale(s_u, s_u, r2s);

    vec3_scale(s_v, v, sin(r1));
    vec3_scale(s_v, s_v, r2s);

    vec3_scale(s_w, normal, sqrt(1 - r2));
    vec3_add(ret, s_u, s_v);
    vec3_add(ret, ret, s_w);
    vec3_norm(ret, ret);
}

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
// shamelessly copied, i'll understand it soon
bool ray_triangle_intersect(Ray ray, vec3 v0, vec3 v1, vec3 v2, vec3 out_inter_pt)
{
    vec3 edge1, edge2, h, s, q;
    double a,f,u,v;
    vec3_sub(edge1, v1, v0);
    vec3_sub(edge2, v2, v0);
    vec3_mul_cross(h, ray.direction, edge2);
    a = vec3_mul_inner(edge1, h);
    if (a > -DBL_EPSILON && a < DBL_EPSILON)
        return false;    // This ray is parallel to this triangle.
    f = 1.0/a;
    vec3_sub(s, ray.origin, v0);
    u = f * vec3_mul_inner(s, h);
    if (u < 0.0 || u > 1.0)
        return false;
    vec3_mul_cross(q, s, edge1);
    v = f * vec3_mul_inner(ray.direction, q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * vec3_mul_inner(edge2, q);
    if (t > DBL_EPSILON) // ray intersection
    {
        vec3 ss;
        vec3_scale(ss, ray.direction, t);
        vec3_add(out_inter_pt, ray.origin, ss);
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}

bool cast_ray(Ray ray, Mesh meshes[N_MESHES], vec3 out_hit, vec3 out_norm, Material **out_hit_mat)
{
    list inter_pts;
    list_init(&inter_pts, sizeof(vec3), 10);

    list mesh_idxs;
    list_init(&mesh_idxs, sizeof(int), 10);

    list vert_idxs;
    list_init(&vert_idxs, sizeof(int), 10);

    vec3 intersection;
    for (int i = 0; i < N_MESHES; i++)
    {
        for (int j = 0; j < meshes[i].size; j+=TRI_NPTS)
        {
            bool ret = ray_triangle_intersect(ray, meshes[i].verts[j], meshes[i].verts[j+1], meshes[i].verts[j+2],
                    intersection);
            if (ret)
            {
                list_append(&inter_pts, intersection);
                list_append(&mesh_idxs, &i);
                list_append(&vert_idxs, &j);
            }
        }
    }

    if (inter_pts.used == 0)
    {
        list_free(&inter_pts);
        list_free(&mesh_idxs);
        list_free(&vert_idxs);
        return false;
    }

    double closest_dist = DBL_MAX;
    int closest_idx = -1;
    for (int i = 0; i < inter_pts.used; i++)
    {
        vec3 *inter_pt = list_index(&inter_pts, i);
        double dist = distance(ray.origin, *inter_pt);
        if (dist < closest_dist)
        {
            closest_dist = dist;
            closest_idx = i;
        }
    }

    assert(closest_idx != -1);

    vec3 *closest_pt = list_index(&inter_pts, closest_idx);
    int closest_mesh_idx = *((int *)list_index(&mesh_idxs, closest_idx));
    int closest_vert_idx = *((int *)list_index(&vert_idxs, closest_idx));
    Mesh *closest_mesh = &meshes[closest_mesh_idx];

    triangle_normal(out_norm,
                   (*closest_mesh).verts[closest_vert_idx + 2],
                   (*closest_mesh).verts[closest_vert_idx + 1],
                   (*closest_mesh).verts[closest_vert_idx + 0]);

    memcpy(out_hit, *closest_pt, sizeof(vec3));
    *out_hit_mat = &(closest_mesh->material);

    // moving the hit point a small amount towards the vector to avoid a 2nd collision
    vec3 normal_small_scale;
    vec3_scale(normal_small_scale, out_norm, RAY_BUMP_AMT);
    vec3_add(out_hit, out_hit, normal_small_scale);

    list_free(&inter_pts);
    list_free(&mesh_idxs);
    list_free(&vert_idxs);
    return true;
}

void trace_path(fcolor_t out_color, Ray ray, unsigned int depth, Mesh meshes[N_MESHES])
{
    memset(out_color, 0, sizeof(fcolor_t));

    if (depth >= MAX_DEPTH)
        return;

    vec3 ray_hit_pt;
    vec3 ray_hit_norm;
    Material *ray_hit_mat;
    bool ray_hit = cast_ray(ray, meshes, ray_hit_pt, ray_hit_norm, &ray_hit_mat);
    if (!ray_hit)
        return;

    Ray new_ray;
    memcpy(new_ray.origin, ray_hit_pt, sizeof(vec3));
    cosine_sample_hemisphere(new_ray.direction, ray_hit_norm);

    fcolor_t incoming;
    trace_path(incoming, new_ray, depth + 1, meshes);

    // https://i.redd.it/802mndge03t01.png
    // to get light towards eye
    // a = light emitted from pt (material emit)
    // b = all the light coming into the point from the unit hemisphere samples (the recursive output)
    // c = BRDF - chances of such light rays bouncing towards the eye
    // d = irradiance factor over the normal at the point (ray_normal_dot)
    // ((a + b) * c) * d

    const float pdf = 1 / (2 * M_PI);
    const float emitted_light = ray_hit_mat->emittance;
    const float ray_normal_dot = vec3_mul_inner(new_ray.direction, ray_hit_norm);
    const fcolor_t *surf_col = &(ray_hit_mat->color);

    const float precomp = ((ray_hit_mat->reflectance / M_PI) * ray_normal_dot) / pdf;
    const float col_r = emitted_light + (((incoming[0]) * (*surf_col)[0]) * precomp);
    const float col_g = emitted_light + (((incoming[1]) * (*surf_col)[1]) * precomp);
    const float col_b = emitted_light + (((incoming[2]) * (*surf_col)[2]) * precomp);

    out_color[0] = col_r;
    out_color[1] = col_g;
    out_color[2] = col_b;
}

void read_meshes(Mesh out_mesh_arr[N_MESHES], const char *mesh_list_path)
{
    FILE *fp = fopen(mesh_list_path, "r");
    assert(fp != NULL);

    // tmp pointer for malloc in parse_obj
    vec2 *tmp_texcoords;

    int i = 0;
    char line_buf[MAX_LINE_SIZE];
    while(fgets(line_buf, sizeof(line_buf), fp))
    {
        Mesh *mesh = &out_mesh_arr[i++];

        // stripping newline
        line_buf[strlen(line_buf)-1] = '\0';

        parse_obj(line_buf, &(mesh->size), &(mesh->verts), &tmp_texcoords, &(mesh->normals));
        free(tmp_texcoords);
    }
    fclose(fp);
}

void free_meshes(Mesh mesh_arr[N_MESHES])
{
    for (int i = 0; i < N_MESHES; i++)
    {
        free(mesh_arr[i].verts);
        free(mesh_arr[i].normals);
    }
    free(mesh_arr);
}
