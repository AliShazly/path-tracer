#include "linmath_d.h"
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
#include <wait.h>
#include <fcntl.h>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MAX_LINE_SIZE 256
#define TRI_NPTS 3
#define N_MESHES 8

#define CAM_X 0
#define CAM_Y 0
#define CAM_Z 0
#define ROWS 256
#define COLS 256
#define MAX_DEPTH 8
#define N_SAMPLES 200
#define DOF_SAMPLES 3
#define FOV_RAD (1.0472)
#define RAY_BUMP_AMT (0.001)
#define FOCAL_LENGTH (0.7)
#define APERTURE (0.04)
#define SUBPIX_RAND_OFFSET (.25)

#define CLAMP(val, min, max) (val < min ? min : (val > max ? max : val))

typedef struct
{
    vec3 origin;
    vec3 direction;
}Ray;

// assuming lambertian
typedef struct
{
    fcolor_t color;
    float emittance;
}Material;

typedef struct
{
    vec3 *verts;
    vec3 *normals;
    size_t size;
    Material material;
}Mesh;

typedef struct
{
    double radius;
    vec3 center;
    Material material;
}Sphere;

typedef struct
{
    size_t n_meshes;
    size_t n_spheres;
    Mesh *mesh_arr;
    Sphere *sphere_arr;
}Scene;

typedef struct
{
    size_t rows;
    size_t cols;
    fcolor_t *buffer;
}Framebuffer;

void print_progress(char* msg, int len, float current, float max);
void print_col(fcolor_t c);
int coord_to_idx(int x, int y, int rows, int cols);
bool almost_equal(double a, double b, double eps);
double frand(double low,double high);
double distance(vec3 a, vec3 b);
double square_dist(vec3 a, vec3 b);
void rand_grid_sample(vec2 out_offset, int n_samples, int sample_iter);
void triangle_normal(vec3 dst, vec3 a, vec3 b, vec3 c);
bool point_in_circle(vec2 pt, vec2 center, int radius);
void reinhard_cmap(fcolor_t ret, fcolor_t c);
void gen_camera_ray(Ray *ret, vec2 pixel, Framebuffer *fb);
void gen_camera_rays_dof(vec2 pixel, Framebuffer *fb,
        int n_samples, double focal_length, double aperture, Ray out_rays[n_samples]);
void cosine_sample_hemisphere(vec3 ret, vec3 normal);
bool ray_triangle_intersect(Ray *ray, vec3 v0, vec3 v1, vec3 v2, vec3 out_inter_pt);
bool ray_sphere_intersect(Ray *ray, Sphere *sphere, vec3 out_inter, vec3 out_norm);
bool cast_ray(Ray *ray, Scene *scene, vec3 out_hit, vec3 out_norm, Material **out_hit_mat);
void trace_path(fcolor_t out_color, Ray *ray, unsigned int depth, Scene *scene);
void render(Framebuffer *fb, Scene *scene);
void read_meshes(Mesh out_mesh_arr[N_MESHES], const char *mesh_list_path);
void free_meshes(Mesh mesh_arr[N_MESHES]);

int main(void)
{
    Mesh *cornell_meshes = malloc(sizeof(Mesh) * N_MESHES);
    read_meshes(cornell_meshes, "./objs.txt");

    const fcolor_t lgt_color = {1.0f, 0.83f, 0.66f};
    const fcolor_t white   = {0.7f, 0.7f, 0.7f};
    const fcolor_t green   = {0, 1.0f, 0};
    const fcolor_t red     = {1.0f, 0, 0};

    Material white_mat = {.emittance = 0};
    Material green_mat  = {.emittance = 0};
    Material red_mat  = {.emittance = 0};
    Material light_mat = {.emittance = 25};
    memcpy(red_mat.color, red, sizeof(fcolor_t));
    memcpy(green_mat.color, green, sizeof(fcolor_t));
    memcpy(white_mat.color, white, sizeof(fcolor_t));
    memcpy(light_mat.color, lgt_color, sizeof(fcolor_t));

    // ordering is determined in ./objs.txt
    cornell_meshes[0].material = red_mat;
    cornell_meshes[1].material = light_mat;
    cornell_meshes[2].material = green_mat;
    cornell_meshes[3].material = white_mat;
    cornell_meshes[4].material = white_mat;
    cornell_meshes[5].material = white_mat;
    cornell_meshes[6].material = white_mat;
    cornell_meshes[7].material = white_mat;

    const int n_spheres = 1;
    Sphere sphere_arr[n_spheres] = {
        {.center = {0,0,-0.7}, .radius = 0.08, .material=white_mat}
    };

    Scene scene = {.n_meshes = N_MESHES, .mesh_arr = cornell_meshes,
        .n_spheres = n_spheres, .sphere_arr=sphere_arr};

    Framebuffer fb;
    fb.rows = ROWS;
    fb.cols = COLS;
    const int fb_size = fb.rows * fb.cols;
    fb.buffer = calloc(fb_size, sizeof(fcolor_t));

    srandom(time(NULL));

    time_t begin = time(NULL);
    render(&fb, &scene);
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

    if (fork() == 0)
    {
        execl("/bin/cp", "cp", "./out.png", "./out_bak.png", (char*)0);
        exit(EXIT_SUCCESS);
    }
    else
        wait(NULL);

    const int png_stride = fb.cols * sizeof(write_buf[0]);
    stbi_write_jpg("./out.png",fb.cols,fb.rows,COL_NCHANNELS,write_buf, png_stride);

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

int coord_to_idx(int x, int y, int rows, int cols)
{
    // flipping vertically
    int row = (rows - 1) - y;
    // 2D idx (row, col) to a 1D index
    return row * cols + x;
}

bool almost_equal(double a, double b, double eps)
{
    return fabs(a-b) < eps;
}

double frand(double low,double high)
{
    return (random()/(double)(RAND_MAX))*fabs(low-high)+low;
}

double distance(vec3 a, vec3 b)
{
    return sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2) + pow(b[2] - a[2], 2));
}

double square_dist(vec3 a, vec3 b)
{
    return pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2) + pow(b[2] - a[2], 2);
}

void rand_grid_sample(vec2 out_offset, int n_samples, int sample_iter)
{
    const double step = 1. / n_samples;
    out_offset[0] = (step * ((sample_iter % n_samples) + 1));
    out_offset[1] = (step * (floor(sample_iter /(double) n_samples) + 1));
    out_offset[0] += frand(-SUBPIX_RAND_OFFSET, SUBPIX_RAND_OFFSET);
    out_offset[1] += frand(-SUBPIX_RAND_OFFSET, SUBPIX_RAND_OFFSET);
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

void reinhard_cmap(fcolor_t ret, fcolor_t c)
{
    fcolor_t t;
    fcolor_offset(t, c, 1);
    fcolor_divide(ret, c, t);
}

// switch rows/cols for a fb index as a pixel, and you're in raster space
void gen_camera_ray(Ray *ret, vec2 pixel, Framebuffer *fb)
{
    const double image_aspect = fb->rows / (double) fb->cols;

    const vec2 pixel_screen = {
        pixel[0] / fb->cols,
        1 - (pixel[1] / fb->rows)
    };

    const double t = tan(FOV_RAD / 2);
    ret->origin[0] = CAM_X;
    ret->origin[1] = CAM_Y;
    ret->origin[2] = CAM_Z;
    ret->direction[0] = (2 * pixel_screen[0] - 1) * t;
    ret->direction[1] = (1 - 2 * pixel_screen[1]) * image_aspect * t;
    ret->direction[2] = -1;
    vec3_norm(ret->direction, ret->direction);
}

void gen_camera_rays_dof(vec2 pixel, Framebuffer *fb,
        int n_samples, double focal_length, double aperture, Ray out_rays[n_samples])
{
    assert(n_samples > 0);

    gen_camera_ray(&(out_rays[0]), pixel, fb);
    vec3 *orig_origin = &(out_rays[0].origin);

    vec3 converge_pt;
    vec3 dir_scaled;
    vec3_scale(dir_scaled, out_rays[0].direction, focal_length);
    vec3_add(converge_pt, *orig_origin, dir_scaled);
    for (int i = 1; i < n_samples - 1; i++)
    {
        vec3 ray_offset = {frand(-aperture, aperture), frand(-aperture, aperture), 0};
        vec3_add(out_rays[i].origin, *orig_origin, ray_offset);
        vec3_sub(out_rays[i].direction, converge_pt, out_rays[i].origin);
        vec3_norm(out_rays[i].direction, out_rays[i].direction);
    }
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
bool ray_triangle_intersect(Ray *ray, vec3 v0, vec3 v1, vec3 v2, vec3 out_inter_pt)
{
    vec3 edge1, edge2, h, s, q;
    double a,f,u,v;
    vec3_sub(edge1, v1, v0);
    vec3_sub(edge2, v2, v0);
    vec3_mul_cross(h, ray->direction, edge2);
    a = vec3_mul_inner(edge1, h);
    if (a > -DBL_EPSILON && a < DBL_EPSILON)
        return false;    // This ray is parallel to this triangle.
    f = 1.0/a;
    vec3_sub(s, ray->origin, v0);
    u = f * vec3_mul_inner(s, h);
    if (u < 0.0 || u > 1.0)
        return false;
    vec3_mul_cross(q, s, edge1);
    v = f * vec3_mul_inner(ray->direction, q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * vec3_mul_inner(edge2, q);
    if (t > DBL_EPSILON) // ray intersection
    {
        vec3 ss;
        vec3_scale(ss, ray->direction, t);
        vec3_add(out_inter_pt, ray->origin, ss);
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}

bool ray_sphere_intersect(Ray *ray, Sphere *sphere, vec3 out_inter, vec3 out_norm)
{
    vec3 a;
    vec3_sub(a, ray->origin, sphere->center);
    float b = vec3_mul_inner(a, ray->direction);
    float c = vec3_mul_inner(a, a) - pow(sphere->radius, 2);

    if (c > 0 && b > 0)
        return false;

    float d = b * b - c;
    if (d < 0)
        return false;

    float t = (-b) - sqrt(d);
    if (t < 0)
        t = 0;

    vec3 td;
    vec3_scale(td, ray->direction, t);
    vec3_add(out_inter, ray->origin, td);
    vec3_sub(out_norm, out_inter, sphere->center);
    vec3_norm(out_norm, out_norm);
    return true;
}


bool cast_ray(Ray *ray, Scene *scene, vec3 out_hit, vec3 out_norm, Material **out_hit_mat)
{
    double closest_dist = DBL_MAX;
    bool did_intersect = false;

    // checking for intersections with meshes
    for (int i = 0; i < scene->n_meshes; i++)
    {
        vec3 intersection;
        for (int j = 0; j < scene->mesh_arr[i].size; j+=TRI_NPTS)
        {
            // i have no clue what i'm doing but i like the arrow
            Mesh *mesh = &scene->mesh_arr[i];
            bool ret = ray_triangle_intersect(
                    ray, mesh->verts[j], mesh->verts[j+1], mesh->verts[j+2], intersection);
            if (ret)
            {
                double distance = square_dist(ray->origin, intersection);
                if (distance < closest_dist)
                {
                    did_intersect = true;
                    closest_dist = distance;
                    memcpy(out_hit, intersection, sizeof(vec3));
                    triangle_normal(out_norm, mesh->verts[j+2], mesh->verts[j+1], mesh->verts[j+0]);
                    *out_hit_mat = &(mesh->material);
                }
            }
        }
    }

    // checking for intersections with spheres
    for (int i = 0; i < scene->n_spheres; i++)
    {
        vec3 intersection;
        vec3 normal;
        bool ret = ray_sphere_intersect(ray, &(scene->sphere_arr[i]), intersection, normal);
        if (ret)
        {
            double distance = square_dist(ray->origin, intersection);
            if (distance < closest_dist)
            {
                did_intersect = true;
                closest_dist = distance;
                memcpy(out_hit, intersection, sizeof(vec3));
                memcpy(out_norm, normal, sizeof(vec3));
                *out_hit_mat = &(scene->sphere_arr[i].material);
            }
        }
    }

    if (did_intersect)
    {
        vec3 normal_small_scale;
        vec3_scale(normal_small_scale, out_norm, RAY_BUMP_AMT);
        vec3_add(out_hit, out_hit, normal_small_scale);
    }

    return did_intersect;
}


void trace_path(fcolor_t out_color, Ray *ray, unsigned int depth, Scene *scene)
{
    memset(out_color, 0, sizeof(fcolor_t));

    if (depth >= MAX_DEPTH)
        return;

    vec3 ray_hit_pt;
    vec3 ray_hit_norm;
    Material *ray_hit_mat;
    bool ray_hit = cast_ray(ray, scene, ray_hit_pt, ray_hit_norm, &ray_hit_mat);
    if (!ray_hit)
        return;

    Ray new_ray;
    memcpy(new_ray.origin, ray_hit_pt, sizeof(vec3));
    cosine_sample_hemisphere(new_ray.direction, ray_hit_norm);

    fcolor_t incoming;
    trace_path(incoming, &new_ray, depth + 1, scene);

    // https://i.redd.it/802mndge03t01.png
    // https://computergraphics.stackexchange.com/questions/8578/how-to-set-equivalent-pdfs-for-cosine-weighted-and-uniform-sampled-hemispheres
    // to get light towards eye
    // a = light emitted from pt (material emit)
    // b = all the light coming into the point from the unit hemisphere samples (the recursive output)
    // c = BRDF - chances of such light rays bouncing towards the eye (color / pi)
    // d = irradiance factor over the normal at the point (ray_normal_dot)
    // ((a + b) * c) * d

    const double emitted_light = ray_hit_mat->emittance;
    const double ray_normal_dot = vec3_mul_inner(new_ray.direction, ray_hit_norm);
    const fcolor_t *surf_col = &(ray_hit_mat->color);
    const double pdf = almost_equal(ray_normal_dot, 0.f, FLT_EPSILON) ? 1 : ray_normal_dot / M_PI;

    for (int i = 0; i < COL_NCHANNELS; i++)
    {
        const double color = ((*surf_col)[i] / M_PI) * ray_normal_dot;
        const double emit = color * emitted_light;
        const double inc =  color * incoming[i];

        out_color[i] = (emit + inc) / pdf;
    }
}

void render(Framebuffer *fb, Scene *scene)
{
    const int fb_size = fb->rows * fb->cols;

    omp_lock_t writelock;
    omp_init_lock(&writelock);

    int loop_count = 0;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < fb->rows; i++)
    {
        for (int j = 0; j < fb->cols; j++)
        {
            fcolor_t color_avg = {0, 0, 0};
            for (int k = 0; k < N_SAMPLES; k++)
            {
                vec2 pixel = {j, i};
                fcolor_t out_color = {0};

                // pure stochastic sampling works better than my grid sampling technique ;-;
                vec2 pix_sample_offset = {frand(0,1),frand(0,1)};

                pixel[0] += pix_sample_offset[0];
                pixel[1] += pix_sample_offset[1];

                Ray dof_rays[DOF_SAMPLES];
                gen_camera_rays_dof(pixel, fb, DOF_SAMPLES, FOCAL_LENGTH, APERTURE, dof_rays);
                for (int l = 0; l < DOF_SAMPLES; l++)
                {
                    fcolor_t dof_color;
                    trace_path(dof_color, &(dof_rays[l]), 0, scene);
                    fcolor_add(out_color, out_color, dof_color);
                }
                fcolor_scale_inv(out_color, out_color, DOF_SAMPLES);

                fcolor_add(color_avg, color_avg, out_color);
            }

            fcolor_scale_inv(color_avg, color_avg, N_SAMPLES);

            for (int i = 0; i < COL_NCHANNELS; i++)
            {
                color_avg[i] = CLAMP(pow(color_avg[i], 1/2.2), 0.f, 1.f) * 255;
            }

            const int buf_idx = coord_to_idx(j, i, fb->rows, fb->cols);
            memcpy(fb->buffer[buf_idx], color_avg, sizeof(fcolor_t));

            omp_set_lock(&writelock);
            print_progress("Rendering... ", 50, loop_count++, fb_size);
            omp_unset_lock(&writelock);
        }
    }
    omp_destroy_lock(&writelock);
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
