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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TRI_NPTS 3
#define MAX_LINE_SIZE 256
#define N_MESHES 8
#define FOV_RAD (1.0472)
#define MAX_DEPTH 100

#define CAM_X 0
#define CAM_Y 0
#define CAM_Z 0

typedef struct Ray
{
    vec3 origin;
    vec3 direction;
}Ray;

typedef struct Mesh
{
    vec3 *verts;
    vec3 *normals;
    size_t size;
}Mesh;

typedef struct Framebuffer
{
    size_t rows;
    size_t cols;
    color_t *buffer;
}Framebuffer;

int coord_to_idx(int x, int y, int rows, int cols);
double randf(double low,double high);
double distance(vec3 a, vec3 b);
void stereographic_proj(vec3 dst, const vec2 pt);
void triangle_normal(vec3 dst, vec3 a, vec3 b, vec3 c);
bool point_in_circle(vec2 pt, vec2 center, int radius);
Ray gen_camera_ray(const unsigned int pixel[2], const Framebuffer *fb);
bool ray_triangle_intersect(Ray ray, vec3 v0, vec3 v1, vec3 v2, vec3 out_inter_pt);
bool cast_ray(Ray ray, Mesh meshes[N_MESHES], vec3 out_hit, vec3 out_norm);
void random_vec_in_hemisphere(vec3 ret, vec3 normal);
void trace_path(fcolor_t out_color, Ray ray, unsigned int depth, Mesh meshes[N_MESHES]);
void read_meshes(Mesh out_mesh_arr[N_MESHES], const char *mesh_list_path);
void free_meshes(Mesh mesh_arr[N_MESHES]);

int main(void)
{
    srand(time(NULL));
    Mesh *cornell_meshes = malloc(sizeof(Mesh) * N_MESHES);
    read_meshes(cornell_meshes, "./objs.txt");

    Framebuffer f;
    f.rows = 256;
    f.cols = 256;
    f.buffer = malloc(sizeof(color_t) * f.rows * f.cols);

    for (int i = 0; i < f.rows; i++)
    {
        for (int j = 0; j < f.cols; j++)
        {
            const unsigned int pixel[2] = {j, i};
            Ray r = gen_camera_ray(pixel, &f);

            fcolor_t out_color;
            trace_path(out_color, r, 0, cornell_meshes);

            color_t c = {out_color[0], out_color[1], out_color[2]};

            memcpy(f.buffer[coord_to_idx(j, i, f.rows, f.cols)], c, sizeof(color_t));
        }
    }

    stbi_write_jpg("./out.jpg",f.cols,f.rows,3,f.buffer,100);

    free_meshes(cornell_meshes);
}

int coord_to_idx(int x, int y, int rows, int cols)
{
    // flipping vertically
    int row = (rows - 1) - y;
    // 2D idx (row, col) to a 1D index
    return row * cols + x;
}

double randf(double low,double high)
{
    return (rand()/(double)(RAND_MAX))*fabs(low-high)+low;
}

double distance(vec3 a, vec3 b)
{
    return sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2) + pow(b[2] - a[2], 2));
}

// https://en.wikipedia.org/wiki/Stereographic_projection
void stereographic_proj(vec3 dst, const vec2 pt)
{
    double x = pt[0];
    double y = pt[1];
    double denom = 1 + pow(x, 2) + pow(y, 2);

    dst[0] = (2 * x) / denom;
    dst[1] = (2 * y) / denom;
    dst[2] = (denom - 2) / denom; /* i may have the biggest brain on the planet */
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
Ray gen_camera_ray(const unsigned int pixel[2], const Framebuffer *fb)
{
    const double image_aspect = fb->rows / (double) fb->cols;

    const vec2 pixel_screen = {
        (pixel[0] + randf(0, 1)) / fb->cols,
        1 - ((pixel[1] + randf(0, 1)) / fb->rows)
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

// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
// shamelessly copied, i'll understand it later
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

bool cast_ray(Ray ray, Mesh meshes[N_MESHES], vec3 out_hit, vec3 out_norm)
{
    list inter_pts;
    list_init(&inter_pts, sizeof(vec3), 10);

    list inter_normals;
    list_init(&inter_normals, sizeof(vec3), 10);

    vec3 intersection;
    vec3 normal;
    for (int i = 0; i < N_MESHES; i++)
    {
        Mesh mesh = meshes[i];
        for (int j = 0; j < mesh.size; j+=TRI_NPTS)
        {
            bool ret = ray_triangle_intersect(ray, mesh.verts[j], mesh.verts[j+1], mesh.verts[j+2],
                    intersection);
            if (ret)
            {
                list_append(&inter_pts, intersection);

                triangle_normal(normal, mesh.verts[j], mesh.verts[j+1], mesh.verts[j+2]);
                list_append(&inter_normals, normal);
            }
        }
    }

    if (inter_pts.used == 0)
    {
        list_free(&inter_pts);
        list_free(&inter_normals);
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

    vec3 *closest_norm = list_index(&inter_normals, closest_idx);
    vec3 *closest_pt = list_index(&inter_pts, closest_idx);
    memcpy(out_hit, *closest_pt, sizeof(vec3));
    memcpy(out_norm, *closest_norm, sizeof(vec3));

    list_free(&inter_pts);
    list_free(&inter_normals);
    return true;
}

void random_vec_in_hemisphere(vec3 ret, vec3 normal)
{
    // generating random disk point and projecting onto hemisphere
    vec2 rand_pt;
    do
    {
        rand_pt[0] = randf(-1, 1);
        rand_pt[1] = randf(-1, 1);
    }
    while (!point_in_circle(rand_pt, normal, 1));

    stereographic_proj(ret, rand_pt);
    vec3_norm(ret, ret);
}

void trace_path(fcolor_t out_color, Ray ray, unsigned int depth, Mesh meshes[N_MESHES])
{
    if (depth >= MAX_DEPTH)
    {
        memset(out_color, 0, sizeof(float) * 3);
        return;
    }


    vec3 ray_hit_pt;
    vec3 ray_hit_norm;
    bool ray_hit = cast_ray(ray, meshes, ray_hit_pt, ray_hit_norm);
    if (!ray_hit)
    {
        memset(out_color, 0, sizeof(float) * 3);
        return;
    }

    Ray new_ray;
    memcpy(new_ray.origin, ray_hit_pt, sizeof(vec3));
    random_vec_in_hemisphere(new_ray.direction, ray_hit_norm);

    fcolor_t color = {255, 255, 255};
    const float p = 1 / (2 * M_PI);
    const float cos_theta = vec3_mul_inner(new_ray.direction, ray_hit_norm);
    const float reflectance = 0.5 / M_PI;
    const float emittance = (float)(depth % 1000) / 1000;

    fcolor_t incoming;
    trace_path(incoming, new_ray, depth + 1, meshes);

    // putting it all together (rendering eqn)
    fcolor_mul(out_color, incoming, color);
    fcolor_scale(out_color, reflectance);
    fcolor_scale(out_color, cos_theta / p);
    fcolor_offset(out_color, emittance);
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
