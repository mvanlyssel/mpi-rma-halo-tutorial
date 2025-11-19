#ifndef GRID_H
#define GRID_H

#include <stddef.h>

typedef struct {
    int nx;       // local rows (interior)
    int ny;       // local cols (interior)
    int nghost;   // halo width (cells)
    int pitch;    // total columns including halos (ny + 2*nghost)
    double *data; // row-major: (nx + 2*nghost) x (ny + 2*nghost)
} Grid;

int grid_alloc(Grid *g, int nx, int ny, int nghost);
void grid_free(Grid *g);
void grid_fill(Grid *g, int rank);

static inline size_t grid_index(const Grid *g, int i, int j) {
    return (size_t)i * (size_t)g->pitch + (size_t)j;
}
static inline double* grid_rowptr(const Grid *g, int i) {
    return (double*)(&g->data[grid_index(g, i, 0)]);
}

double grid_checksum(const Grid *g);
int grid_verify_row_halos(const Grid *g, int up_rank, int down_rank);
int grid_verify_col_halos(const Grid *g, int left_rank, int right_rank);

#endif