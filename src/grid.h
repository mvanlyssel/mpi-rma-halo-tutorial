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

// Allocate a grid with ghost layers; returns 0 on success
int grid_alloc(Grid *g, int nx, int ny, int nghost);

// Free allocated memory in Grid
void grid_free(Grid *g);

// Fill interior with a simple pattern based on rank; halos set to sentinel (-1.0)
void grid_fill(Grid *g, int rank);

// Index helpers
static inline size_t grid_index(const Grid *g, int i, int j) {
    return (size_t)i * (size_t)g->pitch + (size_t)j;
}
static inline double* grid_rowptr(const Grid *g, int i) {
    return (double*)(&g->data[grid_index(g, i, 0)]);
}

// Simple checksum of the whole array (debug/sanity)
double grid_checksum(const Grid *g);

// Verify row halos equal neighbor ranks after an exchange.
// If neighbor is MPI_PROC_NULL, halos are expected to remain sentinel (-1.0).
int grid_verify_row_halos(const Grid *g, int up_rank, int down_rank);

#endif // GRID_H
