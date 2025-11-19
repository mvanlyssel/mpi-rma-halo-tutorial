#include "grid.h"
#include <stdlib.h>
#include <math.h>

static const double HALO_SENTINEL = -1.0;

int grid_alloc(Grid *g, int nx, int ny, int nghost) {
    if (nx <= 0 || ny <= 0 || nghost < 0) return -1;
    g->nx = nx;
    g->ny = ny;
    g->nghost = nghost;
    g->pitch = ny + 2 * nghost;
    size_t rows = (size_t)nx + 2u * (size_t)nghost;
    size_t cols = (size_t)g->pitch;
    size_t n = rows * cols;
    g->data = (double*)malloc(n * sizeof(double));
    if (!g->data) return -1;
    return 0;
}

void grid_free(Grid *g) {
    if (g && g->data) {
        free(g->data);
        g->data = NULL;
    }
}

void grid_fill(Grid *g, int rank) {
    size_t rows = (size_t)g->nx + 2u * (size_t)g->nghost;
    size_t cols = (size_t)g->pitch;

    // Initialize everything to halo sentinel
    for (size_t i = 0; i < rows * cols; ++i) g->data[i] = HALO_SENTINEL;

    // Fill interior with the rank value (makes halo verification trivial)
    for (int i = g->nghost; i < g->nghost + g->nx; ++i)
        for (int j = g->nghost; j < g->nghost + g->ny; ++j)
            g->data[grid_index(g, i, j)] = (double)rank;
}

double grid_checksum(const Grid *g) {
    size_t rows = (size_t)g->nx + 2u * (size_t)g->nghost;
    size_t cols = (size_t)g->pitch;
    long double s = 0.0L;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            s += g->data[grid_index(g, (int)i, (int)j)];
    return (double)s;
}

int grid_verify_row_halos(const Grid *g, int up_rank, int down_rank) {
    int ok = 1;
    int top = 0;
    int bottom = g->nghost + g->nx; // first ghost row after interior

    // Top halo should equal up_rank (or remain sentinel if no neighbor)
    for (int j = g->nghost; j < g->nghost + g->ny; ++j) {
        double v = g->data[grid_index(g, top, j)];
        double want = (up_rank < 0) ? -1.0 : (double)up_rank;
        if (fabsl(v - want) > 1e-12L) { ok = 0; break; }
    }

    // Bottom halo should equal down_rank (or sentinel)
    if (ok) {
        for (int j = g->nghost; j < g->nghost + g->ny; ++j) {
            double v = g->data[grid_index(g, bottom, j)];
            double want = (down_rank < 0) ? -1.0 : (double)down_rank;
            if (fabsl(v - want) > 1e-12L) { ok = 0; break; }
        }
    }
    return ok;
}