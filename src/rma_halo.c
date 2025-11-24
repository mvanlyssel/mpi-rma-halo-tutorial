#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "grid.h"

// --- CLI & compute helpers ---------------------------------------------------
static void parse_args(int argc, char**argv, int *nx, int *ny, int *iters) {
    *iters = 50;          // default iterations
    if (argc >= 3) { *nx = atoi(argv[1]); *ny = atoi(argv[2]); }
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i+1 < argc) { *iters = atoi(argv[++i]); }
    }
}

static void jacobi_step(const Grid *in, double *out) {
    const int g = in->nghost;
    for (int i = g; i < g + in->nx; ++i) {
        for (int j = g; j < g + in->ny; ++j) {
            out[grid_index(in,i,j)] = 0.25 * (
               in->data[grid_index(in,i-1,j)] + in->data[grid_index(in,i+1,j)]
             + in->data[grid_index(in,i,j-1)] + in->data[grid_index(in,i,j+1)]
            );
        }
    }
}

// --- Main --------------------------------------------------------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int nx = 8, ny = 16, iters = 50;
    parse_args(argc, argv, &nx, &ny, &iters);
    const int nghost = 1;

    // 2-D Cartesian topology
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);

    int rank, up, down, left, right;
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_shift(cart, 0, 1, &up, &down);
    MPI_Cart_shift(cart, 1, 1, &left, &right);

    Grid g = (Grid){0};
    if (grid_alloc(&g, nx, ny, nghost) != 0) {
        if (rank == 0) fprintf(stderr, "Allocation failed\n");
        MPI_Abort(cart, 1);
    }
    grid_fill(&g, rank);

    // Scratch buffer for Jacobi
    double *scratch = (double*)malloc((size_t)(g.nx + 2*g.nghost) * (size_t)g.pitch * sizeof(double));
    if (!scratch) { if (rank==0) fprintf(stderr,"scratch alloc failed\n"); MPI_Abort(cart, 1); }

    // Window exposes entire buffer (disp_unit = sizeof(double))
    MPI_Win win;
    MPI_Aint nbytes = (MPI_Aint)((size_t)(g.nx + 2*g.nghost) * (size_t)g.pitch * sizeof(double));
    MPI_Win_create(g.data, nbytes, sizeof(double), MPI_INFO_NULL, cart, &win);

    // Row halos via contiguous MPI_Get
    double *top_halo    = grid_rowptr(&g, 0);
    double *bottom_halo = grid_rowptr(&g, g.nghost + g.nx);

    const int up_src_row   = nghost + nx - 1;
    const int down_src_row = nghost;
    const int pitch = ny + 2*nghost;

    MPI_Aint up_src_disp   = (up   < 0) ? 0 : (MPI_Aint)((MPI_Aint)up_src_row   * (MPI_Aint)pitch + (MPI_Aint)nghost);
    MPI_Aint down_src_disp = (down < 0) ? 0 : (MPI_Aint)((MPI_Aint)down_src_row * (MPI_Aint)pitch + (MPI_Aint)nghost);

    // Column halos via vector datatypes and MPI_Get
    MPI_Datatype col_type;
    MPI_Type_vector(g.nx, 1, g.pitch, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    double *left_halo_start   = &g.data[grid_index(&g, g.nghost, 0)];
    double *right_halo_start  = &g.data[grid_index(&g, g.nghost, g.nghost + g.ny)];

    MPI_Aint left_src_disp  = (left  < 0) ? 0 : (MPI_Aint)((MPI_Aint)g.nghost * (MPI_Aint)g.pitch + (MPI_Aint)(g.nghost + g.ny - 1));
    MPI_Aint right_src_disp = (right < 0) ? 0 : (MPI_Aint)((MPI_Aint)g.nghost * (MPI_Aint)g.pitch + (MPI_Aint)g.nghost);

    // --- One RMA halo exchange for correctness check ------------------------
    {
        // Rows
        if (up   >= 0) MPI_Win_lock(MPI_LOCK_SHARED, up,   0, win);
        if (down >= 0) MPI_Win_lock(MPI_LOCK_SHARED, down, 0, win);

        if (up   >= 0) MPI_Get(top_halo + g.nghost,    g.ny, MPI_DOUBLE, up,   up_src_disp,   g.ny, MPI_DOUBLE, win);
        if (down >= 0) MPI_Get(bottom_halo + g.nghost, g.ny, MPI_DOUBLE, down, down_src_disp, g.ny, MPI_DOUBLE, win);

        if (up   >= 0) MPI_Win_flush(up, win);
        if (down >= 0) MPI_Win_flush(down, win);

        if (up   >= 0) MPI_Win_unlock(up, win);
        if (down >= 0) MPI_Win_unlock(down, win);

        // Columns
        if (left  >= 0) MPI_Win_lock(MPI_LOCK_SHARED, left,  0, win);
        if (right >= 0) MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);

        if (left  >= 0) MPI_Get(left_halo_start,  1, col_type, left,  left_src_disp,  1, col_type, win);
        if (right >= 0) MPI_Get(right_halo_start, 1, col_type, right, right_src_disp, 1, col_type, win);

        if (left  >= 0) MPI_Win_flush(left, win);
        if (right >= 0) MPI_Win_flush(right, win);

        if (left  >= 0) MPI_Win_unlock(left, win);
        if (right >= 0) MPI_Win_unlock(right, win);

        int ok_rows = grid_verify_row_halos(&g, up, down);
        int ok_cols = grid_verify_col_halos(&g, left, right);
        int ok = ok_rows && ok_cols;
        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, cart);

        if (rank == 0) {
            if (all_ok)
                printf("[rma_halo] halo correctness: PASS (rows+cols via MPI_Get)\n");
            else
                printf("[rma_halo] halo correctness: FAIL\n");
        }
    }

    // --- Timing + iteration loop (performance only) -------------------------
    MPI_Barrier(cart);
    double t0 = MPI_Wtime();

    for (int it = 0; it < iters; ++it) {
        // Rows
        if (up   >= 0) MPI_Win_lock(MPI_LOCK_SHARED, up,   0, win);
        if (down >= 0) MPI_Win_lock(MPI_LOCK_SHARED, down, 0, win);

        if (up   >= 0) MPI_Get(top_halo + g.nghost,    g.ny, MPI_DOUBLE, up,   up_src_disp,   g.ny, MPI_DOUBLE, win);
        if (down >= 0) MPI_Get(bottom_halo + g.nghost, g.ny, MPI_DOUBLE, down, down_src_disp, g.ny, MPI_DOUBLE, win);

        if (up   >= 0) MPI_Win_flush(up, win);
        if (down >= 0) MPI_Win_flush(down, win);

        if (up   >= 0) MPI_Win_unlock(up, win);
        if (down >= 0) MPI_Win_unlock(down, win);

        // Columns
        if (left  >= 0) MPI_Win_lock(MPI_LOCK_SHARED, left,  0, win);
        if (right >= 0) MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);

        if (left  >= 0) MPI_Get(left_halo_start,  1, col_type, left,  left_src_disp,  1, col_type, win);
        if (right >= 0) MPI_Get(right_halo_start, 1, col_type, right, right_src_disp, 1, col_type, win);

        if (left  >= 0) MPI_Win_flush(left, win);
        if (right >= 0) MPI_Win_flush(right, win);

        if (left  >= 0) MPI_Win_unlock(left, win);
        if (right >= 0) MPI_Win_unlock(right, win);

        // compute one step (reads g.data, writes scratch), then swap interior
        jacobi_step(&g, scratch);
        for (int i = g.nghost; i < g.nghost + g.nx; ++i)
            for (int j = g.nghost; j < g.nghost + g.ny; ++j)
                g.data[grid_index(&g,i,j)] = scratch[grid_index(&g,i,j)];
    }

    double t1 = MPI_Wtime();
    double dt = t1 - t0;
    double minT, maxT, sumT;
    MPI_Reduce(&dt, &minT, 1, MPI_DOUBLE, MPI_MIN, 0, cart);
    MPI_Reduce(&dt, &maxT, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    MPI_Reduce(&dt, &sumT, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
    if (rank == 0) {
        printf("[rma_halo] dims=(%d,%d) iters=%d time_min=%.6f time_avg=%.6f time_max=%.6f\n",
               dims[0], dims[1], iters, minT, sumT/(double)(dims[0]*dims[1]), maxT);
    }

    // Cleanup
    MPI_Type_free(&col_type);
    MPI_Win_free(&win);
    free(scratch);
    grid_free(&g);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
