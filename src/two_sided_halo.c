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

// One Jacobi step over the interior (reads 'in->data', writes 'out' at same indexes)
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

    // Local problem size (interior per rank). Override with args.
    int nx = 8, ny = 16, iters = 50;
    parse_args(argc, argv, &nx, &ny, &iters);
    const int nghost = 1;

    // Create a 2-D Cartesian communicator (periodic = false)
    int dims[2] = {0, 0};
    MPI_Dims_create(world_size, 2, dims); // choose factors close to square
    int periods[2] = {0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart);

    int rank, up, down, left, right;
    MPI_Comm_rank(cart, &rank);
    MPI_Cart_shift(cart, 0, 1, &up, &down);    // dim 0 = rows
    MPI_Cart_shift(cart, 1, 1, &left, &right); // dim 1 = cols

    Grid g = (Grid){0};
    if (grid_alloc(&g, nx, ny, nghost) != 0) {
        if (rank == 0) fprintf(stderr, "Allocation failed\n");
        MPI_Abort(cart, 1);
    }
    grid_fill(&g, rank);

    // Scratch buffer (same shape) used for Jacobi
    double *scratch = (double*)malloc((size_t)(g.nx + 2*g.nghost) * (size_t)g.pitch * sizeof(double));
    if (!scratch) { if (rank==0) fprintf(stderr,"scratch alloc failed\n"); MPI_Abort(cart, 1); }

    // Row halos (top/bottom)
    double *top_halo        = grid_rowptr(&g, 0);
    double *top_interior    = grid_rowptr(&g, nghost);
    double *bottom_interior = grid_rowptr(&g, nghost + g.nx - 1);
    double *bottom_halo     = grid_rowptr(&g, nghost + g.nx);

    // Column halos (left/right): use vector datatype with stride = pitch
    MPI_Datatype col_type;
    MPI_Type_vector(g.nx, 1, g.pitch, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    double *left_halo_start      = &g.data[grid_index(&g, nghost, 0)];
    double *left_interior_start  = &g.data[grid_index(&g, nghost, g.nghost)];
    double *right_interior_start = &g.data[grid_index(&g, nghost, g.nghost + g.ny - 1)];
    double *right_halo_start     = &g.data[grid_index(&g, nghost, g.nghost + g.ny)];

    const int TAG_UPWARD = 0, TAG_DOWNWARD = 1, TAG_LEFTWARD = 2, TAG_RIGHTWARD = 3;

    // --- One halo exchange for correctness check ----------------------------
    {
        MPI_Request reqs[8]; int rcount = 0;

        // Row exchanges (recv then send)
        MPI_Irecv(top_halo + g.nghost,    g.ny, MPI_DOUBLE, up,   TAG_DOWNWARD, cart, &reqs[rcount++]);
        MPI_Irecv(bottom_halo + g.nghost, g.ny, MPI_DOUBLE, down, TAG_UPWARD,   cart, &reqs[rcount++]);

        MPI_Isend(top_interior + g.nghost,    g.ny, MPI_DOUBLE, up,   TAG_UPWARD,   cart, &reqs[rcount++]);
        MPI_Isend(bottom_interior + g.nghost, g.ny, MPI_DOUBLE, down, TAG_DOWNWARD, cart, &reqs[rcount++]);

        // Column exchanges (recv then send)
        MPI_Irecv(left_halo_start,  1, col_type, left,  TAG_RIGHTWARD, cart, &reqs[rcount++]);
        MPI_Irecv(right_halo_start, 1, col_type, right, TAG_LEFTWARD,  cart, &reqs[rcount++]);

        MPI_Isend(left_interior_start,  1, col_type, left,  TAG_LEFTWARD,  cart, &reqs[rcount++]);
        MPI_Isend(right_interior_start, 1, col_type, right, TAG_RIGHTWARD, cart, &reqs[rcount++]);

        MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);

        int ok_rows = grid_verify_row_halos(&g, up, down);
        int ok_cols = grid_verify_col_halos(&g, left, right);
        int ok = ok_rows && ok_cols;
        int all_ok = 0;
        MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, cart);

        if (rank == 0) {
            if (all_ok)
                printf("[two_sided_halo] halo correctness: PASS (rows+cols)\n");
            else
                printf("[two_sided_halo] halo correctness: FAIL\n");
        }
    }

    // --- Timing + iteration loop (performance only) -------------------------
    MPI_Barrier(cart);
    double t0 = MPI_Wtime();

    for (int it = 0; it < iters; ++it) {
        MPI_Request reqs[8]; int rcount = 0;

        // Row exchanges (recv then send)
        MPI_Irecv(top_halo + g.nghost,    g.ny, MPI_DOUBLE, up,   TAG_DOWNWARD, cart, &reqs[rcount++]);
        MPI_Irecv(bottom_halo + g.nghost, g.ny, MPI_DOUBLE, down, TAG_UPWARD,   cart, &reqs[rcount++]);

        MPI_Isend(top_interior + g.nghost,    g.ny, MPI_DOUBLE, up,   TAG_UPWARD,   cart, &reqs[rcount++]);
        MPI_Isend(bottom_interior + g.nghost, g.ny, MPI_DOUBLE, down, TAG_DOWNWARD, cart, &reqs[rcount++]);

        // Column exchanges (recv then send), using vector datatype
        MPI_Irecv(left_halo_start,  1, col_type, left,  TAG_RIGHTWARD, cart, &reqs[rcount++]);
        MPI_Irecv(right_halo_start, 1, col_type, right, TAG_LEFTWARD,  cart, &reqs[rcount++]);

        MPI_Isend(left_interior_start,  1, col_type, left,  TAG_LEFTWARD,  cart, &reqs[rcount++]);
        MPI_Isend(right_interior_start, 1, col_type, right, TAG_RIGHTWARD, cart, &reqs[rcount++]);

        MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);

        // compute one Jacobi step over the interior (reads g.data, writes scratch)
        jacobi_step(&g, scratch);

        // swap buffers: copy new interior into g.data; keep halos as-is
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
        printf("[two_sided_halo] dims=(%d,%d) iters=%d time_min=%.6f time_avg=%.6f time_max=%.6f\n",
               dims[0], dims[1], iters, minT, sumT/(double)(dims[0]*dims[1]), maxT);
    }

    // Cleanup
    MPI_Type_free(&col_type);
    free(scratch);
    grid_free(&g);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
