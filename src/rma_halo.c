#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "grid.h"

// 2-D Cartesian decomposition. RMA halo exchanges:
// - Rows via contiguous MPI_Get
// - Cols via MPI_Get using MPI_Type_vector for strided columns
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int nx = 8, ny = 16;
    if (argc >= 3) { nx = atoi(argv[1]); ny = atoi(argv[2]); }
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

    // Window exposes entire buffer (disp_unit = sizeof(double))
    MPI_Win win;
    MPI_Aint nbytes = (MPI_Aint)((size_t)(g.nx + 2*g.nghost) * (size_t)g.pitch * sizeof(double));
    MPI_Win_create(g.data, nbytes, sizeof(double), MPI_INFO_NULL, cart, &win);

    // --- Row halos via contiguous MPI_Get ---
    double *top_halo    = grid_rowptr(&g, 0);
    double *bottom_halo = grid_rowptr(&g, g.nghost + g.nx);

    const int up_src_row   = nghost + nx - 1;
    const int down_src_row = nghost;
    const int pitch = ny + 2*nghost;

    MPI_Aint up_src_disp   = (up   < 0) ? 0 : (MPI_Aint)((MPI_Aint)up_src_row   * (MPI_Aint)pitch + (MPI_Aint)nghost);
    MPI_Aint down_src_disp = (down < 0) ? 0 : (MPI_Aint)((MPI_Aint)down_src_row * (MPI_Aint)pitch + (MPI_Aint)nghost);

    if (up   >= 0) MPI_Win_lock(MPI_LOCK_SHARED, up,   0, win);
    if (down >= 0) MPI_Win_lock(MPI_LOCK_SHARED, down, 0, win);

    if (up   >= 0) MPI_Get(top_halo + g.nghost,    g.ny, MPI_DOUBLE, up,   up_src_disp,   g.ny, MPI_DOUBLE, win);
    if (down >= 0) MPI_Get(bottom_halo + g.nghost, g.ny, MPI_DOUBLE, down, down_src_disp, g.ny, MPI_DOUBLE, win);

    if (up   >= 0) MPI_Win_flush(up, win);
    if (down >= 0) MPI_Win_flush(down, win);

    if (up   >= 0) MPI_Win_unlock(up, win);
    if (down >= 0) MPI_Win_unlock(down, win);

    // --- Column halos via vector datatypes and MPI_Get ---
    MPI_Datatype col_type;
    MPI_Type_vector(g.nx, 1, g.pitch, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    // Local column starts (first element of each column region)
    double *left_halo_start     = &g.data[grid_index(&g, g.nghost, 0)];
    double *right_halo_start    = &g.data[grid_index(&g, g.nghost, g.nghost + g.ny)];

    // Target displacements (in double units) into neighbor memories:
    // From LEFT neighbor, fetch its right interior column (col = nghost + ny - 1)
    // From RIGHT neighbor, fetch its left interior column (col = nghost)
    MPI_Aint left_src_disp  = (left  < 0) ? 0 : (MPI_Aint)((MPI_Aint)g.nghost * (MPI_Aint)g.pitch + (MPI_Aint)(g.nghost + g.ny - 1));
    MPI_Aint right_src_disp = (right < 0) ? 0 : (MPI_Aint)((MPI_Aint)g.nghost * (MPI_Aint)g.pitch + (MPI_Aint)g.nghost);

    if (left  >= 0) MPI_Win_lock(MPI_LOCK_SHARED, left,  0, win);
    if (right >= 0) MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);

    if (left  >= 0) MPI_Get(left_halo_start,  1, col_type, left,  left_src_disp,  1, col_type, win);
    if (right >= 0) MPI_Get(right_halo_start, 1, col_type, right, right_src_disp, 1, col_type, win);

    if (left  >= 0) MPI_Win_flush(left, win);
    if (right >= 0) MPI_Win_flush(right, win);

    if (left  >= 0) MPI_Win_unlock(left, win);
    if (right >= 0) MPI_Win_unlock(right, win);

    // Verify both row and column halos
    int ok_rows = grid_verify_row_halos(&g, up, down);
    int ok_cols = grid_verify_col_halos(&g, left, right);
    int ok = ok_rows && ok_cols;
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, cart);

    if (rank == 0) {
        if (all_ok)
            printf("[rma_halo] PASS (rows+cols via MPI_Get) on %d ranks; dims=(%d,%d), nx=%d, ny=%d\n",
                   world_size, dims[0], dims[1], nx, ny);
        else
            printf("[rma_halo] FAIL\n");
    }

    MPI_Type_free(&col_type);
    MPI_Win_free(&win);
    grid_free(&g);
    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}
