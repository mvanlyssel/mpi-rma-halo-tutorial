#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "grid.h"

// Passive-target RMA halo exchange for row halos using MPI_Get.
// Each rank fetches its top/bottom halo rows from its neighbors' interior rows.
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = 8, ny = 16;
    if (argc >= 3) { nx = atoi(argv[1]); ny = atoi(argv[2]); }
    const int nghost = 1;

    // Neighbors for a 1-D row decomposition
    int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    Grid g = (Grid){0};
    if (grid_alloc(&g, nx, ny, nghost) != 0) {
        if (rank == 0) fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    grid_fill(&g, rank);

    // Expose the entire buffer (disp_unit = sizeof(double); target disps are in "doubles")
    MPI_Win win;
    MPI_Aint nbytes = (MPI_Aint)((size_t)(g.nx + 2*g.nghost) * (size_t)g.pitch * sizeof(double));
    MPI_Win_create(g.data, nbytes, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Local halo row pointers
    double *top_halo    = grid_rowptr(&g, 0);
    double *bottom_halo = grid_rowptr(&g, g.nghost + g.nx);

    // We fetch: top halo  <= up neighbor's last interior row
    //           bottom halo <= down neighbor's first interior row
    const int up_src_row   = nghost + nx - 1;  // in the UP neighbor's grid
    const int down_src_row = nghost;           // in the DOWN neighbor's grid
    const int pitch = ny + 2*nghost;

    // Target displacements (units = double, because disp_unit = sizeof(double))
    MPI_Aint up_src_disp   = (up   < 0) ? 0 : (MPI_Aint)((MPI_Aint)up_src_row   * (MPI_Aint)pitch + (MPI_Aint)nghost);
    MPI_Aint down_src_disp = (down < 0) ? 0 : (MPI_Aint)((MPI_Aint)down_src_row * (MPI_Aint)pitch + (MPI_Aint)nghost);

    // Begin passive-target epochs and fetch halo rows
    if (up   >= 0) MPI_Win_lock(MPI_LOCK_SHARED, up,   0, win);
    if (down >= 0) MPI_Win_lock(MPI_LOCK_SHARED, down, 0, win);

    if (up   >= 0) MPI_Get(top_halo    + g.nghost, g.ny, MPI_DOUBLE, up,   up_src_disp,   g.ny, MPI_DOUBLE, win);
    if (down >= 0) MPI_Get(bottom_halo + g.nghost, g.ny, MPI_DOUBLE, down, down_src_disp, g.ny, MPI_DOUBLE, win);

    if (up   >= 0) MPI_Win_flush(up, win);    // complete remote ops to UP
    if (down >= 0) MPI_Win_flush(down, win);  // complete remote ops to DOWN

    if (up   >= 0) MPI_Win_unlock(up, win);
    if (down >= 0) MPI_Win_unlock(down, win);

    // Verify: halos should now equal neighbor ranks (or remain sentinel at domain boundaries)
    int ok = grid_verify_row_halos(&g, up, down);
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        if (all_ok)
            printf("[rma_halo] PASS (MPI_Get row halos) across %d ranks (nx=%d, ny=%d)\n", size, nx, ny);
        else
            printf("[rma_halo] FAIL\n");
    }

    MPI_Win_free(&win);
    grid_free(&g);
    MPI_Finalize();
    return 0;
}
