#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "grid.h"

// 1-D row decomposition across ranks. Exchanges top/bottom halo rows using Isend/Irecv.
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Problem size per rank (interior). Override via: mpirun -np 4 ./two_sided_halo 8 16
    int nx = 8;   // rows per rank
    int ny = 16;  // cols per rank
    if (argc >= 3) { nx = atoi(argv[1]); ny = atoi(argv[2]); }
    const int nghost = 1;

    // Neighbors in a simple 1-D split
    int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    Grid g = {0};
    if (grid_alloc(&g, nx, ny, nghost) != 0) {
        if (rank == 0) fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    grid_fill(&g, rank);

    // Pointers to halo and interior rows
    double *top_halo         = grid_rowptr(&g, 0);
    double *top_interior     = grid_rowptr(&g, nghost);
    double *bottom_interior  = grid_rowptr(&g, nghost + g.nx - 1);
    double *bottom_halo      = grid_rowptr(&g, nghost + g.nx);

    // Tag convention: 0 = upward-bound messages; 1 = downward-bound messages
    const int TAG_UPWARD = 0;
    const int TAG_DOWNWARD = 1;

    MPI_Request reqs[4];
    int rcount = 0;

    // Post receives first (safe pattern), then sends.
    // Receive top halo from 'up' (their msg travels downward -> TAG_DOWNWARD)
    MPI_Irecv(top_halo + g.nghost,    g.ny, MPI_DOUBLE, up,   TAG_DOWNWARD, MPI_COMM_WORLD, &reqs[rcount++]);
    // Receive bottom halo from 'down' (their msg travels upward -> TAG_UPWARD)
    MPI_Irecv(bottom_halo + g.nghost, g.ny, MPI_DOUBLE, down, TAG_UPWARD,   MPI_COMM_WORLD, &reqs[rcount++]);

    // Send our first interior row upward (travels upward -> TAG_UPWARD)
    MPI_Isend(top_interior + g.nghost,    g.ny, MPI_DOUBLE, up,   TAG_UPWARD,   MPI_COMM_WORLD, &reqs[rcount++]);
    // Send our last interior row downward (travels downward -> TAG_DOWNWARD)
    MPI_Isend(bottom_interior + g.nghost, g.ny, MPI_DOUBLE, down, TAG_DOWNWARD, MPI_COMM_WORLD, &reqs[rcount++]);

    MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);

    // Verify halos reflect neighbor ranks (or remain sentinel at boundaries)
    int ok = grid_verify_row_halos(&g, up, down);
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        if (all_ok)
            printf("[two_sided_halo] PASS across %d ranks (nx=%d, ny=%d)\n", size, nx, ny);
        else
            printf("[two_sided_halo] FAIL\n");
    }

    // Optional: print checksum per rank for debugging
    double cs = grid_checksum(&g);
    printf("Rank %d checksum: %.1f (up=%d, down=%d)\n", rank, cs, up, down);

    grid_free(&g);
    MPI_Finalize();
    return 0;
}