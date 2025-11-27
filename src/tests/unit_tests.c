#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "../grid.h"

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int nx = 4;
	int ny = 5;
	int nghost = 1;
	Grid g;

	if (grid_alloc(&g, nx, ny, nghost) != 0) {
    	if (rank == 0) printf("Allocation test FAILED\n");
    	MPI_Abort(MPI_COMM_WORLD, 1);
	}

	grid_fill(&g, rank);

	int passed_fill = 1;
	for (int i = nghost; i < nghost + nx; i++) {
    	for (int j = nghost; j < nghost + ny; j++) {
        	if (g.data[grid_index(&g,i,j)] != (double)rank) {
            	passed_fill = 0;
        	}
    	}
	}

	if (rank == 0) {
    	if (passed_fill)
        	printf("grid_fill test PASSED\n");
    	else
        	printf("grid_fill test FAILED\n");
	}

	grid_free(&g);
	MPI_Finalize();
	return 0;
}