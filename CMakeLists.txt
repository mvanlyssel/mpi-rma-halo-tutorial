cmake_minimum_required(VERSION 3.12)
project(mpi_rma_halo_tutorial C)

target_link_libraries(rma_halo PRIVATE grid MPI::MPI_C)
