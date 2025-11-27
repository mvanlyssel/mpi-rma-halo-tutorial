cmake_minimum_required(VERSION 3.12)
project(mpi_rma_halo_tutorial C)

find_package(MPI REQUIRED C)

set(CMAKE_C_STANDARD 99)

# Grid helper library
add_library(grid STATIC src/grid.c)
target_include_directories(grid
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${MPI_C_INCLUDE_DIRS}
)

# Two-sided halo executable
add_executable(two_sided_halo src/two_sided_halo.c)
target_link_libraries(two_sided_halo
    PRIVATE
        grid
        MPI::MPI_C
)

# RMA halo executable
add_executable(rma_halo src/rma_halo.c)
target_link_libraries(rma_halo
    PRIVATE
        grid
        MPI::MPI_C
)

# Unit tests executable
add_executable(grid_unit_tests src/tests/unit_tests.c)
target_link_libraries(grid_unit_tests
    PRIVATE
        grid
        MPI::MPI_C
)