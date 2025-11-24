This project demonstrates and compares two different methods of performing halo exchanges in a two-dimensional domain 
decomposition using MPI. The goal is to show the differences between traditional two-sided communication and MPI’s 
one-sided RMA communication, and to provide a clean, working example of both approaches. This tutorial is designed as 
an advanced extension of the basic MPI halo exchange material used in our class.

Halo exchanges are used in parallel scientific simulations where each MPI rank owns a portion of a global grid and must 
periodically exchange boundary data with its neighbors. Each rank stores its interior region as well as one layer of ghost 
cells, also called halos. These halos are filled with data pulled from neighboring ranks. This project implements exchanges 
of both the top and bottom rows and the left and right columns. Row halos are contiguous in memory, while column halos 
require a strided transfer.

The code is organized into several files. grid.c and grid.h contain the data structure and helper routines for allocating 
a grid, accessing elements, initializing interior values with a rank identifier, and verifying halo correctness. 
two_sided_halo.c contains the implementation that uses nonblocking MPI_Isend and MPI_Irecv calls to exchange halo rows and 
columns. rma_halo.c contains the implementation that uses MPI’s remote-memory access model, issuing MPI_Get operations inside 
passive target epochs defined by MPI_Win_lock, MPI_Win_flush, and MPI_Win_unlock. A CMakeLists.txt file builds the grid 
library and both executables.

To build the project, create a build directory, configure with CMake using mpicc as the compiler, and run make. This produces 
two executables named two_sided_halo and rma_halo. Both programs take the interior dimensions nx and ny and an optional --iters 
flag specifying the number of Jacobi iterations to run. Typical usage is: 

mpirun -np 4 ./two_sided_halo 64 64 --iters 50 
mpirun -np 4 ./rma_halo 64 64 --iters 50

Each program begins by creating a two-dimensional Cartesian communicator and determining the ranks of the up, down, left, and right 
neighbors. The grid is allocated with one ghost layer and filled with the rank ID so that halo regions can be checked for correctness. 
Before performing any timed iterations, each program performs one initial halo exchange and verifies that the halos now contain the 
neighbor rank values. This initial verification is the only time correctness is checked. After that, the program runs the specified 
number of iterations of halo exchange followed by a Jacobi update applied only to the interior region. Halos are updated solely 
by communication.

In the two-sided version, each iteration posts receives for incoming row halos, receives for incoming column halos (using a vector datatype), 
and sends the matching outgoing rows and columns. All eight operations are completed with MPI_Waitall. In the RMA version, each rank locks 
the neighbor’s window, issues MPI_Get operations to pull the neighbor’s interior boundary into its own halos, flushes the operations, and 
unlocks the window. The row gets use contiguous transfers and the column gets use the same vector datatype as the two-sided version.

After the iteration loop, each program reports the minimum, average, and maximum runtime across the ranks. These measurements allow comparing 
how the two communication styles perform under the same conditions. In practice, the two-sided communication is often slightly faster for small 
halos because there is less synchronization overhead, while RMA may perform better when dealing with large messages or irregular access patterns. 
The example provided here is simple enough that both implementations behave predictably and provide stable timing.

This tutorial provides a clean, working example of halo exchanges using both two-sided and one-sided MPI, along with correctness checking, a 
small stencil computation, performance measurement, and a reusable grid framework. It can be extended with hybrid MPI+OpenMP computation, deeper 
halo layers, more complex stencils, non-blocking RMA epochs, or derived datatypes for larger subdomain transfers.