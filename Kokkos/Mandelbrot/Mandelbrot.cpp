#include <iostream>
#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include "mpi.h"
//#include <assert.h>
//#include <limits> 
using std::cout;
using std::endl;

int main(int argc, char **argv) {
  // Initialize MPI before Kokkos
  MPI_Init(&argc, &argv);

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  // Initialze grid properties from command line
  if (argc < 5){
    cout << "Need x, y, len_x, len_y" << endl;
    exit(1);
  }

  float cen_x, cen_y, len_x, len_y;
  cen_x = std::atoi(argv[1]);
  cen_y = std::atoi(argv[2]);
  len_x = std::atoi(argv[3]);
  len_y = std::atoi(argv[4]);

  Kokkos::finalize();

  MPI_Finalize();

  return 0;
}
