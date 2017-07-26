#include <iostream>
#include <stdlib>
#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include "mpi.h"
//#include <assert.h>
//#include <limits> 
using std::cout;

int main(int argc, char **argv) {
  // Initialize MPI before Kokkos
  MPI_Init(&argc, &argv);

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  // Initialze grid properties from command line
  if (argc < 5){
    cout << "Need x, y, len_x, len_y" << endl;
    std::exit();
  }

  float cen_x, cen_y, len_x, len_y;
  cen_x = std::stof(std::to_string(argv[1]));
  cen_y = std::stof(std::to_string(argv[2]));
  len_x = std::stof(std::to_string(argv[3]));
  len_y = std::stof(std::to_string(argv[4]));

  Kokkos::finalize();

  MPI_Finalize();

  return 0;
}
