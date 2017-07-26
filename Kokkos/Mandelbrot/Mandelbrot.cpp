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

using namespace Kokkos;

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
  cen_x = std::atof(argv[1]);
  cen_y = std::atof(argv[2]);
  len_x = std::atof(argv[3]);
  len_y = std::atof(argv[4]);

  float min_x = cen_x - len_x/2.0;
  float max_y = cen_y + len_y/2.0;

  float count_x = 8192; // pixel count
  float pix_size = len_x/count_x;
  float count_y = len_y/pix_size;

  // Initialize grid
  View<float**> grid("grid", count_x, count_y);

  Kokkos::finalize();

  MPI_Finalize();

  return 0;
}

struct MandelbrotEv{
  unsigned int *color;
  complex<double> *C;
  MandelbrotEv(complex<double> *C, unsigned int *color): C{C}, color{color} {}

  KOKKOS_INLINE_FUNCTION
  void operator() (int i) const{
    // actual function operations
    color = 2;
  }
};
