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

struct MandelbrotEv{
  View<unsigned int **> color;
  View<complex<double>**> C;
  int count_x;
  
  MandelbrotEv(View<Kokkos::complex<double> **> C, View<unsigned int **> color,
               int count_x): C(C), color(color), count_x(count_x) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (int k) const{

    int i = k%count_x;
    int j = k/count_x;

    // Iterate a single pixel

    int iter=0, max = 1000;
    double rad=0.0, rad_max=2.0;
    complex<double> Z(0,0);

    while (rad<rad_max && iter<max){
      Z = Z*Z + C(i,j);
      rad = abs(Z);
      ++iter;
      if (iter<max) {
        color(i,j) = 0;
      }
      else {
        color(i,j) = 1;
      }
    }
  }
};

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
  View<unsigned int**> dcolors("color_grid", count_x, count_y);
  View<complex<double>**> dcmplx("cmplx_grid", count_x, count_y);
  
  auto hcolors = create_mirror_view(dcolors);
  auto hcmplx = create_mirror_view(dcmplx);
  
  // fill cmplx and copy to device
  for (int i=0; i<count_y; ++i) {
    for (int j=0; j<count_x; ++j) {
      hcmplx(i,j) = complex<double>(min_x + i*pix_size, max_y - j*pix_size);
    }
  }
  deep_copy(dcmplx, hcmplx); // Destination, source
  
  // Solve Mandelbrot
  int prod = count_x*count_y;
  MandelbrotEv grid(dcmplx, dcolors, count_x);
  parallel_for(prod, grid);
  
  // deep copy back to host
  deep_copy(hcolors, dcolors);
  
  // write
  fstream output("output.out");
  output << 'P5' << endl;
  output << count_x << count_y << endl;
  output << 255 << endl;
  for (int i=0; i<count_y; ++i) {
    for (int j=0; j<count_x; ++j) {
      output << hcolors(i,j);
    }
  } 
  output.close()

  Kokkos::finalize();

  MPI_Finalize();

  return 0;
}

