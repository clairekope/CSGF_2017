#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include "mpi.h"
//#include <assert.h>
//#include <limits> 
using std::cout;
using std::endl;
using std::ofstream;

using namespace Kokkos;

struct MandelbrotEv{
  View<unsigned int **> color;
  View<complex<double>**> C;
  int count_x;
  int psize;
  
  MandelbrotEv(View<complex<double> **> C, View<unsigned int **> color,
               int count_x, int psize): 
                C(C), color(color), count_x(count_x), psize(psize) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (int k) const{

    int i = k%count_x;
    int j = k/count_x;

    // Iterate a single pixel
    int iter=0, max = 10000;
    double rad=0.0, rad_max=18.0, d;
    complex<double> Z(0,0), dZ(0,0);

    while (rad<rad_max && iter<max){
      dZ = 2.0*Z*dZ+1;
      Z = Z*Z + C(i,j);
      rad = abs(Z);
      d = 2.0 * ln(abs(Z)) * abs(Z)/abs(dZ);
      ++iter;
      
      // Convert distance to greyscale
      if (d < 0.5*psize) {
        color(i,j) = pow( d/(0.5*psize), 1.0/3.0) * 255;
      }
      else {
        color(i,j) = 255;
      }
    }
};

int main(int argc, char **argv) {
  // Initialize MPI before Kokkos
  MPI_Init(&argc, &argv);

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  {

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

  int count_x = 8192; // pixel count
  float pix_size = len_x/count_x;
  int count_y = len_y/pix_size;

  // Initialize grid
  View<unsigned int**> dcolors("color_grid", count_x, count_y);
  View<complex<double>**> dcmplx("cmplx_grid", count_x, count_y);
  
  auto hcolors = create_mirror_view(dcolors);
  auto hcmplx = create_mirror_view(dcmplx);

  cout << "Fine up to here!" << endl;
  
  // fill cmplx and copy to device
  for (int i=0; i<count_x; ++i) {
    for (int j=0; j<count_y; ++j) {
      //cout << min_x + i*pix_size << max_y - j*pix_size << endl;
      hcmplx(i,j) = complex<double>(min_x + i*pix_size, max_y - j*pix_size);
    }
  }
  deep_copy(dcmplx, hcmplx); // Destination, source
cout << "cmplx copy" << endl;  

  // Solve Mandelbrot
  int prod = count_x*count_y;
  MandelbrotEv grid(dcmplx, dcolors, count_x);
  parallel_for(prod, grid);
  
cout << "parallel for" << endl;
  // deep copy back to host
  deep_copy(hcolors, dcolors);
  
cout << "colors copy" << endl;
  // write
  FILE * fid = fopen("image.txt","wb");
  fprintf(fid,"P5\n");
  fprintf(fid,"%i %i\n",count_x,count_y);
  fprintf(fid,"255\n");
  for (int j=0; j<count_y; ++j) {
    for (int i=0; i<count_x; ++i) {
        unsigned char tmp = hcolors(i,j);
      fwrite(&tmp,sizeof(unsigned char),1,fid);      
    }
  }
 fclose(fid); 
 }
  Kokkos::finalize();

  MPI_Finalize();

  return 0;
}

