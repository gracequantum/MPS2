/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-12 11:10
* 
* Description: GraceQ/mps2 project. Testing utilities.
*/
#ifndef GQMPS2_TESTING_UTILS_H
#define GQMPS2_TESTING_UTILS_H


#include "gqten/gqten.h"

#include <iostream>
#include <vector>
#include <cstdlib>


using namespace gqten;


inline double Rand(void) { return double(rand()) / RAND_MAX; }


inline void RandRealSymMat(double *mat, long dim) {
  srand(0);
  for (long i = 0; i < dim; ++i) {
    for (long j = 0; j < dim; ++j) {
      mat[(i*dim + j)] = Rand();
    }
  }
  srand(0);
  for (long i = 0; i < dim; ++i) {
    for (long j = 0; j < dim; ++j) {
      mat[(j*dim + i)] += Rand();
    }
  }
}


template<typename T>
inline void PrintVec(const std::vector<T> &v) {
  for (auto &e : v) { std::cout << e << " "; }
  std::cout << "\n";
}
#endif /* ifndef GQMPS2_TESTING_UTILS_H */
