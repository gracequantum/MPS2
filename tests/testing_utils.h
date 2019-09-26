// SPDX-License-Identifier: LGPL-3.0-only
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

#include "mkl.h"


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


inline void RandCplxHerMat(GQTEN_Complex *mat, long dim) {
  for (long i = 0; i < dim; ++i) {
    for (long j = 0; j < i; ++j) {
      GQTEN_Complex elem(Rand(), Rand());
      mat[(i*dim + j)] = elem;
      mat[(j*dim + i)] = std::conj(elem);
    }
  }
  for (long i = 0; i < dim; ++i) {
    mat[i*dim + i] = Rand();
  }
}


inline void LapackeSyev(
    int matrix_layout, char jobz, char uplo,
    MKL_INT n, double *a, MKL_INT lda, double *w) {
  LAPACKE_dsyev(matrix_layout, jobz, uplo, n, a, lda, w);
}


inline void LapackeSyev(
    int matrix_layout, char jobz, char uplo,
    MKL_INT n, GQTEN_Complex *a, MKL_INT lda, double *w) {
  LAPACKE_zheev(matrix_layout, jobz, uplo, n, a, lda, w);
}
#endif /* ifndef GQMPS2_TESTING_UTILS_H */
