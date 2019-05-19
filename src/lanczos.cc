/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 21:44
* 
* Description: GraceQ/mps2 project. Private objects for Lanczos solver,
*              implementation.
*/
#include "lanczos.h"
#include "gqten/gqten.h"

#include <iostream>

#include <cstring>

#include "mkl.h"


namespace gqmps2 {
using namespace gqten;


GQTensor *eff_ham_mul_state_cent(
    const std::vector<GQTensor *> &eff_ham, const GQTensor *state) {
  auto res = Contract(*eff_ham[0], *state, {{0}, {0}});
  InplaceContract(res, *eff_ham[1], {{0, 2}, {0, 1}});
  InplaceContract(res, *eff_ham[2], {{4, 1}, {0, 1}});
  InplaceContract(res, *eff_ham[3], {{4, 1}, {1, 0}});
  return res;
}


GQTensor *eff_ham_mul_state_lend(
    const std::vector<GQTensor *> &eff_ham, const GQTensor *state) {
  auto res = Contract(*state, *eff_ham[1], {{0}, {0}});
  InplaceContract(res, *eff_ham[2], {{0, 2}, {1, 0}});
  InplaceContract(res, *eff_ham[3], {{0, 3}, {0, 1}});
  return res;
}


GQTensor *eff_ham_mul_state_rend(
    const std::vector<GQTensor *> &eff_ham, const GQTensor *state) {
  auto res = Contract(*state, *eff_ham[0], {{0}, {0}});
  InplaceContract(res, *eff_ham[1], {{2, 0}, {0, 1}});
  InplaceContract(res, *eff_ham[2], {{3, 0}, {1,0}});
  return res;
}


void TridiagGsSolver(
    const std::vector<double> &a, const std::vector<double> &b, const long n,
    double &gs_eng, double * &gs_vec, const char jobz) {
  auto d = new double [n];
  std::memcpy(d, a.data(), n*sizeof(double));
  auto e = new double [n-1];
  std::memcpy(e, b.data(), (n-1)*sizeof(double));
  long ldz;
  auto stev_err_msg = "?stev error.";
  auto stev_jobz_err_msg = "jobz must be  'N' or 'V', but ";
  switch (jobz) {
    case 'N':
      ldz = 1;
      break;
    case 'V':
      ldz = n;
      break;
    default:
      std::cout << stev_jobz_err_msg << jobz << std::endl; 
      std::cout << stev_err_msg << std::endl;
      exit(1);
  }
  auto z = new double [ldz*n];
  auto info = LAPACKE_dstev(    // TODO: Try dstevd dstevx some day.
                  LAPACK_ROW_MAJOR, jobz,
                  n,
                  d, e,
                  z,
                  n);     // TODO: Why can not use ldz???
  if (info != 0) {
    std::cout << stev_err_msg << std::endl;
    exit(1);
  }
  switch (jobz) {
    case 'N':
      gs_eng = d[0];
      delete [] d;
      delete [] e;
      delete [] z;
      break;
    case 'V':
      gs_eng = d[0];
      gs_vec = new double [n];
      for (long i = 0; i < n; ++i) { gs_vec[i] = z[i*n]; }
      delete [] d;
      delete [] e;
      delete [] z;
      break;
    default:
      std::cout << stev_jobz_err_msg << jobz << std::endl; 
      std::cout << stev_err_msg << std::endl;
      exit(1);
  }
}
} /* gqmps2 */ 
