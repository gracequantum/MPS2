// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-24 17:46
* 
* Description: GraceQ/MPS2 project. Implementation details for Lanczos solver.
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <iostream>
#include <cstring>

#include "mkl.h"


namespace gqmps2 {


using namespace gqten;


// Forward declarations.
template <typename TenElemType>
GQTensor<TenElemType> *eff_ham_mul_state_cent(
    const std::vector<GQTensor<TenElemType> *> &, GQTensor<TenElemType> *);

template <typename TenElemType>
GQTensor<TenElemType> *eff_ham_mul_state_lend(
    const std::vector<GQTensor<TenElemType> *> &, GQTensor<TenElemType> *);

template <typename TenElemType>
GQTensor<TenElemType> *eff_ham_mul_state_rend(
    const std::vector<GQTensor<TenElemType> *> &, GQTensor<TenElemType> *);

void TridiagGsSolver(
    const std::vector<double> &, const std::vector<double> &, const long,
    double &, double * &, const char);


// Helpers.
template <typename TenElemType>
inline void InplaceContract(
    GQTensor<TenElemType> * &lhs, const GQTensor<TenElemType> &rhs,
    const std::vector<std::vector<long>> &axes) {
  auto res = Contract(*lhs, rhs, axes);
  delete lhs;
  lhs = res;
}


template <typename TenElemType>
inline void LanczosFree(
    double * &a,
    std::vector<GQTensor<TenElemType> *> &b,
    GQTensor<TenElemType> * &last_mat_mul_vec_res) {
  if (a != nullptr) { delete [] a; }
  for (auto &ptr : b) { delete ptr; }
  delete last_mat_mul_vec_res;
}


inline double Real(const double d) { return d; }


inline double Real(const GQTEN_Complex z) { return z.real(); }


// Lanczos solver.
//
// Effective Hamiltonian setup:
//
// +-----+          2             2        +-----+
// |      0        +             +        2      |
// |               |             |               |
// +-----+      +-----+       +-----+      +-----+
// |      1    0   |   3     0   |   3    1      |
// |               +             +               |
// +-----+          1             1        +-----+
//        2                               0
//                 + 1           + 2
//                 |             |
//                 |             |
//             +---------------------+
//            0                       3
//
template <typename TenElemType>
LanczosRes<TenElemType> LanczosSolver(
    const std::vector<GQTensor<TenElemType> *> &rpeff_ham,
    GQTensor<TenElemType> *pinit_state,
    const LanczosParams &params,
    const std::string &where) {
  // Take care that init_state will be destroyed after call the solver.
  long eff_ham_eff_dim = 1;
  GQTensor<TenElemType> *(* eff_ham_mul_state)(
      const std::vector<GQTensor<TenElemType> *> &,
      GQTensor<TenElemType> *) = nullptr;
  std::vector<std::vector<long>> energy_measu_ctrct_axes;
  LanczosRes<TenElemType> lancz_res;

  // Calculate position dependent parameters.
  if (where == "cent") {
    eff_ham_eff_dim *= rpeff_ham[0]->indexes[0].dim;
    eff_ham_eff_dim *= rpeff_ham[1]->indexes[1].dim;
    eff_ham_eff_dim *= rpeff_ham[2]->indexes[1].dim;
    eff_ham_eff_dim *= rpeff_ham[3]->indexes[0].dim;
    eff_ham_mul_state = &eff_ham_mul_state_cent;
    energy_measu_ctrct_axes = {{0, 1, 2, 3}, {0, 1, 2, 3}};
  } else if (where == "lend") {
    eff_ham_eff_dim *= rpeff_ham[1]->indexes[0].dim;
    eff_ham_eff_dim *= rpeff_ham[2]->indexes[1].dim;
    eff_ham_eff_dim *= rpeff_ham[3]->indexes[0].dim;
    eff_ham_mul_state = &eff_ham_mul_state_lend;
    energy_measu_ctrct_axes = {{0, 1, 2}, {0, 1, 2}};
  } else if (where ==  "rend") {
    eff_ham_eff_dim *= rpeff_ham[0]->indexes[0].dim;
    eff_ham_eff_dim *= rpeff_ham[1]->indexes[1].dim;
    eff_ham_eff_dim *= rpeff_ham[2]->indexes[0].dim;
    eff_ham_mul_state = &eff_ham_mul_state_rend;
    energy_measu_ctrct_axes = {{0, 1, 2}, {0, 1, 2}};
  }

  std::vector<GQTensor<TenElemType> *> bases(params.max_iterations);
  std::vector<double> a(params.max_iterations, 0.0);
  std::vector<double> b(params.max_iterations, 0.0);
  std::vector<double> N(params.max_iterations, 0.0);

  // Initialize Lanczos iteration.
  pinit_state->Normalize();
  bases[0] =  pinit_state;

#ifdef GQMPS2_TIMING_MODE
  Timer mat_vec_timer("mat_vec");
  mat_vec_timer.Restart();
#endif

  auto last_mat_mul_vec_res = (*eff_ham_mul_state)(rpeff_ham, bases[0]);

#ifdef GQMPS2_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif

  auto temp_scalar_ten = Contract(
      *last_mat_mul_vec_res, Dag(*bases[0]),
      energy_measu_ctrct_axes);
  a[0] = Real(temp_scalar_ten->scalar); delete temp_scalar_ten;
  N[0] = 0.0;
  long m = 0;
  double energy0;
  energy0 = a[0];
  // Lanczos iterations.
  while (true) {
    m += 1;
    auto gamma = last_mat_mul_vec_res;
    if (m == 1) {
      LinearCombine(
          {-a[m-1]},
          {bases[m-1]},
          gamma);
    } else {
      LinearCombine(
          {-a[m-1], -std::sqrt(N[m-1])},
          {bases[m-1], bases[m-2]},
          gamma);
    }
    auto norm_gamma = gamma->Normalize();
    double eigval;
    double *eigvec = nullptr;
    if (norm_gamma == 0.0) {
      if (m == 1) {
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = new GQTensor<TenElemType>(*bases[0]);
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        return lancz_res;
      } else {
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
        auto gs_vec = new GQTensor<TenElemType>(bases[0]->indexes);
        LinearCombine(m, eigvec, bases, gs_vec);
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = gs_vec;
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        return lancz_res;
      }
    }
    N[m] = std::pow(norm_gamma, 2.0);
    b[m-1] = norm_gamma;
    bases[m] = gamma;

#ifdef GQMPS2_TIMING_MODE
    mat_vec_timer.Restart();
#endif

    last_mat_mul_vec_res = (*eff_ham_mul_state)(rpeff_ham, bases[m]);

#ifdef GQMPS2_TIMING_MODE
    mat_vec_timer.PrintElapsed();
#endif

    auto temp_scalar_ten = Contract(
        *last_mat_mul_vec_res, Dag(*bases[m]),
        energy_measu_ctrct_axes);
    a[m] = Real(temp_scalar_ten->scalar); delete temp_scalar_ten;
    TridiagGsSolver(a, b, m+1, eigval, eigvec, 'N');
    auto energy0_new = eigval;
    if (((energy0 - energy0_new) < params.error) ||
         (m == eff_ham_eff_dim) ||
         (m == params.max_iterations - 1)) {
      TridiagGsSolver(a, b, m+1, eigval, eigvec, 'V');
      energy0 = energy0_new;
      auto gs_vec = new GQTensor<TenElemType>(bases[0]->indexes);
      LinearCombine(m+1, eigvec, bases, gs_vec);
      lancz_res.iters = m;
      lancz_res.gs_eng = energy0;
      lancz_res.gs_vec = gs_vec;
      LanczosFree(eigvec, bases, last_mat_mul_vec_res);
      return lancz_res;
    } else {
      energy0 = energy0_new;
    }
  }
}


template <typename TenElemType>
GQTensor<TenElemType> *eff_ham_mul_state_cent(
    const std::vector<GQTensor<TenElemType> *> &eff_ham,
    GQTensor<TenElemType> *state) {
  auto res = Contract(*eff_ham[0], *state, {{2}, {0}});
  res->Transpose({0, 3, 4, 2, 1});
  InplaceContract(res, *eff_ham[1], {{4, 3}, {0, 1}});
  res->Transpose({0, 3, 2, 1, 4});
  InplaceContract(res, *eff_ham[2], {{4, 3}, {0, 1}});
  res->Transpose({0, 1, 3, 4, 2});
  InplaceContract(res, *eff_ham[3], {{4, 3}, {0, 1}});
  return res;
}


template <typename TenElemType>
GQTensor<TenElemType> *eff_ham_mul_state_lend(
    const std::vector<GQTensor<TenElemType> *> &eff_ham,
    GQTensor<TenElemType> *state) {
  auto res = Contract(*eff_ham[1], *state, {{0}, {0}});
  res->Transpose({1, 3, 2, 0});
  InplaceContract(res, *eff_ham[2], {{3, 2}, {0, 1}});
  res->Transpose({0, 2, 3, 1});
  InplaceContract(res, *eff_ham[3], {{3, 2}, {0, 1}});
  return res;
}


template <typename TenElemType>
GQTensor<TenElemType> *eff_ham_mul_state_rend(
    const std::vector<GQTensor<TenElemType> *> &eff_ham,
    GQTensor<TenElemType> *state) {
  auto res = Contract(*eff_ham[0], *state, {{2}, {0}});
  res->Transpose({0, 3, 2, 1});
  InplaceContract(res, *eff_ham[1], {{3, 2}, {0, 1}});
  res->Transpose({0, 2, 3, 1});
  InplaceContract(res, *eff_ham[2], {{3, 2}, {0, 1}});
  return res;
}


inline void TridiagGsSolver(
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
