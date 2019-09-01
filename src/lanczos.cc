/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 21:44
* 
* Description: GraceQ/mps2 project. Private objects for Lanczos solver,
*              implementation.
*/
#include "lanczos.h"
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <iostream>
#include <cstring>

#include "mkl.h"


namespace gqmps2 {
using namespace gqten;


LanczosRes LanczosSolver(
    const std::vector<GQTensor *> &rpeff_ham, GQTensor *pinit_state,
    const LanczosParams &params,
    const std::string &where) {
  // Take care that init_state will be destroyed after call the solver.
  long eff_ham_eff_dim = 1;
  GQTensor *(* eff_ham_mul_state)(
      const std::vector<GQTensor *> &, GQTensor *) = nullptr;
  std::vector<std::vector<long>> energy_measu_ctrct_axes;
  LanczosRes lancz_res;

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

  std::vector<GQTensor *> bases(params.max_iterations);
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
      *last_mat_mul_vec_res, MockDag(*bases[0]),
      energy_measu_ctrct_axes);
  a[0] = temp_scalar_ten->scalar; delete temp_scalar_ten;
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
        lancz_res.gs_vec = new GQTensor(*bases[0]);
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        return lancz_res;
      } else {
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
        auto gs_vec = new GQTensor(bases[0]->indexes);
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
        *last_mat_mul_vec_res, MockDag(*bases[m]),
        energy_measu_ctrct_axes);
    a[m] = temp_scalar_ten->scalar; delete temp_scalar_ten;
    TridiagGsSolver(a, b, m+1, eigval, eigvec, 'N');
    auto energy0_new = eigval;
    if (((energy0 - energy0_new) < params.error) ||
         (m == eff_ham_eff_dim) ||
         (m == params.max_iterations - 1)) {
      TridiagGsSolver(a, b, m+1, eigval, eigvec, 'V');
      energy0 = energy0_new;
      auto gs_vec = new GQTensor(bases[0]->indexes);
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


GQTensor *eff_ham_mul_state_cent(
    const std::vector<GQTensor *> &eff_ham, GQTensor *state) {
  auto res = Contract(*eff_ham[0], *state, {{0}, {0}});
  InplaceContract(res, *eff_ham[1], {{0, 2}, {0, 1}});
  InplaceContract(res, *eff_ham[2], {{4, 1}, {0, 1}});
  InplaceContract(res, *eff_ham[3], {{4, 1}, {1, 0}});
  return res;
}


GQTensor *eff_ham_mul_state_lend(
    const std::vector<GQTensor *> &eff_ham, GQTensor *state) {
  auto res = Contract(*state, *eff_ham[1], {{0}, {0}});
  InplaceContract(res, *eff_ham[2], {{0, 2}, {1, 0}});
  InplaceContract(res, *eff_ham[3], {{0, 3}, {0, 1}});
  return res;
}


GQTensor *eff_ham_mul_state_rend(
    const std::vector<GQTensor *> &eff_ham, GQTensor *state) {
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
