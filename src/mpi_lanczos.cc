// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-07-01 10:48
* 
* Description: GraceQ/mps2 project. Private objects for MPI parallel Lanczos solver,
*              implementation.
*/
#include "mpi_lanczos.h"
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include "mpi.h"


namespace gqmps2 {
using namespace gqten;


LanczosRes GQMPS2_MPI_LanczosSolver(
    const std::vector<GQTensor *> &rpeff_ham, GQTensor *pinit_state,
    const LanczosParams &params,
    const std::string &where,
    MPI_Comm comm, const int workers) {
  // Take care that init_state will be destroyed after call the solver.
  long eff_ham_eff_dim = 1;
  GQTensor *(* gqmps2_mpi_eff_ham_mul_state)(
      const std::vector<GQTensor *> &, GQTensor *,
      MPI_Comm, const int) = nullptr;
  std::vector<std::vector<long>> energy_measu_ctrct_axes;
  LanczosRes lancz_res;

  // Calculate position dependent parameters.
  if (where == "cent") {
    eff_ham_eff_dim *= rpeff_ham[0]->indexes[0].dim;
    eff_ham_eff_dim *= rpeff_ham[1]->indexes[1].dim;
    eff_ham_eff_dim *= rpeff_ham[2]->indexes[1].dim;
    eff_ham_eff_dim *= rpeff_ham[3]->indexes[0].dim;
    gqmps2_mpi_eff_ham_mul_state = &gqmps2_mpi_eff_ham_mul_state_cent;
    energy_measu_ctrct_axes = {{0, 1, 2, 3}, {0, 1, 2, 3}};
  } else if (where == "lend") {
    eff_ham_eff_dim *= rpeff_ham[1]->indexes[0].dim;
    eff_ham_eff_dim *= rpeff_ham[2]->indexes[1].dim;
    eff_ham_eff_dim *= rpeff_ham[3]->indexes[0].dim;
    gqmps2_mpi_eff_ham_mul_state = &gqmps2_mpi_eff_ham_mul_state_lend;
    energy_measu_ctrct_axes = {{0, 1, 2}, {0, 1, 2}};
  } else if (where ==  "rend") {
    eff_ham_eff_dim *= rpeff_ham[0]->indexes[0].dim;
    eff_ham_eff_dim *= rpeff_ham[1]->indexes[1].dim;
    eff_ham_eff_dim *= rpeff_ham[2]->indexes[0].dim;
    gqmps2_mpi_eff_ham_mul_state = &gqmps2_mpi_eff_ham_mul_state_rend;
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

  auto last_mat_mul_vec_res = (*gqmps2_mpi_eff_ham_mul_state)(
                                  rpeff_ham, bases[0],
                                  comm, workers);

#ifdef GQMPS2_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif

  auto temp_scalar_ten = GQTEN_MPI_Contract(
      *last_mat_mul_vec_res, MockDag(*bases[0]),
      energy_measu_ctrct_axes,
      comm, workers);
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

    last_mat_mul_vec_res = (*gqmps2_mpi_eff_ham_mul_state)(
                               rpeff_ham, bases[m],
                               comm, workers);

#ifdef GQMPS2_TIMING_MODE
    mat_vec_timer.PrintElapsed();
#endif

    auto temp_scalar_ten = GQTEN_MPI_Contract(
        *last_mat_mul_vec_res, MockDag(*bases[m]),
        energy_measu_ctrct_axes,
        comm, workers);
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


GQTensor *gqmps2_mpi_eff_ham_mul_state_cent(
    const std::vector<GQTensor *> &eff_ham, GQTensor *state,
    MPI_Comm comm, const int workers) {
  auto res = GQTEN_MPI_Contract(*eff_ham[0], *state, {{0}, {0}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[1], {{0, 2}, {0, 1}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[2], {{4, 1}, {0, 1}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[3], {{4, 1}, {1, 0}}, comm, workers);
  return res;
}


GQTensor *gqmps2_mpi_eff_ham_mul_state_lend(
    const std::vector<GQTensor *> &eff_ham, GQTensor *state,
    MPI_Comm comm, const int workers) {
  auto res = GQTEN_MPI_Contract(*state, *eff_ham[1], {{0}, {0}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[2], {{0, 2}, {1, 0}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[3], {{0, 3}, {0, 1}}, comm, workers);
  return res;
}


GQTensor *gqmps2_mpi_eff_ham_mul_state_rend(
    const std::vector<GQTensor *> &eff_ham, GQTensor *state,
    MPI_Comm comm, const int workers) {
  auto res = GQTEN_MPI_Contract(*state, *eff_ham[0], {{0}, {0}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[1], {{2, 0}, {0, 1}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[2], {{3, 0}, {1, 0}}, comm, workers);
  return res;
}
} /* gqmps2 */ 
