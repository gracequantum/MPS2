/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 14:42
* 
* Description: GraceQ/mps2 project. The main source code file.
*/
#include "lanczos.h"
#include "gqmps2/gqmps2.h"

#include <iostream>
#include <iomanip>
#include <cmath>


namespace gqmps2 {


LanczosRes LanczosSolver(
    const std::vector<GQTensor *> &rpeff_ham, GQTensor *pinit_state,
    const LanczosParams &params,
    const std::string &where) {
  // Take care that init_state will be destroyed after call the solver.
  long eff_ham_eff_dim = 1;
  GQTensor *(* eff_ham_mul_state)(
      const std::vector<GQTensor *> &, const GQTensor *) = nullptr;
  std::vector<std::vector<long>> energy_measu_ctrct_axes;
  LanczosRes lancz_res;
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
  pinit_state->Normalize();
  bases[0] =  pinit_state;
  auto temp_mat_mul_vec_res = (*eff_ham_mul_state)(rpeff_ham, bases[0]);
  a[0] = Contract(
             *temp_mat_mul_vec_res,
             Dag(*bases[0]),    // Dag not efficient now, look for move constructor.
             energy_measu_ctrct_axes)->scalar;
  delete temp_mat_mul_vec_res;
  N[0] = 0.0;
  long m;
  m = 0;
  double energy0;
  energy0 = a[0];
  std::cout << m << "\t" <<  std::setprecision(kLanczEnergyOutputPrecision) << energy0 << std::endl;
  while (true) {
    m += 1; 
    GQTensor *gamma;
    if (m == 1) {
      gamma = (*eff_ham_mul_state)(rpeff_ham, bases[m-1]);
      (*gamma) -= a[m-1]*(*bases[m-1]);
    } else {
      gamma = (*eff_ham_mul_state)(rpeff_ham, bases[m-1]);
      (*gamma) -= a[m-1]*(*bases[m-1]);
      (*gamma) -= std::sqrt(N[m-1])*(*bases[m-2]);
    }
    auto norm_gamma = gamma->Norm();
    double eigval;
    double *eigvec = nullptr;
    if (norm_gamma == 0.0) {
      if (m == 1) {
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = new GQTensor(*bases[0]);
        LanczosFree(eigvec,  bases);
        return lancz_res;
      } else {
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
        auto gs_vec = new GQTensor(eigvec[0]*(*bases[0]));
        for (long i = 1; i < m; ++i) {
          *gs_vec += (eigvec[i] * (*bases[i]));
        }
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = gs_vec;
        LanczosFree(eigvec, bases);
        return lancz_res;
      }
    }
    N[m] = std::pow(norm_gamma, 2.0);
    b[m-1] = norm_gamma;
    gamma->Normalize();
    bases[m] = gamma;
    a[m] = Contract(
               *(*eff_ham_mul_state)(rpeff_ham, bases[m]),
               Dag(*bases[m]),    // Dag not efficient now, look for move constructor.
               energy_measu_ctrct_axes)->scalar;
    TridiagGsSolver(a, b, m+1, eigval, eigvec, 'N');
    auto energy0_new = eigval;
    std::cout << m << "\t" << std::setprecision(kLanczEnergyOutputPrecision) << energy0_new << std::endl;
    if (((energy0 - energy0_new) < params.error) || 
         (m == eff_ham_eff_dim) ||
         (m == params.max_iterations - 1)) {
      TridiagGsSolver(a, b, m+1, eigval, eigvec, 'V');
      energy0 = energy0_new;
      auto gs_vec = new GQTensor(eigvec[0]*(*bases[0]));
      for (long i = 0; i < m+1; ++i) {
        *gs_vec += (eigvec[i] * (*bases[i]));
      }
      lancz_res.gs_eng = energy0;
      lancz_res.gs_vec = gs_vec;
      LanczosFree(eigvec, bases);
      return lancz_res;
    } else {
      energy0 = energy0_new;
    }
  }
}

} /* gqmps2 */ 
