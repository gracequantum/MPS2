/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 14:42
* 
* Description: GraceQ/mps2 project. The main source code file.
*/
#include "mpogen.h"
#include "lanczos.h"
#include "two_site_algo.h"
#include "gqmps2/gqmps2.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>


namespace gqmps2 {


// MPO generation.
MPOGenerator::MPOGenerator(const long N, const Index &pb) :
    N_(N), pb_out_(pb) {
  pb_in_ = InverseIndex(pb_out_);
  auto null_edges = std::vector<FSMEdge>();
  edges_set_ = std::vector<std::vector<FSMEdge>>(N_, null_edges);
  mid_state_nums_ = std::vector<long>(N_, 0);
  auto id_op = GQTensor({pb_in_, pb_out_});
  for (long i = 0; i < pb_out_.dim; ++i) { id_op({i, i}) = 1; }
  id_op_ = id_op;
}


void MPOGenerator::AddTerm(
    const double coef,
    const std::vector<OpIdx> &opidxs,
    const GQTensor &inter_op) {
  switch (opidxs.size()) {
    case 1:
      AddOneSiteTerm(coef, opidxs[0]);
      break;
    case 2:
      AddTwoSiteTerm(coef, opidxs[0], opidxs[1], inter_op);
      break;
    default:
      std::cout << "Unsupport term type." << std::endl;
     exit(1); 
  }
}


std::vector<GQTensor *> MPOGenerator::Gen(void) {
  std::vector<GQTensor *> mpo(N_);
  for (long i = 0; i < N_; ++i) {
    if (i == 0) {
      mpo[i] = GenHeadMpo(edges_set_[i], mid_state_nums_[i+1]);
    } else if (i == N_-1) {
      mpo[i] = GenTailMpo(edges_set_[i], mid_state_nums_[i]);
    } else {
      mpo[i] = GenCentMpo(
                   edges_set_[i],
                   mid_state_nums_[i], mid_state_nums_[i+1]);
    }
  }
  return mpo;
}


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
             MockDag(*bases[0]),
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
               MockDag(*bases[m]),
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


double TwoSiteAlgorithm(
    std::vector<GQTensor *> &mps, const std::vector<GQTensor *> &mpo,
    const SweepParams &sweep_params) {
  if ( sweep_params.FileIO && !IsPathExist(kRuntimeTempPath)) {
    CreatPath(kRuntimeTempPath);
  }
  auto rblocks = InitRBlocks(mps, mpo, sweep_params);
  auto N = mps.size();
  std::vector<GQTensor *> lblocks(N-1);
  if (sweep_params.FileIO) {
    auto file = kRuntimeTempPath + "/" +
                "l" + kBlockFileBaseName + "0" + "." + kGQTenFileSuffix; 
    WriteGQTensorTOFile(GQTensor(), file);
  }
  std::cout << "\n";
  double e0;
  for (long sweep = 0; sweep < sweep_params.Sweeps; ++sweep) {
    std::cout << "sweep " << sweep << std::endl;
    e0 = TwoSiteSweep(mps, mpo, lblocks, rblocks, sweep_params);
  }
  return e0;
}
} /* gqmps2 */ 
