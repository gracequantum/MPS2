// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-29 21:20
*
* Description: GraceQ/MPS2 project. Implementation details for two-site algorithm.
*/

/**
@file two_site_update_finite_vmps_impl.h
@brief Implementation details for two-site finite variational MPS algorithm.
*/
#ifndef GQMPS2_ALGORITHM_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_IMPL_H
#define GQMPS2_ALGORITHM_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_IMPL_H


#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps.h"    // SweepParams
#include "gqmps2/one_dim_tn/mpo/mpo.h"                            // MPO
#include "gqmps2/one_dim_tn/mps/mps.h"                            // MPS
#include "gqmps2/utilities.h"                                     // IsPathExist, CreatPath
#include "gqmps2/one_dim_tn/framework/ten_vec.h"                  // TenVec
#include "gqmps2/consts.h"
#include "gqten/gqten.h"
#include "gqten/utility/timer.h"                                  // Timer

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include <stdio.h>    // remove
#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>


namespace gqmps2 {
using namespace gqten;


// Forward declarations


// Helpers
template <typename DTenT>
inline double MeasureEE(const DTenT &s, const size_t sdim) {
  double ee = 0;
  double p;
  for (size_t i = 0; i < sdim; ++i) {
    p = std::pow(s(i, i), 2.0);
    ee += (-p * std::log(p));
  }
  return ee;
}


inline std::string GenEnvTenName(
    const std::string &dir, const long blk_len, const std::string temp_path
) {
  return temp_path + "/" +
         dir + kEnvFileBaseName + std::to_string(blk_len) +
         "." + kGQTenFileSuffix;
}


inline void RemoveFile(const std::string &file) {
  if (remove(file.c_str())) {
    auto error_msg = "Unable to delete " + file;
    perror(error_msg.c_str());
  }
}


/**
Function to perform two-site update finite vMPS algorithm.

@note The input MPS will be considered an empty one.
*/
template <typename TenElemT, typename QNT>
GQTEN_Double TwoSiteFiniteVMPS(
    MPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const SweepParams &sweep_params
) {
  assert(mps.size() == mpo.size());
  // If the runtime temporary directory does not exit, create it and initialize
  // the left/right environments
  if (!IsPathExist(sweep_params.temp_path)) {
    CreatPath(sweep_params.temp_path);
    InitEnvs(mps, mpo, sweep_params);
  }

  std::cout << "\n";
  GQTEN_Double e0;
  for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
    std::cout << "sweep " << sweep << std::endl;
    e0 = TwoSiteFiniteVMPSSweep(mps, mpo, sweep_params);
    std::cout << "\n";
  }
  return e0;
}


template <typename TenElemT, typename QNT>
void InitEnvs(
    MPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const SweepParams &sweep_params) {
  using TenT = GQTensor<TenElemT, QNT>;
  auto N = mps.size();

  TenT renv;
  for (size_t i = 1; i <= N - 2; ++i) {
    mps.LoadTen(N-i, GenMPSTenName(sweep_params.mps_path, N-i));
    auto file = GenEnvTenName("r", i, sweep_params.temp_path);
    if (i == 1) {
      TenT temp;
      Contract(&mps[N-i], &mpo.back(), {{1}, {0}}, &temp);
      auto mps_ten_dag = Dag(mps[N-i]);
      Contract(&temp, &mps_ten_dag, {{2}, {1}}, &renv);
      WriteGQTensorTOFile(renv, file);
    } else {
      TenT temp1;
      Contract(&mps[N-i], &renv, {{2}, {0}}, &temp1);
      renv = TenT();
      TenT temp2;
      Contract(&temp1, &mpo[N-i], {{1, 2}, {1, 3}}, &temp2);
      auto mps_ten_dag = Dag(mps[N-i]);
      Contract(&temp2, &mps_ten_dag, {{3, 1}, {1, 2}}, &renv);
      WriteGQTensorTOFile(renv, file);
    }
    mps.dealloc(N-i);
  }

}


/**
Function to perform a single two-site finite vMPS sweep.

@note Before the sweep and after the sweep, the MPS is empty.
*/
template <typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSSweep(
    MPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const SweepParams &sweep_params
) {
  auto N = mps.size();
  using TenT = GQTensor<TenElemT, QNT>;
  TenVec<TenT> lenvs(N - 1);
  TenVec<TenT> renvs(N - 1);
  double e0;
  for (size_t i = 0; i < N - 1; ++i) {
    e0 = TwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'r', i);
  }
  for (size_t i = N-1; i > 0; --i) {
    e0 = TwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'l', i);
  }
  return e0;
}


template <typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSUpdate(
    MPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const SweepParams &sweep_params,
    const char dir,
    const size_t target_site
) {
  Timer update_timer("update");

  // Assign some parameters
  auto N = mps.size();
  std::vector<std::vector<size_t>> init_state_ctrct_axes, us_ctrct_axes;
  std::string where;
  size_t svd_ldims;
  size_t lsite_idx, rsite_idx;
  size_t lenv_len, renv_len;
  std::string lblock_file, rblock_file;
  switch (dir) {
    case 'r':
      lsite_idx = target_site;
      rsite_idx = target_site + 1;
      lenv_len = target_site;
      renv_len = N - (target_site + 2);
      if (target_site == 0) {
        init_state_ctrct_axes = {{1}, {0}};
        where = "lend";
        svd_ldims = 1;
      } else if (target_site == N-2) {
        init_state_ctrct_axes = {{2}, {0}};
        where = "rend";
        svd_ldims = 2;
      } else {
        init_state_ctrct_axes = {{2}, {0}};
        where = "cent";
        svd_ldims = 2;
      }
      break;
    case 'l':
      lsite_idx = target_site - 1;
      rsite_idx = target_site;
      lenv_len = target_site - 1;
      renv_len = N - target_site - 1;
      if (target_site == N-1) {
        init_state_ctrct_axes = {{2}, {0}};
        where = "rend";
        svd_ldims = 2;
        us_ctrct_axes = {{2}, {0}};
      } else if (target_site == 1) {
        init_state_ctrct_axes = {{1}, {0}};
        where = "lend";
        svd_ldims = 1;
        us_ctrct_axes = {{1}, {0}};
      } else {
        init_state_ctrct_axes = {{2}, {0}};
        where = "cent";
        svd_ldims = 2;
        us_ctrct_axes = {{2}, {0}};
      }
      break;
    default:
      std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
      exit(1);
  }

  // Load to-be-used tensors
  LoadRelatedTens(mps, lenvs, renvs, target_site, dir, sweep_params);

  // Lanczos
  using TenT = GQTensor<TenElemT, QNT>;
  std::vector<TenT *>eff_ham(4);
  eff_ham[0] = lenvs(lenv_len);
  // Safe const casts for MPO local tensors.
  eff_ham[1] = const_cast<TenT *>(&mpo[lsite_idx]);
  eff_ham[2] = const_cast<TenT *>(&mpo[rsite_idx]);
  eff_ham[3] = renvs(renv_len);
  auto init_state = new TenT;
  Contract(&mps[lsite_idx], &mps[rsite_idx], init_state_ctrct_axes, init_state);
  Timer lancz_timer("Lancz");
  auto lancz_res = LanczosSolver(
                       eff_ham, init_state,
                       sweep_params.lancz_params,
                       where
                   );
  auto lancz_elapsed_time = lancz_timer.Elapsed();

  // SVD and measure entanglement entropy
  TenT u, vt;
  using DTenT = GQTensor<GQTEN_Double, QNT>;
  DTenT s;
  GQTEN_Double actual_trunc_err;
  size_t D;
  SVD(
      lancz_res.gs_vec,
      svd_ldims, Div(mps[lsite_idx]),
      sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
      &u, &s, &vt, &actual_trunc_err, &D
  );
  delete lancz_res.gs_vec;
  auto ee = MeasureEE(s, D);

  // Update MPS local tensor
  TenT the_other_mps_ten;
  switch (dir) {
    case 'r':
      mps[lsite_idx] = std::move(u);
      Contract(&s, &vt, {{1}, {0}}, &the_other_mps_ten);
      mps[rsite_idx] = std::move(the_other_mps_ten);
      break;
    case 'l':
      Contract(&u, &s, us_ctrct_axes, &the_other_mps_ten);
      mps[lsite_idx] = std::move(the_other_mps_ten);
      mps[rsite_idx] = std::move(vt);
      break;
    default:
      assert(false);
  }

  // Update environment tensors
  switch (dir) {
    case 'r':
      if (target_site != N-2) {
        if (target_site == 0) {
          TenT temp, lenv_ten;
          Contract(&mps[target_site], &mpo[target_site], {{0}, {0}}, &temp);
          auto mps_ten_dag = Dag(mps[target_site]);
          Contract(&temp, &mps_ten_dag, {{2}, {0}}, &lenv_ten);
          lenvs[lenv_len + 1] = std::move(lenv_ten);
        } else {
          TenT temp1, temp2, lenv_ten;
          Contract(&lenvs[lenv_len], &mps[target_site], {{0}, {0}}, &temp1);
          Contract(&temp1, &mpo[target_site], {{0, 2}, {0, 1}}, &temp2);
          auto mps_ten_dag = Dag(mps[target_site]);
          Contract(&temp2, &mps_ten_dag, {{0 ,2}, {0, 1}}, &lenv_ten);
          lenvs[lenv_len + 1] = std::move(lenv_ten);
        }
      }
      break;
    case 'l':
      if (target_site != 1) {
        if (target_site == N-1) {
          TenT temp, renv_ten;
          Contract(&mps[target_site], &mpo[target_site], {{1}, {0}}, &temp);
          auto mps_ten_dag = Dag(mps[target_site]);
          Contract(&temp, &mps_ten_dag, {{2}, {1}}, &renv_ten);
          renvs[renv_len + 1] = std::move(renv_ten);
        } else {
          TenT temp1, temp2, renv_ten;
          Contract(&mps[target_site], eff_ham[3], {{2}, {0}}, &temp1);
          Contract(&temp1, &mpo[target_site], {{1, 2}, {1, 3}}, &temp2);
          auto mps_ten_dag = Dag(mps[target_site]);
          Contract(&temp2, &mps_ten_dag, {{3, 1}, {1, 2}}, &renv_ten);
          renvs[renv_len + 1] = std::move(renv_ten);
        }
      }
      break;
    default:
      assert(false);
  }

  // Dump related tensor to HD and remove unused tensor from RAM
  DumpRelatedTens(mps, lenvs, renvs, target_site, dir, sweep_params);

  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << "Site " << std::setw(4) << target_site
            << " E0 = " << std::setw(20) << std::setprecision(kLanczEnergyOutputPrecision) << std::fixed << lancz_res.gs_eng
            << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
            << " D = " << std::setw(5) << D
            << " Iter = " << std::setw(3) << lancz_res.iters
            << " LanczT = " << std::setw(8) << lancz_elapsed_time
            << " TotT = " << std::setw(8) << update_elapsed_time
            << " S = " << std::setw(10) << std::setprecision(7) << ee;
  std::cout << std::scientific << std::endl;
  return lancz_res.gs_eng;
}


template <typename TenElemT, typename QNT>
void LoadRelatedTens(
    MPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const SweepParams &sweep_params
) {
  auto N = mps.size();
  switch (dir) {
    case 'r':
      if (target_site == 0) {
        mps.LoadTen(
            target_site,
            GenMPSTenName(sweep_params.mps_path, target_site)
        );
        mps.LoadTen(
            target_site + 1,
            GenMPSTenName(sweep_params.mps_path, target_site + 1)
        );
        auto renv_len = N - (target_site + 2);
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);
        RemoveFile(renv_file);
      } else if (target_site == N-2) {
        mps.LoadTen(
            target_site + 1,
            GenMPSTenName(sweep_params.mps_path, target_site + 1)
        );
      } else {
        mps.LoadTen(
            target_site + 1,
            GenMPSTenName(sweep_params.mps_path, target_site + 1)
        );
        auto renv_len = N - (target_site + 2);
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);
        RemoveFile(renv_file);
      }
      break;
    case 'l':
      if (target_site == N-1) {
        // Do nothing
      } else if (target_site == 1) {
        mps.LoadTen(
            target_site - 1,
            GenMPSTenName(sweep_params.mps_path, target_site - 1)
        );
      } else {
        mps.LoadTen(
            target_site - 1,
            GenMPSTenName(sweep_params.mps_path, target_site - 1)
        );
        auto lenv_len = (target_site+1) - 2;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        lenvs.LoadTen(lenv_len, lenv_file);
        RemoveFile(lenv_file);
      }
      break;
    default:
      assert(false);
  }
}


template <typename TenElemT, typename QNT>
void DumpRelatedTens(
    MPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const SweepParams &sweep_params
) {
  auto N = mps.size();
  switch (dir) {
    case 'r':
      if (target_site == 0) {
        renvs.dealloc(N - (target_site+2));
        mps.DumpTen(
            target_site,
            GenMPSTenName(sweep_params.mps_path, target_site)
        );
        mps.dealloc(target_site);
        lenvs.DumpTen(
            target_site + 1,
            GenEnvTenName("l", target_site + 1, sweep_params.temp_path)
        );
      } else if (target_site == N-2) {
        mps.DumpTen(
            target_site,
            GenMPSTenName(sweep_params.mps_path, target_site)
        );
      } else {
        lenvs.dealloc(target_site);
        renvs.dealloc(N - (target_site + 2));
        mps.DumpTen(
            target_site,
            GenMPSTenName(sweep_params.mps_path, target_site)
        );
        mps.dealloc(target_site);
        lenvs.DumpTen(
            target_site + 1,
            GenEnvTenName("l", target_site + 1, sweep_params.temp_path)
        );
      }
      break;
    case 'l':
      if (target_site == N - 1) {
        lenvs.dealloc((target_site+1) - 2);
        mps.DumpTen(
            target_site,
            GenMPSTenName(sweep_params.mps_path, target_site)
        );
        mps.dealloc(target_site);
        auto next_renv_len = N - target_site;
        renvs.DumpTen(
            next_renv_len,
            GenEnvTenName("r", next_renv_len, sweep_params.temp_path)
        );
      } else if (target_site == 1) {
        renvs.dealloc(N - (target_site+1));
        mps.DumpTen(
            target_site,
            GenMPSTenName(sweep_params.mps_path, target_site)
        );
        mps.dealloc(target_site);
        mps.DumpTen(
            target_site - 1,
            GenMPSTenName(sweep_params.mps_path, target_site - 1)
        );
        mps.dealloc(target_site - 1);
      } else {
        lenvs.dealloc((target_site+1) - 2);
        renvs.dealloc(N - (target_site+1));
        mps.DumpTen(
            target_site,
            GenMPSTenName(sweep_params.mps_path, target_site)
        );
        mps.dealloc(target_site);
        auto next_renv_len = N - target_site;
        renvs.DumpTen(
            next_renv_len,
            GenEnvTenName("r", next_renv_len, sweep_params.temp_path)
        );
      }
      break;
    default:
      assert(false);
  }
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_ALGORITHM_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_IMPL_H */
