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
#include "gqmps2/one_dim_tn/mps/finite_mps/finite_mps.h"          // FiniteMPS
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


// Helpers
template <typename DTenT>
inline double MeasureEE(const DTenT &s, const size_t sdim) {
  double ee = 0;
  double p;
  double singluar_value;
  for (size_t i = 0; i < sdim; ++i) {
    singluar_value = s(i, i);
    p = singluar_value*singluar_value;
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
@note The canonical center of input MPS should be set at site 0 or 1.
@note The canonical center of output MPS is set at site 1.
*/
template <typename TenElemT, typename QNT>
GQTEN_Double TwoSiteFiniteVMPS(
    FiniteMPS<TenElemT, QNT> &mps,
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
  mps.LoadTen(1, GenMPSTenName(sweep_params.mps_path, 1));
  for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
    std::cout << "sweep " << sweep << std::endl;
    Timer sweep_timer("sweep");
    e0 = TwoSiteFiniteVMPSSweep(mps, mpo, sweep_params);
    sweep_timer.PrintElapsed();
    std::cout << "\n";
  }
  mps.DumpTen(1, GenMPSTenName(sweep_params.mps_path, 1), true);
  return e0;
}


template <typename TenElemT, typename QNT>
void InitEnvs(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const SweepParams &sweep_params,
    const size_t update_site_num = 2
){
  InitEnvs(
      mps,
      mpo,
      sweep_params.mps_path,
      sweep_params.temp_path,
      update_site_num
  );
}


template <typename TenElemT, typename QNT>
void InitEnvs(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const std::string mps_path,
    const std::string temp_path,
    const size_t update_site_num = 2
) {
  using TenT = GQTensor<TenElemT, QNT>;
  auto N = mps.size();

  TenT renv;
  //Write a trivial right environment tensor to disk
  mps.LoadTen(N-1, GenMPSTenName(mps_path, N-1));
  auto mps_trivial_index = mps.back().GetIndexes()[2];
  auto mpo_trivial_index_inv = InverseIndex(mpo.back().GetIndexes()[3]);
  auto mps_trivial_index_inv = InverseIndex(mps_trivial_index);
  renv = TenT({mps_trivial_index_inv, mpo_trivial_index_inv, mps_trivial_index});
  renv({0, 0, 0}) = 1;
  auto file = GenEnvTenName("r", 0, temp_path);
  WriteGQTensorTOFile(renv, file);

  //bulk right environment tensors
  for (size_t i = 1; i <= N - update_site_num; ++i) {
    if (i>1) { mps.LoadTen(N-i, GenMPSTenName(mps_path, N-i)); }
    auto file = GenEnvTenName("r", i, temp_path);
    TenT temp1;
    Contract(&mps[N-i], &renv, {{2}, {0}}, &temp1);
    renv = TenT();
    TenT temp2;
    Contract(&temp1, &mpo[N-i], {{1, 2}, {1, 3}}, &temp2);
    auto mps_ten_dag = Dag(mps[N-i]);
    Contract(&temp2, &mps_ten_dag, {{3, 1}, {1, 2}}, &renv);
    WriteGQTensorTOFile(renv, file);
    mps.dealloc(N-i);
  }

  //Write a trivial left environment tensor to disk
  TenT lenv;
  mps.LoadTen(0, GenMPSTenName(mps_path, 0));
  mps_trivial_index = mps.front().GetIndexes()[0];
  mpo_trivial_index_inv = InverseIndex(mpo.front().GetIndexes()[0]);
  mps_trivial_index_inv = InverseIndex(mps_trivial_index);
  lenv = TenT({mps_trivial_index_inv, mpo_trivial_index_inv, mps_trivial_index});
  lenv({0, 0, 0}) = 1;
  file = GenEnvTenName("l", 0, temp_path);
  WriteGQTensorTOFile(lenv, file);
  mps.dealloc(0);

  assert(mps.empty());
}


/**
Function to perform a single two-site finite vMPS sweep.

@note Before the sweep and after the sweep, the MPS only contains mps[1].
*/
template <typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSSweep(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const SweepParams &sweep_params
) {
  auto N = mps.size();
  using TenT = GQTensor<TenElemT, QNT>;
  TenVec<TenT> lenvs(N - 1);
  TenVec<TenT> renvs(N - 1);
  double e0;

  for (size_t i = 0; i < N - 2; ++i) {
    // Load to-be-used tensors
    LoadRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params);
    e0 = TwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'r', i);
    // Dump related tensor to HD and remove unused tensor from RAM
    DumpRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params);
  }
  for (size_t i = N-1; i > 1; --i) {
    LoadRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
    e0 = TwoSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'l', i);
    DumpRelatedTensTwoSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
  }
  return e0;
}

/**  Single step for two site update.
 *  This function includes below procedure:
 *    ** update mps[lsite_idx]*mps[rsite_idx] tensors according corresponding environment tensors and these two mpo tensors,
 *       using lanczos algorithm;
 *    ** decompose and truncate mps[lsite_idx]*mps[rsite_idx] by svd decomposition. Canonical central are determined by the direction;
 *    ** generate the next environment in the direction.
 *  When using this function, one must make sure memory at least contains mps[lsite_idx] and mps[rsite_idx] tensors, and their environment tensors.
 */
template <typename TenElemT, typename QNT>
double TwoSiteFiniteVMPSUpdate(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const SweepParams &sweep_params,
    const char dir,
    const size_t target_site
) {
  Timer update_timer("two_site_fvmps_update");


  // Assign some parameters
  auto N = mps.size();
  std::vector<std::vector<size_t>> init_state_ctrct_axes;
  size_t svd_ldims;
  size_t lsite_idx, rsite_idx;
  size_t lenv_len, renv_len;
  std::string lblock_file, rblock_file;
  init_state_ctrct_axes = {{2}, {0}};
  svd_ldims = 2;
  switch (dir) {
    case 'r':
      lsite_idx = target_site;
      rsite_idx = target_site + 1;
      lenv_len = target_site;
      renv_len = N - (target_site + 2);
      break;
    case 'l':
      lsite_idx = target_site - 1;
      rsite_idx = target_site;
      lenv_len = target_site - 1;
      renv_len = N - target_site - 1;
      break;
    default:
      std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
      exit(1);
  }

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
  Timer lancz_timer("two_site_fvmps_lancz");
  auto lancz_res = LanczosSolver(
                       eff_ham, init_state,
                       &eff_ham_mul_two_site_state,
                       sweep_params.lancz_params
                   );
#ifdef GQMPS2_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif

  // SVD and measure entanglement entropy
#ifdef GQMPS2_TIMING_MODE
  Timer svd_timer("two_site_fvmps_svd");
#endif

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

#ifdef GQMPS2_TIMING_MODE
  svd_timer.PrintElapsed();
#endif

  // Update MPS local tensor
#ifdef GQMPS2_TIMING_MODE
  Timer update_mps_ten_timer("two_site_fvmps_update_mps_ten");
#endif

  TenT the_other_mps_ten;
  switch (dir) {
    case 'r':
      mps[lsite_idx] = std::move(u);
      Contract(&s, &vt, {{1}, {0}}, &the_other_mps_ten);
      mps[rsite_idx] = std::move(the_other_mps_ten);
      break;
    case 'l':
      Contract(&u, &s, {{2}, {0}}, &the_other_mps_ten);
      mps[lsite_idx] = std::move(the_other_mps_ten);
      mps[rsite_idx] = std::move(vt);
      break;
    default:
      assert(false);
  }

#ifdef GQMPS2_TIMING_MODE
  update_mps_ten_timer.PrintElapsed();
#endif

  // Update environment tensors
#ifdef GQMPS2_TIMING_MODE
  Timer update_env_ten_timer("two_site_fvmps_update_env_ten");
#endif

  switch (dir) {
    case 'r':{
      TenT temp1, temp2, lenv_ten;
      Contract(&lenvs[lenv_len], &mps[target_site], {{0}, {0}}, &temp1);
      Contract(&temp1, &mpo[target_site], {{0, 2}, {0, 1}}, &temp2);
      auto mps_ten_dag = Dag(mps[target_site]);
      Contract(&temp2, &mps_ten_dag, {{0 ,2}, {0, 1}}, &lenv_ten);
      lenvs[lenv_len + 1] = std::move(lenv_ten);
    }break;
    case 'l':{
      TenT temp1, temp2, renv_ten;
      Contract(&mps[target_site], eff_ham[3], {{2}, {0}}, &temp1);
      Contract(&temp1, &mpo[target_site], {{1, 2}, {1, 3}}, &temp2);
      auto mps_ten_dag = Dag(mps[target_site]);
      Contract(&temp2, &mps_ten_dag, {{3, 1}, {1, 2}}, &renv_ten);
      renvs[renv_len + 1] = std::move(renv_ten);
    }break;
    default:
      assert(false);
  }

#ifdef GQMPS2_TIMING_MODE
  update_env_ten_timer.PrintElapsed();
#endif

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
void LoadRelatedTensTwoSiteAlg(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const SweepParams &sweep_params
) {
#ifdef GQMPS2_TIMING_MODE
  Timer preprocessing_timer("two_site_fvmps_preprocessing");
#endif
  auto N = mps.size();
  switch (dir) {
    case 'r':
      if (target_site == 0) {
        mps.LoadTen(
            target_site,
            GenMPSTenName(sweep_params.mps_path, target_site)
        );

        auto renv_len = N - (target_site + 2);
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);
        RemoveFile(renv_file);

        auto lenv_len = 0;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        lenvs.LoadTen(lenv_len, lenv_file);
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
        mps.LoadTen(
            target_site,
            GenMPSTenName(sweep_params.mps_path, target_site)
        );
        auto renv_len = 0;
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);

        auto lenv_len = N-2;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        RemoveFile(lenv_file);
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
#ifdef GQMPS2_TIMING_MODE
  preprocessing_timer.PrintElapsed();
#endif
}


template <typename TenElemT, typename QNT>
void DumpRelatedTensTwoSiteAlg(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const SweepParams &sweep_params
) {
#ifdef GQMPS2_TIMING_MODE
  Timer postprocessing_timer("two_site_fvmps_postprocessing");
#endif
  auto N = mps.size();
  switch (dir) {
    case 'r':{
      lenvs.dealloc(target_site);
      renvs.dealloc(N - (target_site + 2));
      mps.DumpTen(
          target_site,
          GenMPSTenName(sweep_params.mps_path, target_site),
          true
      );
      lenvs.DumpTen(
          target_site + 1,
          GenEnvTenName("l", target_site + 1, sweep_params.temp_path)
      );
    }break;
    case 'l':{
      lenvs.dealloc((target_site+1) - 2);
      renvs.dealloc(N - (target_site+1));
      mps.DumpTen(
          target_site,
          GenMPSTenName(sweep_params.mps_path, target_site),
          true
      );
      auto next_renv_len = N - target_site;
      renvs.DumpTen(
          next_renv_len,
          GenEnvTenName("r", next_renv_len, sweep_params.temp_path)
      );
    }break;
    default:
      assert(false);
  }
#ifdef GQMPS2_TIMING_MODE
  postprocessing_timer.PrintElapsed();
#endif
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_ALGORITHM_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_IMPL_H */
