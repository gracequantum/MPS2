//
// Created by Hao-Xin on 2021/6/12.
//

#ifndef GRACEQ_MPS2_ONE_SITE_UPDATE_FINITE_VMPS_IMPL_H
#define GRACEQ_MPS2_ONE_SITE_UPDATE_FINITE_VMPS_IMPL_H

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
  using std::cout;
  using std::endl;
  using std::vector;
  // Helpers
  template <typename DTenT>
  inline double MeasureEE(const DTenT &s, const size_t sdim);

  /** Fuse two indices of a tensor t. we suppose i<j, and the new index is placed on position i.
   * Original index after index i a placed in original order.
   */
  template <typename TenElemType, typename QNT>
  GQTensor<TenElemType, QNT> FuseIndex(const GQTensor<TenElemType, QNT>& t, unsigned int i, unsigned int j){
    assert(i<j);
    assert(i<t.Rank() && j<t.Rank());
    assert(t.GetIndexes()[i].GetDir()==t.GetIndexes()[j].GetDir());
    Index<QNT> index1 = t.GetIndexes()[i];
    Index<QNT> index2 = t.GetIndexes()[j];

    GQTensor<TenElemType, QNT> combiner = IndexCombine<TenElemType>(
        InverseIndex(index1),InverseIndex(index2), index1.GetDir());
    GQTensor<TenElemType, QNT> fused_tensor;
    Contract(&t, &combiner, {{i,j}, {0,1}}, &fused_tensor);
    unsigned int rank = fused_tensor.Rank();
    std::vector<size_t> axs(rank);
    for(size_t k=0;k<rank;k++){
      if(k<i) axs[k]=k;
      else if(k==i) axs[k]=rank-1;
      else axs[k]=k-1;
    }
    fused_tensor.Transpose(axs);
    return fused_tensor;
  }

  /**
Function to perform single-site update finite vMPS algorithm.

@note The input MPS will be considered an empty one.
*/
template <typename TenElemT, typename QNT>
GQTEN_Double SingleSiteFiniteVMPS(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    SweepParams &sweep_params){
    assert(mps.size() == mpo.size());

    std::cout << "\n";
    std::cout << "=====> Sweep Parameter <=====" << "\n";
    std::cout << "MPS/MPO size: \t " << mpo.size() << "\n";
    std::cout << "The number of sweep times: \t " << sweep_params.sweeps << "\n";
    std::cout << "Bond dimension: \t " << sweep_params.Dmin << "/" << sweep_params.Dmax << "\n";
    std::cout << "Cut off truncation error: \t " <<sweep_params.trunc_err <<"\n";
    std::cout << "Lanczos max iterations \t" <<sweep_params.lancz_params.max_iterations << "\n";
    std::cout << "Preseted noises: \t[";
    for(size_t i = 0; i < sweep_params.noises.size() ; i++){
      std::cout << sweep_params.noises[i];
      if(i!=sweep_params.noises.size()-1){
        std::cout << ", " ;
      }else{
        std::cout << "]\n";
      }
    }
    std::cout << "MPS path: \t" << sweep_params.mps_path <<"\n";
    std::cout << "Temp path: \t" << sweep_params.temp_path << std::endl;

    // If the runtime temporary directory does not exit, create it and initialize
    // the left/right environments
    if (!IsPathExist(sweep_params.temp_path)) {
      CreatPath(sweep_params.temp_path);
      InitEnvs(mps, mpo, sweep_params, 1);
      std::cout << "no exsiting path " <<sweep_params.temp_path
        <<", thus progress created it and generated environment tensors."
        <<std::endl;
    }else{
      std::cout << "finded exsiting path "<<sweep_params.temp_path
        <<", thus progress will use the present environment tensors."
        <<std::endl;
    }

    GQTEN_Double e0;

    if(sweep_params.noises.size()==0) sweep_params.noises.push_back(0.0);
    double noise_start;
    mps.LoadTen(0,GenMPSTenName(sweep_params.mps_path, 0));
    for (size_t sweep = 1; sweep <= sweep_params.sweeps; ++sweep) {
      if(sweep-1 < sweep_params.noises.size()){
        noise_start = sweep_params.noises[sweep-1];
      }else{
        //do nothing, using the last sweep returned noise
      }
      std::cout << "sweep " << sweep << std::endl;
      Timer sweep_timer("sweep");
      e0 = SingleSiteFiniteVMPSSweep(mps, mpo, sweep_params, noise_start);
      sweep_timer.PrintElapsed();
      std::cout << "\n";
    }
    mps.DumpTen(0,GenMPSTenName(sweep_params.mps_path, 0),true);
    return e0;
}

template <typename TenElemT, typename QNT>
double SingleSiteFiniteVMPSSweep(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const SweepParams &sweep_params,
    double& noise_start) {
  auto N = mps.size();
  using TenT = GQTensor<TenElemT, QNT>;
  TenVec<TenT> lenvs(N), renvs(N);
  double e0(0.0), actual_e0(0.0), laststep_e0(1000.0), actual_laststep_e0(0.0);

  double& noise_running = noise_start;
  for (size_t i = 0; i < N - 1; ++i) {
    LoadRelatedTensSingleSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params);
// note: here we need mps[i](do not need load), mps[i+1], lenvs[i](do not need load), and mps[i]'s renvs
// mps[i]'s renvs can be removed

    actual_e0 = CalEnergyEptSingleSite(mps, mpo,lenvs, renvs, i);
    if(actual_e0-laststep_e0 <= 0.0){
      //expand and truncate let the energy lower or not change
      // this case is very rare, but include the boundary mps tensor case
      // so we do nothing now
    }else if(actual_e0-laststep_e0 >= fabs(actual_laststep_e0-laststep_e0)){
      //below two case suppose actual_laststep_e0-laststep_e0>0, usually it is right
      noise_running=noise_running*0.9;
    }else{
      noise_running = std::min(noise_running*1.05, 1.0);
    }
    e0 = SingleSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'r', i, noise_running);
    laststep_e0 = e0;
    actual_laststep_e0 = actual_e0;
    DumpRelatedTensSingleSiteAlg(mps, lenvs, renvs, i, 'r', sweep_params);
// note: here we need dump mps[i](free memory), lenvs[i+1](without free memory)
  }
  for (size_t i = N-1; i > 0; --i) {
    LoadRelatedTensSingleSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
    actual_e0 = CalEnergyEptSingleSite(mps, mpo,lenvs, renvs, i);
    if(actual_e0-laststep_e0 <= 0.0){
      //expand and truncate let the energy lower or not change
      // this case is very rare, but include the boundary mps tensor case
      // so we do nothing now
    }else if(actual_e0-laststep_e0 >= fabs(actual_laststep_e0-laststep_e0)){
      //below two case suppose actual_laststep_e0-laststep_e0>0, usually it is right
      noise_running=noise_running*0.9;
    }else{
      noise_running = std::min(noise_running*1.05, 1.0);
    }
    e0 = SingleSiteFiniteVMPSUpdate(mps, lenvs, renvs, mpo, sweep_params, 'l', i, noise_running);
    laststep_e0 = e0;
    actual_laststep_e0 = actual_e0;
    DumpRelatedTensSingleSiteAlg(mps, lenvs, renvs, i, 'l', sweep_params);
  }
  return e0;
}

/**  Single step for single site update.
 *  This function includes below procedure:
 *    ** update mps[target] tensors according corresponding environment tensors and the mpo tensor,
 *       using lanczos algorithm;
 *    ** expand mps[target] and mps[next_site] by noise, if need
 *    ** canonicalize mps to mps[next_site] by svd, while truncate tensor mps[target] if need
 *    ** generate the next environment in the direction.
 */
template <typename TenElemT, typename QNT>
double SingleSiteFiniteVMPSUpdate(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    const SweepParams &sweep_params,
    const char dir,
    const size_t target_site,
    double& noise
  ){
  Timer update_timer("single_site_fvmps_update");

#ifdef GQMPS2_TIMING_MODE
  Timer preprocessing_timer("single_site_fvmps_preprocessing");
#endif
  auto N = mps.size();
  unsigned lenv_len = target_site;
  unsigned renv_len = N - target_site - 1;
  unsigned svd_ldims;
  unsigned next_site;
  switch (dir) {
    case 'r':
      svd_ldims=2;
      next_site=target_site+1;
      break;
    case 'l':
      svd_ldims=1;
      next_site=target_site-1;
      break;
    default:
      std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
      exit(3);
  }

  using TenT = GQTensor<TenElemT, QNT>;
  std::vector<TenT *>eff_ham(3);
  eff_ham[0] = lenvs(lenv_len);
  // Safe const casts for MPO local tensors.
  eff_ham[1] = const_cast<TenT *>(&mpo[target_site]);
  eff_ham[2] = renvs(renv_len);

  std::vector<size_t> mps_ten_shape = mps[target_site].GetShape();
  Timer lancz_timer("single_site_fvmps_lancz");
  auto lancz_res = LanczosSolver(
      eff_ham, mps(target_site),
      &eff_ham_mul_single_site_state,
      sweep_params.lancz_params);//note here mps(target_site) are destroyed.
#ifdef GQMPS2_TIMING_MODE
auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif
  bool need_expand(true);

  if(fabs(noise)<1e-15){need_expand=false;}
  else if( target_site < N/2 && mps_ten_shape[0]*mps_ten_shape[1]<=mps_ten_shape[2]){need_expand= false;}
  else if( target_site > N/2 && mps_ten_shape[2]*mps_ten_shape[1]<=mps_ten_shape[0]){need_expand= false;}
  Timer expand_timer("single_site_fvmps_expand");
  if(need_expand){
    SingleSiteFiniteVMPSExpand( lancz_res.gs_vec, mps, eff_ham, dir, target_site, noise);
    delete lancz_res.gs_vec;
  }else{
    mps(target_site) = lancz_res.gs_vec;
  }

#ifdef GQMPS2_TIMING_MODE
    auto expand_elapsed_time = expand_timer.PrintElapsed();
#else
    auto expand_elapsed_time = expand_timer.Elapsed();
#endif
#ifdef GQMPS2_TIMING_MODE
    Timer svd_timer("single_site_fvmps_svd");
#endif

    TenT u, vt;
    GQTensor<GQTEN_Double, QNT> s;
    GQTEN_Double actual_trunc_err;
    size_t D;
    QNT zero_div = Div(mps[target_site])-Div(mps[target_site]);
    QNT div_left = (dir=='r'? Div(mps[target_site]): zero_div );
    SVD(
        mps(target_site),
        svd_ldims, div_left,
        sweep_params.trunc_err, sweep_params.Dmin, sweep_params.Dmax,
        &u, &s, &vt, &actual_trunc_err, &D
    );
    auto ee = MeasureEE(s, D);

#ifdef GQMPS2_TIMING_MODE
    svd_timer.PrintElapsed();
#endif


#ifdef GQMPS2_TIMING_MODE
    Timer update_mps_ten_timer("single_site_fvmps_update_mps_ten");
#endif
    TenT* temp_ten1 = new TenT();
    TenT* temp_ten2 = new TenT();
  switch(dir){
    case 'r':
      mps[target_site] = std::move(u);
      Contract(&s, &vt, {{1}, {0}}, temp_ten1);
      Contract(temp_ten1, mps(next_site), {{1},{0}},temp_ten2);
      delete temp_ten1; delete mps(next_site);
      mps(next_site)= temp_ten2;
      break;
    case 'l':
      mps[target_site] = std::move(vt);
      Contract(&u, &s, {{1}, {0}}, temp_ten1);
      Contract(mps(next_site), temp_ten1, {{2}, {0}}, temp_ten2);
      delete temp_ten1; delete mps(next_site);
      mps(next_site) =temp_ten2;
      break;
  }


#ifdef GQMPS2_TIMING_MODE
    update_mps_ten_timer.PrintElapsed();
#endif

    // Update environment tensors
#ifdef GQMPS2_TIMING_MODE
    Timer update_env_ten_timer("single_site_fvmps_update_env_ten");
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
        Contract(&mps[target_site], eff_ham[2], {{2}, {0}}, &temp1);
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
              << " noise = " <<  std::setprecision(2) << std::scientific  << noise << std::fixed
              << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
              << " D = " << std::setw(5) << D
              << " Iter = " << std::setw(3) << lancz_res.iters
              << " LanczT = " << std::setw(8) << lancz_elapsed_time
              << " ExpandT = " << std::setw(8) << expand_elapsed_time
              << " TotT = " << std::setw(8) << update_elapsed_time
              << " S = " << std::setw(10) << std::setprecision(7) << ee;
    if(!need_expand){
      std::cout << "(no noise)";
    }
    std::cout << std::scientific << std::endl;
    return lancz_res.gs_eng;
}

template <typename TenElemT, typename QNT>
void SingleSiteFiniteVMPSExpand(
    GQTensor<TenElemT, QNT>* gs_vec,
    FiniteMPS<TenElemT, QNT> &mps,
    std::vector< GQTensor<TenElemT, QNT> *> eff_ham,
    const char dir,
    const size_t target_site,
    double noise
    ){
  // we suppose mps only contain mps[next_site]
  using TenT = GQTensor<TenElemT, QNT>;
  TenT* ten_tmp = new TenT();
  mps(target_site) = new TenT();
  if(dir=='r'){
    size_t next_site = target_site+1;
    Contract(eff_ham[0], gs_vec, {{0}, {0}}, ten_tmp);
    InplaceContract(ten_tmp, eff_ham[1], {{0, 2}, {0, 1}});
    ten_tmp->Transpose({0, 2, 1, 3});
    TenT expanded_ten = FuseIndex( *ten_tmp, 2, 3);
    expanded_ten = noise*expanded_ten;;
    Expand(gs_vec, &expanded_ten, {2},  mps(target_site));

    auto expanded_index = InverseIndex(expanded_ten.GetIndexes()[2]);
    TenT expanded_zero_ten = TenT({expanded_index, mps[next_site].GetIndexes()[1],mps[next_site].GetIndexes()[2]});
    Expand(mps(next_site), &expanded_zero_ten, {0}, ten_tmp);
    mps(next_site) = ten_tmp;
  }else if(dir=='l'){
    size_t next_site = target_site-1;
    Contract(gs_vec,eff_ham[2],{{2},{0}}, ten_tmp);
    InplaceContract(ten_tmp, eff_ham[1],{{1,2},{1,3}});
    ten_tmp->Transpose({0,2,3,1});
    TenT expanded_ten = FuseIndex(*ten_tmp, 0, 1);
    expanded_ten = noise*expanded_ten;
    Expand(gs_vec, &expanded_ten, {0}, mps(target_site));

    auto expanded_index = InverseIndex(expanded_ten.GetIndexes()[0]);
    TenT expanded_zero_ten = TenT({mps[next_site].GetIndexes()[0],mps[next_site].GetIndexes()[1], expanded_index});
    Expand(mps(next_site), &expanded_zero_ten, {2}, ten_tmp);
    mps(next_site) = ten_tmp;
  }
//The expanded tensors are saved in mps
}


template <typename TenElemT, typename QNT>
void LoadRelatedTensSingleSiteAlg(
    FiniteMPS<TenElemT, QNT> &mps,
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
            target_site+1,
            GenMPSTenName(sweep_params.mps_path, target_site+1)
        );
        auto renv_len = N - target_site -1;
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
        auto renv_len = N - target_site - 1;
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);
        RemoveFile(renv_file);
      }
      break;
    case 'l':
      if (target_site == N-1) {
        mps.LoadTen(
            target_site -1,
            GenMPSTenName(sweep_params.mps_path, target_site-1)
        );
        auto renv_len = 0;
        auto renv_file = GenEnvTenName("r", renv_len, sweep_params.temp_path);
        renvs.LoadTen(renv_len, renv_file);

        auto lenv_len = N-1;
        auto lenv_file = GenEnvTenName("l", lenv_len, sweep_params.temp_path);
        RemoveFile(lenv_file);
      } else {
        mps.LoadTen(
            target_site - 1,
            GenMPSTenName(sweep_params.mps_path, target_site - 1)
        );
        auto lenv_len = target_site;
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
void DumpRelatedTensSingleSiteAlg(
    FiniteMPS<TenElemT, QNT> &mps,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site,
    const char dir,
    const SweepParams &sweep_params
) {
  auto N = mps.size();
  lenvs.dealloc(target_site);
  renvs.dealloc(N - target_site -1);
  mps.DumpTen(
      target_site,
      GenMPSTenName(sweep_params.mps_path, target_site),
      true);
  switch (dir) {
    case 'r':{
      lenvs.DumpTen(
          target_site + 1,
          GenEnvTenName("l", target_site + 1, sweep_params.temp_path));
    }break;
    case 'l':{
      auto next_renv_len = N - target_site;
      renvs.DumpTen(
          next_renv_len,
          GenEnvTenName("r", next_renv_len, sweep_params.temp_path));
    }break;
    default:
      assert(false);
  }
}

template <typename TenElemT, typename QNT>
double CalEnergyEptSingleSite(
    FiniteMPS<TenElemT, QNT> &mps,
    const MPO<GQTensor<TenElemT, QNT>> &mpo,
    TenVec<GQTensor<TenElemT, QNT>> &lenvs,
    TenVec<GQTensor<TenElemT, QNT>> &renvs,
    const size_t target_site
){
  using TenT = GQTensor<TenElemT, QNT>;
  std::vector<TenT *>eff_ham(3);
  unsigned lenv_len = target_site;
  unsigned renv_len = mps.size() - target_site - 1;
  eff_ham[0] = lenvs(lenv_len);
  // Safe const casts for MPO local tensors.
  eff_ham[1] = const_cast<TenT *>(&mpo[target_site]);
  eff_ham[2] = renvs(renv_len);
  TenT *h_mul_state = eff_ham_mul_single_site_state(eff_ham, mps(target_site));
  TenT scalar_ten;
  TenT mps_ten_dag = Dag(mps[target_site]);
  Contract(h_mul_state, &mps_ten_dag,{{0,1,2},{0,1,2}}, &scalar_ten);
  delete h_mul_state;
  double energy = Real(scalar_ten());
  return energy;
}

}

#endif //GRACEQ_MPS2_ONE_SITE_UPDATE_FINITE_VMPS_IMPL_H
