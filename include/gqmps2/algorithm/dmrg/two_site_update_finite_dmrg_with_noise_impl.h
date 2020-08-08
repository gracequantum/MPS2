// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2020-07-22
*
* Description: GraceQ/MPS2 project. Implementation details for two-site algorithm, with support for noise.
* Reference: Arxiv: 1501.05504v2
*/
#include "gqmps2/algorithm/dmrg/two_site_update_finite_dmrg.h"
#include "gqmps2/consts.h"
#include "gqmps2/one_dim_tn/mpo.h"    // MPO
#include "gqmps2/utilities.h"         // ReadGQTensorFromFile, WriteGQTensorTOFile, IsPathExist, CreatPath
#include "gqten/gqten.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include <assert.h>

#ifdef Release
#define NDEBUG
#endif


namespace gqmps2 {
using namespace gqten;
// Forward declarations
template <typename TenType>
void InplaceExpand(TenType* &A,TenType* B,int i);
template <typename TenType>
double TwoSiteUpdate(
  const long i,
  std::vector<TenType *> &mps,
  const std::vector<TenType *> &mpo,
  std::vector<TenType*> &lblocks,
  std::vector<TenType *> &rblocks,
  const SweepParams &sweep_params, const char dir, const double noise);

template <typename TenElemType>
void FuseIndex(GQTensor<TenElemType>* &T, int i, int j);

inline double MeasureEE(const DGQTensor *s, const long sdim);
inline std::string GenBlockFileName(const std::string &dir, const long blk_len);
inline void RemoveFile(const std::string &file);

// Two-site algorithm
template <typename TenType>
double TwoSiteAlgorithm(
  std::vector<TenType *> &mps, const MPO<TenType> &mpo,
  const SweepParams &sweep_params, std::vector<double> noise) {
  if ( sweep_params.FileIO && !IsPathExist(kRuntimeTempPath)) {
    CreatPath(kRuntimeTempPath);
  }

  while(noise.size()<sweep_params.Sweeps){
    noise.push_back(0.0);
  }
  auto l_and_r_blocks = InitBlocks(mps, mpo, sweep_params);

  std::cout << "\n";
  double e0;
  Timer sweep_timer("sweep");
  for (long sweep = 0; sweep < sweep_params.Sweeps; ++sweep) {
    std::cout << "sweep " << sweep << std::endl;
    sweep_timer.Restart();
    e0 = TwoSiteSweep(
      mps, mpo,
      l_and_r_blocks.first, l_and_r_blocks.second,
      sweep_params, noise[sweep]);
    sweep_timer.PrintElapsed();
    std::cout << "\n";
  }
  return e0;
}


template <typename TenType>
double TwoSiteSweep(
  std::vector<TenType *> &mps, const MPO<TenType> &mpo,
  std::vector<TenType *> &lblocks, std::vector<TenType *> &rblocks,
  const SweepParams &sweep_params,const double noise_sweep) {
  auto N = mps.size();
  double e0;
  for (size_t i = 0; i < N-1; ++i) {
    e0 = TwoSiteUpdate(i, mps, mpo, lblocks, rblocks, sweep_params, 'r',noise_sweep);
  }
  for (size_t i = N-1; i > 0; --i) {
    e0 = TwoSiteUpdate(i, mps, mpo, lblocks, rblocks, sweep_params, 'l',noise_sweep);
  }
  return e0;
}


template <typename TenType>
double TwoSiteUpdate(
  const long i,
  std::vector<TenType *> &mps,
  const MPO<TenType> &mpo,
  std::vector<TenType*> &lblocks,
  std::vector<TenType *> &rblocks,
  const SweepParams &sweep_params, const char dir, const double noise) {
  Timer update_timer("update");
  update_timer.Restart();

#ifdef GQMPS2_TIMING_MODE
  Timer bef_lanc_timer("bef_lanc");
  bef_lanc_timer.Restart();
#endif
  auto N = mps.size();
  std::vector<std::vector<long>> init_state_ctrct_axes, us_ctrct_axes;
  std::string where;
  long svd_ldims, svd_rdims;
  long lsite_idx, rsite_idx;
  long lblock_len, rblock_len;
  std::string lblock_file, rblock_file;

  switch (dir) {
    case 'r':
      lsite_idx = i;
      rsite_idx = i + 1;
      lblock_len = i;
      rblock_len = N - (i + 2);
      if (i == 0) {
        init_state_ctrct_axes = {{1}, {0}};
        where = "lend";
        svd_ldims = 1;
        svd_rdims = 2;
      } else if (i == N-2) {
        init_state_ctrct_axes = {{2}, {0}};
        where = "rend";
        svd_ldims = 2;
        svd_rdims = 1;
      } else {
        init_state_ctrct_axes = {{2}, {0}};
        where = "cent";
        svd_ldims = 2;
        svd_rdims = 2;
      }
      break;
    case 'l':
      lsite_idx = i-1;
      rsite_idx = i;
      lblock_len = i-1;
      rblock_len = N-i-1;
      if (i == N-1) {
        init_state_ctrct_axes = {{2}, {0}};
        where = "rend";
        svd_ldims = 2;
        svd_rdims = 1;
        us_ctrct_axes = {{2}, {0}};
      } else if (i == 1) {
        init_state_ctrct_axes = {{1}, {0}};
        where = "lend";
        svd_ldims = 1;
        svd_rdims = 2;
        us_ctrct_axes = {{1}, {0}};
      } else {
        init_state_ctrct_axes = {{2}, {0}};
        where = "cent";
        svd_ldims = 2;
        svd_rdims = 2;
        us_ctrct_axes = {{2}, {0}};
      }
      break;
    default:
      std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
      exit(1);
  }

  if (sweep_params.FileIO) {
    switch (dir) {
      case 'r':
        rblock_file = GenBlockFileName("r", rblock_len);
        ReadGQTensorFromFile(rblocks[rblock_len], rblock_file);
        if (rblock_len != 0) {
          RemoveFile(rblock_file);
        }
        break;
      case 'l':
        lblock_file = GenBlockFileName("l", lblock_len);
        ReadGQTensorFromFile(lblocks[lblock_len], lblock_file);
        if (lblock_len != 0) {
          RemoveFile(lblock_file);
        }
        break;
      default:
        std::cout << "dir must be 'r' or 'l', but " << dir << std::endl;
        exit(1);
    }
  }

#ifdef GQMPS2_TIMING_MODE
  bef_lanc_timer.PrintElapsed();
#endif

// Lanczos
  std::vector<TenType *>eff_ham(4);
  eff_ham[0] = lblocks[lblock_len];
  // Safe const casts for MPO local tensors.
  eff_ham[1] = const_cast<TenType *>(&mpo[lsite_idx]);
  eff_ham[2] = const_cast<TenType *>(&mpo[rsite_idx]);
  eff_ham[3] = rblocks[rblock_len];
  auto init_state = Contract(
    *mps[lsite_idx], *mps[rsite_idx],
    init_state_ctrct_axes);

  Timer lancz_timer("Lancz");
  lancz_timer.Restart();

  auto lancz_res = LanczosSolver(
    eff_ham, init_state,
    sweep_params.LanczParams,
    where);

#ifdef GQMPS2_TIMING_MODE
  auto lancz_elapsed_time = lancz_timer.PrintElapsed();
#else
  auto lancz_elapsed_time = lancz_timer.Elapsed();
#endif

// SVD
#ifdef GQMPS2_TIMING_MODE
  Timer svd_timer("svd");
  svd_timer.Restart();
#endif
  TenType* svd_pu = new TenType();
  TenType* svd_pv = new TenType();
  GQTensor<GQTEN_Double> * svd_ps = new GQTensor<GQTEN_Double>();
  double svd_truncation_error;
  long svd_D; //final bond dimension

  if(std::fabs(noise)==0||( dir=='r' && i==N-2  )||( dir=='l' && i == 1  )){
    // We do nothing
  }else if(i==0 && dir=='r' ){
    auto P=Contract(*lancz_res.gs_vec, *eff_ham[1], {{0}, {0}});
    InplaceContract(P, noise*(*eff_ham[2]), {{0, 2}, {1, 0}});
    P->Transpose({1,2,3,0});
    FuseIndex(P,2,3);
    InplaceExpand( lancz_res.gs_vec, P, 2);
    Index ExtendedSubSpace = InverseIndex(P->indexes[2]);
    delete P;
    //assume N>3
    auto ExtendedZeroTensor = TenType({ExtendedSubSpace, mps[i+2]->indexes[1],mps[i+2]->indexes[2]});
    InplaceExpand(mps[i+2], &ExtendedZeroTensor,0 );
  }else if(i==N-1 && dir =='l'){
    auto P = Contract(*lancz_res.gs_vec, *eff_ham[2],{{2},{0} });
    InplaceContract(P, noise*(*eff_ham[1]),{{1,2 },{1,3} });
    P->Transpose({0,2,3,1});
    FuseIndex(P,0,1);

    InplaceExpand( lancz_res.gs_vec, P, 0);
    auto ExtendedSubSpace = InverseIndex(P->indexes[0]);
    delete P;
    //assume N>3
    auto ExtendedZeroTensor =TenType({mps[i-2]->indexes[0],mps[i-2]->indexes[1],ExtendedSubSpace});
    InplaceExpand(mps[i-2], &ExtendedZeroTensor,2 );

  }else if( dir =='r') {
    // P = eff_ham[0]*mpo[i]*mps[i]*mpo[i+1]*mps[i+1]
    auto P = Contract(*eff_ham[0], *lancz_res.gs_vec, {{0},{0}});
    InplaceContract(P, *eff_ham[1], {{0, 2},{0, 1}});
    InplaceContract(P, noise*(*eff_ham[2]), {{4, 1},{0, 1}});
    P->Transpose({0, 2, 3, 4, 1});//-> index combine 0,1,2,{3,4} -> P
    FuseIndex(P,3,4);
    InplaceExpand( lancz_res.gs_vec, P, 3);
    auto ExtendedSubSpace =InverseIndex(P->indexes[3]) ;
    delete P;
    TenType ExtendedZeroTensor;
    if(i!=N-3){
      ExtendedZeroTensor = TenType({ExtendedSubSpace, mps[i+2]->indexes[1],mps[i+2]->indexes[2]});
    }else{
      ExtendedZeroTensor =TenType({ExtendedSubSpace, mps[i+2]->indexes[1]});
    }
    InplaceExpand(mps[i+2], &ExtendedZeroTensor,0 );
  }else{ //dir =='l'
    // P = eff_ham[4]*mpo[i]*mps[i]*mpo[i-1]*mps[i-1]
    auto P = Contract(*lancz_res.gs_vec,*eff_ham[3],{{3},{0}});
    InplaceContract(P,*eff_ham[2],{{2,3},{1,3}});
    InplaceContract(P,noise*(*eff_ham[1]),{{1,3},{1,3}});
    P->Transpose({0,3,4,2,1});
    FuseIndex(P, 0,1);
    InplaceExpand( lancz_res.gs_vec, P, 0);
    auto ExtendedSubSpace = InverseIndex(P->indexes[0]);
    delete P;
    if(i!=2){
      auto ExtendedZeroTensor =TenType({mps[i-2]->indexes[0],mps[i-2]->indexes[1],ExtendedSubSpace});
      InplaceExpand(mps[i-2], &ExtendedZeroTensor,2 );
    }else{
      auto ExtendedZeroTensor =TenType({mps[i-2]->indexes[0],ExtendedSubSpace});
      InplaceExpand(mps[i-2], &ExtendedZeroTensor,1 );
    }
  }

  Svd(lancz_res.gs_vec,
      svd_ldims, svd_rdims,
      Div(*mps[lsite_idx]), Div(*mps[rsite_idx]),
      sweep_params.Cutoff,
      sweep_params.Dmin, sweep_params.Dmax,
      svd_pu, svd_ps, svd_pv,
      &svd_truncation_error, &svd_D);


#ifdef GQMPS2_TIMING_MODE
  svd_timer.PrintElapsed();
#endif

  delete lancz_res.gs_vec;

// Measure entanglement entropy.
  auto ee = MeasureEE(svd_ps, svd_D);
// Update MPS sites and blocks.
#ifdef GQMPS2_TIMING_MODE
  Timer blk_update_timer("blk_update");
  blk_update_timer.Restart();
  Timer new_blk_timer("gen_new_blk");
  Timer dump_blk_timer("dump_blk");
#endif

  TenType *new_lblock, *new_rblock;
  bool update_block = true;
  switch (dir) {
    case 'r':

#ifdef GQMPS2_TIMING_MODE
      new_blk_timer.Restart();
#endif

      delete mps[lsite_idx];
      mps[lsite_idx] = svd_pu;
      delete mps[rsite_idx];
      mps[rsite_idx] = Contract(*svd_ps, *svd_pv, {{1}, {0}});
      delete svd_ps;
      delete svd_pv;

      if (i == 0) {
        new_lblock = Contract(*mps[i], mpo[i], {{0}, {0}});
        auto temp_new_lblock = Contract(
          *new_lblock, Dag(*mps[i]),
          {{2}, {0}});
        delete new_lblock;
        new_lblock = temp_new_lblock;
      } else if (i != N-2) {
        new_lblock = Contract(*lblocks[i], *mps[i], {{0}, {0}});
        auto temp_new_lblock = Contract(*new_lblock, mpo[i], {{0, 2}, {0, 1}});
        delete new_lblock;
        new_lblock = temp_new_lblock;
        temp_new_lblock = Contract(*new_lblock, Dag(*mps[i]), {{0, 2}, {0, 1}});
        delete new_lblock;
        new_lblock = temp_new_lblock;
      } else {
        update_block = false;
      }

#ifdef GQMPS2_TIMING_MODE
      new_blk_timer.PrintElapsed();
      dump_blk_timer.Restart();
#endif

      if (sweep_params.FileIO) {
        if (update_block) {
          auto target_blk_len = i+1;
          lblocks[target_blk_len] = new_lblock;
          auto target_blk_file = GenBlockFileName("l", target_blk_len);
          WriteGQTensorTOFile(*new_lblock, target_blk_file);
          delete eff_ham[0];
          delete eff_ham[3];
        } else {
          delete eff_ham[0];
        }
      } else {
        if (update_block) {
          auto target_blk_len = i+1;
          delete lblocks[target_blk_len];
          lblocks[target_blk_len] = new_lblock;
        }
      }

#ifdef GQMPS2_TIMING_MODE
      dump_blk_timer.PrintElapsed();
#endif

      break;
    case 'l':

#ifdef GQMPS2_TIMING_MODE
      new_blk_timer.Restart();
#endif

      delete mps[lsite_idx];
      mps[lsite_idx] = Contract(*svd_pu, *svd_ps, us_ctrct_axes);
      delete svd_pu;
      delete svd_ps;
      delete mps[rsite_idx];
      mps[rsite_idx] = svd_pv;

      if (i == N-1) {
        new_rblock = Contract(*mps[i], mpo[i], {{1}, {0}});
        auto temp_new_rblock = Contract(*new_rblock, Dag(*mps[i]), {{2}, {1}});
        delete new_rblock;
        new_rblock = temp_new_rblock;
      } else if (i != 1) {
        new_rblock = Contract(*mps[i], *eff_ham[3], {{2}, {0}});
        auto temp_new_rblock = Contract(*new_rblock, mpo[i], {{1, 2}, {1, 3}});
        delete new_rblock;
        new_rblock = temp_new_rblock;
        temp_new_rblock = Contract(*new_rblock, Dag(*mps[i]), {{3, 1}, {1, 2}});
        delete new_rblock;
        new_rblock = temp_new_rblock;
      } else {
        update_block = false;
      }

#ifdef GQMPS2_TIMING_MODE
      new_blk_timer.PrintElapsed();
      dump_blk_timer.Restart();
#endif

      if (sweep_params.FileIO) {
        if (update_block) {
          auto target_blk_len = N-i;
          rblocks[target_blk_len] = new_rblock;
          auto target_blk_file = GenBlockFileName("r", target_blk_len);
          WriteGQTensorTOFile(*new_rblock, target_blk_file);
          delete eff_ham[0];
          delete eff_ham[3];
        } else {
          delete eff_ham[3];
        }
      } else {
        if (update_block) {
          auto target_blk_len = N-i;
          delete rblocks[target_blk_len];
          rblocks[target_blk_len] = new_rblock;
        }
      }

#ifdef GQMPS2_TIMING_MODE
      dump_blk_timer.PrintElapsed();
#endif

  }

#ifdef GQMPS2_TIMING_MODE
  blk_update_timer.PrintElapsed();
#endif

  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << "Site " << std::setw(4) << i
            << " E0 = " << std::setw(20) << std::setprecision(kLanczEnergyOutputPrecision) << std::fixed << lancz_res.gs_eng
            << " TruncErr = " << std::setprecision(2) << std::scientific << svd_truncation_error<< std::fixed
            << " D = " << std::setw(5) << svd_D
            << " Iter = " << std::setw(3) << lancz_res.iters
            << " LanczT = " << std::setw(8) << lancz_elapsed_time
            << " TotT = " << std::setw(8) << update_elapsed_time
            << " S = " << std::setw(10) << std::setprecision(7) << ee;
  std::cout << std::scientific << std::endl;
  return lancz_res.gs_eng;
}

/** FuseIndex
 *  Fuse 2 indices i and j of tensor T, inplacement operation.
 * @tparam TenElemType
 * @param T
 * @param i
 * @param j
 */
template <typename TenElemType>
void FuseIndex(GQTensor<TenElemType>* &T, int i, int j){
  assert(T->indexes[i].dir==T->indexes[j].dir);
  GQTensor<TenElemType> C = IndexCombine<TenElemType>(
    InverseIndex(T->indexes[i]),InverseIndex(T->indexes[j]), T->indexes[i].dir);
  InplaceContract(T,C, {{i,j}, {0,1}});
  int legnumber = T->indexes.size();
  std::vector<long> axs(legnumber);
  for(int k=0;k<legnumber;k++){
    if(k<i) axs[k]=k;
    else if(k==i) axs[k]=legnumber-1;
    else axs[k]=k-1;
  }
  T->Transpose(axs);
}
/** InplaceExpand
 * Inplacement version of Expand function,only expand on one index
 * @tparam TenType
 * @param A
 * @param B
 * @param i
 */
template <typename TenType>
void InplaceExpand(TenType* &A,TenType* B,int i){
TenType* C = new TenType();
std::vector<size_t> axs= {(unsigned long)i};
Expand(A,B, axs, C);
delete A;
A = C;
}

} /* gqmps2 */
