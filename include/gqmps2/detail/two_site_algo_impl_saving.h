// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-29 21:20
*
* Description: GraceQ/MPS2 project. Implementation details for two-site algorithm, with saving mps to hard disk.
*/
#include "gqmps2/gqmps2.h"
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

// Helpers
inline double MeasureEE(const DGQTensor *s, const long sdim);

inline std::string GenBlockFileName(const std::string &dir, const long blk_len);

inline void RemoveFile(const std::string &file);
template <typename TenType>
void ReadMps(const int i, std::vector<TenType *> & mps){
  assert(mps[i]==NULL);
  mps[i] = new TenType();
  std::string file;
  file = kMpsPath + "/" + kMpsTenBaseName + std::to_string(i) + "." + kGQTenFileSuffix;
  std::ifstream ifs(file, std::ifstream::binary);
  bfread(ifs, *mps[i]);
  ifs.close();
  return;
}
template <typename TenType>
void SaveMps(const int i, std::vector<TenType *> & mps){
  std::string file;
  file = kMpsPath + "/" + kMpsTenBaseName + std::to_string(i) + "." + kGQTenFileSuffix;
  std::ofstream ofs(file, std::ifstream::binary);
  bfwrite(ofs, *mps[i]);
  ofs.close();
  delete mps[i];
  mps[i] = NULL;
  return;
}

// Two-site algorithm
template <typename TenType>
double TwoSiteAlgorithm(
  const std::vector<TenType *> &mpo,
  const SweepParams &sweep_params) {
  if ( sweep_params.FileIO && !IsPathExist(kRuntimeTempPath)) {
    CreatPath(kRuntimeTempPath);
  }
  if ( !IsPathExist(kMpsPath)){
    std::cout << "Cannot find the mps folder" << std::endl;
    abort();
  }

  auto l_and_r_blocks = InitBlocks(mpo, sweep_params);

  std::cout << "\n";
  double e0;
  Timer sweep_timer("sweep");
  int N = mpo.size();
  for (long sweep = 0; sweep < sweep_params.Sweeps; ++sweep) {
    std::cout << "sweep " << sweep << std::endl;
    sweep_timer.Restart();
    e0 = TwoSiteSweep(mpo,
      l_and_r_blocks.first, l_and_r_blocks.second,
      sweep_params);
    sweep_timer.PrintElapsed();
    std::cout << "\n";
  }
  return e0;
}


template<typename TenType>
std::pair<std::vector<TenType *>, std::vector<TenType *>> InitBlocks(
  const std::vector<TenType *> &mpo,
  const SweepParams &sweep_params) {
  auto N = mpo.size();
  int NumberOfMpsFile = 0;
  for(NumberOfMpsFile=0; NumberOfMpsFile < N+2; NumberOfMpsFile++ ){
    std::string file;
    file = kMpsPath + "/" + kMpsTenBaseName + std::to_string(NumberOfMpsFile) + "." + kGQTenFileSuffix;
    std::ifstream ifs(file, std::ifstream::binary);
    if(ifs.good()){
      ifs.close();
    }else{
      break;
    }
  }
  std::cout<< "The number of files in path mps =" <<  NumberOfMpsFile << endl;
  assert(NumberOfMpsFile == mpo.size());
  std::vector<TenType *> rblocks(N-1);
  std::vector<TenType *> lblocks(N-1);
  TenType*  mps_tensor;

  if (sweep_params.Workflow == kTwoSiteAlgoWorkflowContinue) {
    return std::make_pair(lblocks, rblocks);
  }

  // Generate blocks.
  // Right blocks.
  auto rblock0 = new TenType();
  rblocks[0] = rblock0;

  mps_tensor = new TenType();
  std::string file;
  file = kMpsPath + "/" + kMpsTenBaseName + std::to_string((int)(N)-1) + "." + kGQTenFileSuffix;
  std::ifstream ifs(file, std::ifstream::binary);
  bfread(ifs, *mps_tensor);
  ifs.close();

  auto rblock1 = Contract(*mps_tensor, *mpo.back(), {{1}, {0}});
  auto temp_rblock1 = Contract(*rblock1, Dag(*mps_tensor), {{2}, {1}});
  delete rblock1;
  delete mps_tensor;
  rblock1 = temp_rblock1;
  rblocks[1] = rblock1;

  if (sweep_params.FileIO) {
    file = GenBlockFileName("r", 0);
    WriteGQTensorTOFile(*rblock0, file);
    delete rblocks[0];
    file = GenBlockFileName("r", 1);
    WriteGQTensorTOFile(*rblock1, file);
  }
  for (size_t i = 2; i < N-1; ++i) {
    mps_tensor = new TenType();
    file = kMpsPath + "/" + kMpsTenBaseName + std::to_string((int)(N-i)) + "." + kGQTenFileSuffix;
    ifs.open(file,std::ifstream::binary);
    bfread(ifs, *mps_tensor);
    ifs.close();

    auto rblocki = Contract(*mps_tensor, *rblocks[i-1], {{2}, {0}});
    auto temp_rblocki = Contract(*rblocki, *mpo[N-i], {{1, 2}, {1, 3}});
    delete rblocki;
    rblocki = temp_rblocki;
    temp_rblocki = Contract(*rblocki, Dag(*mps_tensor), {{3, 1}, {1, 2}});
    delete rblocki;
    delete mps_tensor;
    rblocki = temp_rblocki;
    rblocks[i] = rblocki;
    if (sweep_params.FileIO) {
      auto file = GenBlockFileName("r", i);
      WriteGQTensorTOFile(*rblocki, file);
      delete rblocks[i-1];
    }
  }
  if (sweep_params.FileIO) { delete rblocks[N-2]; }

  // Left blocks.
  if (sweep_params.FileIO) {
    auto file = GenBlockFileName("l", 0);
    WriteGQTensorTOFile(TenType(), file);
  }

  return std::make_pair(lblocks, rblocks);
}


template <typename TenType>
double TwoSiteSweep( //The single sweep of DMRG
  const std::vector<TenType *> &mpo,
  std::vector<TenType *> &lblocks, std::vector<TenType *> &rblocks,
  const SweepParams &sweep_params) {
  auto N = mpo.size();
  double e0;
  std::vector<TenType *> mps(N);
  for (size_t i = 0; i < N-1; ++i) {
    e0 = TwoSiteUpdate_saving(i, mps, mpo, lblocks, rblocks, sweep_params, 'r');
  }
  for (size_t i = N-1; i > 0; --i) {
    e0 = TwoSiteUpdate_saving(i, mps, mpo, lblocks, rblocks, sweep_params, 'l');
  }
  return e0;
}



template <typename TenType>
double TwoSiteUpdate_saving( // Single step of two site update,
  const long i, //'r' mode, from i=0 to N-2, update (i,i+1); 'l' mode, from i-1->2, update (i-1,i)
  std::vector<TenType *> &mps, // point to nothing actually
  const std::vector<TenType *> &mpo,
  std::vector<TenType *> &lblocks, std::vector<TenType *> &rblocks, //point to nothing
  const SweepParams &sweep_params, const char dir) { //dir is the mode 'r' or 'l'
  Timer update_timer("update");
  update_timer.Restart();

#ifdef GQMPS2_TIMING_MODE
  Timer bef_lanc_timer("bef_lanc");
  bef_lanc_timer.Restart();
#endif

  auto N = mpo.size();
  std::vector<std::vector<long>> init_state_ctrct_axes, us_ctrct_axes;
  std::string where;
  long svd_ldims, svd_rdims;
  long lsite_idx, rsite_idx;
  long lblock_len, rblock_len;
  std::string lblock_file, rblock_file;

  switch (dir) {
    case 'r':
      lsite_idx = i;
      rsite_idx = i+1;
      lblock_len = i;
      rblock_len = N-(i+2);
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
  if(i==0 && dir =='r'){
    ReadMps(lsite_idx, mps);
    ReadMps(rsite_idx, mps);
  }else if(dir=='r'){
    ReadMps(rsite_idx, mps);
  }else if(dir=='l'){
    ReadMps(lsite_idx, mps);
  }


  // Lanczos
  std::vector<TenType *>eff_ham(4);
  eff_ham[0] = lblocks[lblock_len];
  eff_ham[1] = mpo[lsite_idx];
  eff_ham[2] = mpo[rsite_idx];
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

  auto svd_res = Svd(
    *lancz_res.gs_vec,
    svd_ldims, svd_rdims,
    Div(*mps[lsite_idx]), Div(*mps[rsite_idx]),
    sweep_params.Cutoff,
    sweep_params.Dmin, sweep_params.Dmax);

#ifdef GQMPS2_TIMING_MODE
  svd_timer.PrintElapsed();
#endif

  delete lancz_res.gs_vec;

  // Measure entanglement entropy.
  auto ee = MeasureEE(svd_res.s, svd_res.D);

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
      mps[lsite_idx] = svd_res.u;
      delete mps[rsite_idx];
      mps[rsite_idx] = Contract(*svd_res.s, *svd_res.v, {{1}, {0}});
      delete svd_res.s;
      delete svd_res.v;

      if (i == 0) {
        new_lblock = Contract(*mps[i], *mpo[i], {{0}, {0}});
        auto temp_new_lblock = Contract(
          *new_lblock, Dag(*mps[i]),
          {{2}, {0}});
        delete new_lblock;
        new_lblock = temp_new_lblock;
      } else if (i != N-2) {
        new_lblock = Contract(*lblocks[i], *mps[i], {{0}, {0}});
        auto temp_new_lblock = Contract(*new_lblock, *mpo[i], {{0, 2}, {0, 1}});
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
      mps[lsite_idx] = Contract(*svd_res.u, *svd_res.s, us_ctrct_axes);
      delete svd_res.u;
      delete svd_res.s;
      delete mps[rsite_idx];
      mps[rsite_idx] = svd_res.v;

      if (i == N-1) {
        new_rblock = Contract(*mps[i], *mpo[i], {{1}, {0}});
        auto temp_new_rblock = Contract(*new_rblock, Dag(*mps[i]), {{2}, {1}});
        delete new_rblock;
        new_rblock = temp_new_rblock;
      } else if (i != 1) {
        new_rblock = Contract(*mps[i], *eff_ham[3], {{2}, {0}});
        auto temp_new_rblock = Contract(*new_rblock, *mpo[i], {{1, 2}, {1, 3}});
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

  if( dir =='r'){
    SaveMps(lsite_idx, mps);
  }else if(dir=='l' && i>1){
    SaveMps(rsite_idx, mps);
  }else if(dir=='l'){
    SaveMps(0, mps);
    SaveMps(1, mps);
  }


  auto update_elapsed_time = update_timer.Elapsed();
  std::cout << "Site " << std::setw(4) << i
            << " E0 = " << std::setw(20) << std::setprecision(kLanczEnergyOutputPrecision) << std::fixed << lancz_res.gs_eng
            << " TruncErr = " << std::setprecision(2) << std::scientific << svd_res.trunc_err << std::fixed
            << " D = " << std::setw(5) << svd_res.D
            << " Iter = " << std::setw(3) << lancz_res.iters
            << " LanczT = " << std::setw(8) << lancz_elapsed_time
            << " TotT = " << std::setw(8) << update_elapsed_time
            << " S = " << std::setw(10) << std::setprecision(7) << ee;
  std::cout << std::scientific << std::endl;
  return lancz_res.gs_eng;
}
} /* gqmps2 */
