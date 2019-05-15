/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 12:22
* 
* Description: GraceQ/mps2 project. Private objects for two sites algorithm. Implementation.
*/
#include "two_site_algo.h"
#include "gqten/gqten.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>


namespace gqmps2 {
using namespace gqten;


std::vector<GQTensor *> InitRBlocks(
    const std::vector<GQTensor *> &mps, const std::vector<GQTensor *> &mpo) {
  assert(mps.size() == mpo.size());
  auto N = mps.size();
  std::vector<GQTensor *> rblocks(N-1);
  auto rblock1 = Contract(*mps.back(), *mpo.back(), {{1}, {0}});
  auto temp_rblock1 = Contract(*rblock1, MockDag(*mps.back()), {{2}, {1}});
  delete rblock1;
  rblock1 = temp_rblock1;
  rblocks[1] = rblock1;
  for (size_t i = 2; i < N-1; ++i) {
    auto rblocki = Contract(*mps[N-i], *rblocks[i-1], {{2}, {0}});
    auto temp_rblocki = Contract(*rblocki, *mpo[N-i], {{1, 2}, {1, 3}});
    delete rblocki;
    rblocki = temp_rblocki;
    temp_rblocki = Contract(*rblocki, MockDag(*mps[N-i]), {{3, 1}, {1, 2}});
    delete rblocki;
    rblocki = temp_rblocki;
    rblocks[i] = rblocki;
  }
  return rblocks;
}


double TwoSiteSweep(
    std::vector<GQTensor *> &mps, const std::vector<GQTensor *> &mpo,
    std::vector<GQTensor *> &lblocks, std::vector<GQTensor *> &rblocks,
    const SweepParams &sweep_params) {
  auto N = mps.size();
  double e0;
  for (size_t i = 0; i < N-1; ++i) {
    e0 = TwoSiteUpdate(i, mps, mpo, lblocks, rblocks, sweep_params, 'r');
  }
  for (size_t i = N-1; i > 0; --i) {
    e0 = TwoSiteUpdate(i, mps, mpo, lblocks, rblocks, sweep_params, 'l');
  }
  return e0;
}


double TwoSiteUpdate(
    const long i,
    std::vector<GQTensor *> &mps, const std::vector<GQTensor *> &mpo,
    std::vector<GQTensor *> &lblocks, std::vector<GQTensor *> &rblocks,
    const SweepParams &sweep_params, const char dir) {
  std::cout << "Site " << i << " updating ..." << std::endl;
  auto N = mps.size();
  std::vector<GQTensor *>eff_ham(4);
  std::vector<std::vector<long>> init_state_ctrct_axes, us_ctrct_axes;
  std::string where;
  long svd_ldims, svd_rdims;
  long lsite_idx, rsite_idx;
  switch (dir) {
    case 'r':
      lsite_idx = i;
      rsite_idx = i+1;
      eff_ham[0] = lblocks[i];
      eff_ham[1] = mpo[i];
      eff_ham[2] = mpo[i+1];
      eff_ham[3] = rblocks[N-(i+2)];
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
      eff_ham[0] = lblocks[i-1];
      eff_ham[1] = mpo[i-1];
      eff_ham[2] = mpo[i];
      eff_ham[3] = rblocks[N-i-1];
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

  // Lanczos and SVD.
  auto init_state = Contract(
                        *mps[lsite_idx], *mps[rsite_idx],
                        init_state_ctrct_axes);
  auto lancz_res = LanczosSolver(
                       eff_ham, init_state,
                       sweep_params.LanczParams,
                       where);
  auto svd_res = Svd(
      *lancz_res.gs_vec,
      svd_ldims, svd_rdims,
      Div(*mps[lsite_idx]), Div(*mps[rsite_idx]),
      sweep_params.Cutoff,
      sweep_params.Dmin, sweep_params.Dmax);
  delete lancz_res.gs_vec;

  // Update MPS sites and blocks.
  switch (dir) {
    case 'r':
      delete mps[lsite_idx];
      mps[lsite_idx] = svd_res.u;
      delete mps[rsite_idx];
      mps[rsite_idx] = Contract(*svd_res.s, *svd_res.v, {{1}, {0}});
      delete svd_res.s;
      delete svd_res.v;
      if (i == 0) {
        auto new_lblock = Contract(*mps[i], *mpo[i], {{0}, {0}});
        auto temp_new_lblock = Contract(
                                   *new_lblock, MockDag(*mps[i]),
                                   {{2}, {0}});
        delete new_lblock;
        new_lblock = temp_new_lblock;
        delete lblocks[i+1];
        lblocks[i+1] = new_lblock;
      } else if (i != N-2) {
        auto new_lblock = Contract(*lblocks[i], *mps[i], {{0}, {0}});
        auto temp_new_lblock = Contract(*new_lblock, *mpo[i], {{0, 2}, {0, 1}});
        delete new_lblock;
        new_lblock = temp_new_lblock;
        temp_new_lblock = Contract(*new_lblock, MockDag(*mps[i]), {{0, 2}, {0, 1}});
        delete new_lblock;
        new_lblock = temp_new_lblock;
        delete lblocks[i+1];
        lblocks[i+1] = new_lblock;
      }
      break;
    case 'l':
      delete mps[lsite_idx];
      mps[lsite_idx] = Contract(*svd_res.u, *svd_res.s, us_ctrct_axes);
      delete svd_res.u;
      delete svd_res.s;
      delete mps[rsite_idx];
      mps[rsite_idx] = svd_res.v;
      if (i == N-1) {
        auto new_rblock = Contract(*mps[i], *mpo[i], {{1}, {0}});
        auto temp_new_rblock = Contract(*new_rblock, MockDag(*mps[i]), {{2}, {1}});
        delete new_rblock;
        new_rblock = temp_new_rblock;
        delete rblocks[N-i];
        rblocks[N-i] = new_rblock;
      } else if (i != 1) {
        auto new_rblock = Contract(*mps[i], *eff_ham[3], {{2}, {0}});
        auto temp_new_rblock = Contract(*new_rblock, *mpo[i], {{1, 2}, {1, 3}});
        delete new_rblock;
        new_rblock = temp_new_rblock;
        temp_new_rblock = Contract(*new_rblock, MockDag(*mps[i]), {{3, 1}, {1, 2}});
        delete new_rblock;
        new_rblock = temp_new_rblock;
        delete rblocks[N-i];
        rblocks[N-i] = new_rblock;
      }
      break;
    default:
      std::cout << "dir must be 'r' or 'l', but " << dir << std::endl; 
      exit(1);
  }
  std::cout << "Site " << i << " updated E0 = " << std::setprecision(16) << lancz_res.gs_eng << " TruncErr = " << svd_res.trunc_err << " D = " << svd_res.D << std::endl;
  std::cout << "\n";
  return lancz_res.gs_eng;
}
} /* gqmps2 */ 
