/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-12 10:19
* 
* Description: GraceQ/mps2 project. Lanczos algorithm unittests.
*/
#include "gtest/gtest.h"
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"
#include "testing_utils.h"

#include <vector>
#include <iostream>

#include <assert.h>

#include "mkl.h"

#ifdef Release
  #define NDEBUG
#endif


using namespace gqmps2;
using namespace gqten;


long d = 2;
long D = 20;
long dh = 2;


struct TestLanczos : public testing::Test {
  Index idx_din = Index({QNSector(QN({QNNameVal("Sz", 0)}), d)}, IN);
  Index idx_dout = InverseIndex(idx_din);
  Index idx_Din = Index({QNSector(QN({QNNameVal("Sz", 0)}), D)}, IN);
  Index idx_Dout = InverseIndex(idx_Din);
  Index idx_dh =  Index({QNSector(QN(), dh)});
};


void RunTestCentLanczosSolverCase(
    const std::vector<GQTensor *> &eff_ham,
    GQTensor *init_state,
    const LanczosParams &lanczos_params) {
  std::cout << "\n";
  auto lancz_res = LanczosSolver(
                       eff_ham, init_state,
                       lanczos_params,
                       "cent");

  std::vector<long> ta_ctrct_axes1 = {1};
  std::vector<long> ta_ctrct_axes2 = {4};
  std::vector<long> ta_ctrct_axes3 = {6};
  std::vector<long> tb_ctrct_axes1 = {0};
  std::vector<long> tb_ctrct_axes2 = {0};
  std::vector<long> tb_ctrct_axes3 = {1};
  auto eff_ham_ten = Contract(*eff_ham[0], *eff_ham[1], {{1}, {0}});
  InplaceContract(eff_ham_ten, *eff_ham[2], {{4}, {0}});
  InplaceContract(eff_ham_ten, *eff_ham[3], {{6}, {1}});
  eff_ham_ten->Transpose({0, 2, 4, 6, 1, 3, 5, 7});

  assert(eff_ham_ten->cblocks().size() == 1);
  auto dense_mat = eff_ham_ten->blocks()[0]->data();
  auto dense_mat_dim = D * d * d * D;
  for (long i = 0; i < dense_mat_dim; ++i) {
    for (long j = 0; j < dense_mat_dim; ++j) {
      if (i > j) {
        dense_mat[i*dense_mat_dim + j] = 0.0;
      }
    }
  }
  auto w = new double [dense_mat_dim];
  LAPACKE_dsyev(
      LAPACK_ROW_MAJOR, 'N', 'U',
      dense_mat_dim, dense_mat, dense_mat_dim, w);

  EXPECT_NEAR(lancz_res.gs_eng, w[0], 1.0E-8);

  delete eff_ham_ten;
  delete[] w;
}


TEST_F(TestLanczos, TestCentLanczosSolver) {
  auto lblock = new GQTensor({idx_Dout, idx_dh, idx_Din});
  auto lsite  = new GQTensor({idx_dh, idx_din, idx_dout, idx_dh});
  auto rblock = new GQTensor({idx_Din, idx_dh, idx_Dout});
  auto block_random_mat =  new double [D*D];
  RandRealSymMat(block_random_mat, D);
  for (long i = 0; i < D; ++i) {
    for (long j = 0; j < D; ++j) {
      for (long k = 0; k < dh; ++k) {
        (*lblock)({i, k, j}) = block_random_mat[(i*D+j)];
        (*rblock)({j, k, i}) = block_random_mat[(i*D+j)];
      }
    }
  }
  delete[] block_random_mat;
  auto site_random_mat = new double [d*d];
  RandRealSymMat(site_random_mat, d);
  for (long i = 0; i < d; ++i) {
    for (long j = 0; j < d; ++j) {
      for (long k = 0; k < dh; ++k) {
        (*lsite)({k, i, j, k}) = site_random_mat[(i*d+j)];
      }
    }
  }
  delete[] site_random_mat;
  auto rsite  = new GQTensor(*lsite);
  auto init_state = new GQTensor({idx_Din, idx_dout, idx_dout, idx_Dout});

  // Finish iteration when Lanczos error targeted.
  srand(0);
  init_state->Random(QN({QNNameVal("Sz", 0)}));
  LanczosParams lanczos_params(1.0E-9);
  RunTestCentLanczosSolverCase(
      {lblock, lsite, rsite, rblock},
      init_state,
      lanczos_params);

  // Finish iteration when maximal Lanczos iteration number targeted.
  init_state = new GQTensor({idx_Din, idx_dout, idx_dout, idx_Dout});
  srand(0);
  init_state->Random(QN({QNNameVal("Sz", 0)}));
  LanczosParams lanczos_params2(1.0E-16, 20);
  RunTestCentLanczosSolverCase(
      {lblock, lsite, rsite, rblock},
      init_state,
      lanczos_params2);
}


void RunTestLendLanczosSolverCase(
    const std::vector<GQTensor *> &eff_ham,
    GQTensor *init_state,
    const LanczosParams &lanczos_params) {
  std::cout << "\n";
  auto lancz_res = LanczosSolver(
                       eff_ham, init_state,
                       lanczos_params,
                       "lend");
  std::vector<long> ta_ctrct_axes1 = {1};
  std::vector<long> ta_ctrct_axes2 = {4};
  std::vector<long> tb_ctrct_axes1 = {0};
  std::vector<long> tb_ctrct_axes2 = {1};
  auto eff_ham_ten = Contract(*eff_ham[1], *eff_ham[2], {{1}, {0}});
  InplaceContract(eff_ham_ten, *eff_ham[3], {{4}, {1}});
  eff_ham_ten->Transpose({0, 2, 4, 1, 3, 5});

  assert(eff_ham_ten->cblocks().size() == 1);
  auto dense_mat = eff_ham_ten->blocks()[0]->data();
  auto dense_mat_dim = d * d * D;
  for (long i = 0; i < dense_mat_dim; ++i) {
    for (long j = 0; j < dense_mat_dim; ++j) {
      if (i > j) {
        dense_mat[i*dense_mat_dim + j] = 0.0;
      }
    }
  }
  auto w = new double [dense_mat_dim];
  LAPACKE_dsyev(
      LAPACK_ROW_MAJOR, 'N', 'U',
      dense_mat_dim, dense_mat, dense_mat_dim, w);

  EXPECT_NEAR(lancz_res.gs_eng, w[0], 1.0E-8);

  delete eff_ham_ten;
  delete[] w;
}


TEST_F(TestLanczos, TestLendLanczosSolver) {
  auto lsite = new GQTensor({idx_din, idx_dh, idx_dout});
  auto rsite = new GQTensor({idx_dh, idx_din, idx_dout, idx_dh});
  auto rblock = new GQTensor({idx_Din, idx_dh, idx_Dout});
  auto block_random_mat = new double [D*D];
  RandRealSymMat(block_random_mat, D);
  for (long i = 0; i < D; ++i) {
    for (long j = 0; j < D; ++j) {
      for (long k = 0; k < dh; ++k) {
        (*rblock)({j, k, i}) = block_random_mat[(i*D+j)];
      }
    }
  }
  delete[] block_random_mat;
  auto site_random_mat = new double [d*d];
  RandRealSymMat(site_random_mat, d);
  for (long i = 0; i < d; ++i) {
    for (long j = 0; j < d; ++j) {
      for (long k = 0; k < dh; ++k) {
        (*lsite)({i, k, j}) = site_random_mat[(i*d+j)];
        (*rsite)({k, i, j, k}) = site_random_mat[(i*d+j)];
      }
    }
  }
  delete[] site_random_mat;
  auto null_ten = new GQTensor();
  auto init_state = new GQTensor({idx_dout, idx_dout, idx_Dout});

  srand(0);
  init_state->Random(QN({QNNameVal("Sz", 0)}));
  LanczosParams lanczos_params(1.0E-9);
  RunTestLendLanczosSolverCase(
      {null_ten, lsite, rsite, rblock},
      init_state,
      lanczos_params);
}
