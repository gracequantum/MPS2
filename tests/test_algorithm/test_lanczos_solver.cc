// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-12 10:19
* 
* Description: GraceQ/mps2 project. Lanczos algorithm unittests.
*/
#include "gqmps2/algorithm/lanczos_solver.h"
#include "../testing_utils.h"
#include "gqten/gqten.h"

#include "gtest/gtest.h"

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


template <typename TenElemType>
void RunTestCentLanczosSolverCase(
    const std::vector<GQTensor<TenElemType> *> &eff_ham,
    GQTensor<TenElemType> *pinit_state,
    const LanczosParams &lanczos_params) {
  std::cout << "\n";
  auto lancz_res = LanczosSolver(
                       eff_ham, pinit_state,
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
  LapackeSyev(
      LAPACK_ROW_MAJOR, 'N', 'U',
      dense_mat_dim, dense_mat, dense_mat_dim, w);

  EXPECT_NEAR(lancz_res.gs_eng, w[0], 1.0E-8);

  delete lancz_res.gs_vec;
  delete eff_ham_ten;
  delete[] w;
}


TEST_F(TestLanczos, TestCentLanczosSolver) {
  // Tensor with double elements.
  auto dlblock = DGQTensor({idx_Dout, idx_dh, idx_Din});
  auto dlsite  = DGQTensor({idx_dh, idx_din, idx_dout, idx_dh});
  auto drblock = DGQTensor({idx_Din, idx_dh, idx_Dout});
  auto dblock_random_mat =  new double [D*D];
  RandRealSymMat(dblock_random_mat, D);
  for (long i = 0; i < D; ++i) {
    for (long j = 0; j < D; ++j) {
      for (long k = 0; k < dh; ++k) {
        dlblock({i, k, j}) = dblock_random_mat[(i*D+j)];
        drblock({j, k, i}) = dblock_random_mat[(i*D+j)];
      }
    }
  }
  delete[] dblock_random_mat;
  auto dsite_random_mat = new double [d*d];
  RandRealSymMat(dsite_random_mat, d);
  for (long i = 0; i < d; ++i) {
    for (long j = 0; j < d; ++j) {
      for (long k = 0; k < dh; ++k) {
        dlsite({k, i, j, k}) = dsite_random_mat[(i*d+j)];
      }
    }
  }
  delete[] dsite_random_mat;
  auto drsite  = DGQTensor(dlsite);
  auto pdinit_state = new DGQTensor({idx_Din, idx_dout, idx_dout, idx_Dout});

  // Finish iteration when Lanczos error targeted.
  srand(0);
  pdinit_state->Random(QN({QNNameVal("Sz", 0)}));
  LanczosParams lanczos_params(1.0E-9);
  RunTestCentLanczosSolverCase(
      {&dlblock, &dlsite, &drsite, &drblock},
      pdinit_state,
      lanczos_params);

  // Finish iteration when maximal Lanczos iteration number targeted.
  pdinit_state = new DGQTensor({idx_Din, idx_dout, idx_dout, idx_Dout});
  srand(0);
  pdinit_state->Random(QN({QNNameVal("Sz", 0)}));
  LanczosParams lanczos_params2(1.0E-16, 20);
  RunTestCentLanczosSolverCase(
      {&dlblock, &dlsite, &drsite, &drblock},
      pdinit_state,
      lanczos_params2);

  // Tensor with complex elements.
  auto zlblock = ZGQTensor({idx_Dout, idx_dh, idx_Din});
  auto zlsite  = ZGQTensor({idx_dh, idx_din, idx_dout, idx_dh});
  auto zrblock = ZGQTensor({idx_Din, idx_dh, idx_Dout});
  auto zblock_random_mat =  new GQTEN_Complex [D*D];
  RandCplxHerMat(zblock_random_mat, D);
  for (long i = 0; i < D; ++i) {
    for (long j = 0; j < D; ++j) {
      for (long k = 0; k < dh; ++k) {
        zlblock({i, k, j}) = zblock_random_mat[(i*D+j)];
        zrblock({j, k, i}) = zblock_random_mat[(i*D+j)];
      }
    }
  }
  delete [] zblock_random_mat;
  auto zsite_random_mat = new GQTEN_Complex [d*d];
  RandCplxHerMat(zsite_random_mat, d);
  for (long i = 0; i < d; ++i) {
    for (long j = 0; j < d; ++j) {
      for (long k = 0; k < dh; ++k) {
        zlsite({k, i, j, k}) = zsite_random_mat[(i*d+j)];
      }
    }
  }
  delete[] zsite_random_mat;
  auto zrsite  = ZGQTensor(zlsite);
  auto pzinit_state = new ZGQTensor({idx_Din, idx_dout, idx_dout, idx_Dout});

  // Finish iteration when Lanczos error targeted.
  srand(0);
  pzinit_state->Random(QN({QNNameVal("Sz", 0)}));
  RunTestCentLanczosSolverCase(
      {&zlblock, &zlsite, &zrsite, &zrblock},
      pzinit_state,
      lanczos_params);
}


template <typename TenElemType>
void RunTestLendLanczosSolverCase(
    const std::vector<GQTensor<TenElemType> *> &eff_ham,
    GQTensor<TenElemType> *pinit_state,
    const LanczosParams &lanczos_params) {
  std::cout << "\n";
  auto lancz_res = LanczosSolver(
                       eff_ham, pinit_state,
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
  LapackeSyev(
      LAPACK_ROW_MAJOR, 'N', 'U',
      dense_mat_dim, dense_mat, dense_mat_dim, w);

  EXPECT_NEAR(lancz_res.gs_eng, w[0], 1.0E-8);

  delete lancz_res.gs_vec;
  delete eff_ham_ten;
  delete[] w;
}


TEST_F(TestLanczos, TestLendLanczosSolver) {
  // Tensor with double element.
  auto dlsite = DGQTensor({idx_din, idx_dh, idx_dout});
  auto drsite = DGQTensor({idx_dh, idx_din, idx_dout, idx_dh});
  auto drblock = DGQTensor({idx_Din, idx_dh, idx_Dout});
  auto dblock_random_mat = new double [D*D];
  RandRealSymMat(dblock_random_mat, D);
  for (long i = 0; i < D; ++i) {
    for (long j = 0; j < D; ++j) {
      for (long k = 0; k < dh; ++k) {
        drblock({j, k, i}) = dblock_random_mat[(i*D+j)];
      }
    }
  }
  delete[] dblock_random_mat;
  auto dsite_random_mat = new double [d*d];
  RandRealSymMat(dsite_random_mat, d);
  for (long i = 0; i < d; ++i) {
    for (long j = 0; j < d; ++j) {
      for (long k = 0; k < dh; ++k) {
        dlsite({i, k, j}) = dsite_random_mat[(i*d+j)];
        drsite({k, i, j, k}) = dsite_random_mat[(i*d+j)];
      }
    }
  }
  delete[] dsite_random_mat;
  auto dnull_ten = DGQTensor();
  auto pdinit_state = new DGQTensor({idx_dout, idx_dout, idx_Dout});

  srand(0);
  pdinit_state->Random(QN({QNNameVal("Sz", 0)}));
  LanczosParams lanczos_params(1.0E-9);
  RunTestLendLanczosSolverCase(
      {&dnull_ten, &dlsite, &drsite, &drblock},
      pdinit_state,
      lanczos_params);

  // Tensor with complex element.
  auto zlsite = ZGQTensor({idx_din, idx_dh, idx_dout});
  auto zrsite = ZGQTensor({idx_dh, idx_din, idx_dout, idx_dh});
  auto zrblock = ZGQTensor({idx_Din, idx_dh, idx_Dout});
  auto zblock_random_mat = new GQTEN_Complex [D*D];
  RandCplxHerMat(zblock_random_mat, D);
  for (long i = 0; i < D; ++i) {
    for (long j = 0; j < D; ++j) {
      for (long k = 0; k < dh; ++k) {
        zrblock({j, k, i}) = zblock_random_mat[(i*D+j)];
      }
    }
  }
  delete[] zblock_random_mat;
  auto zsite_random_mat = new GQTEN_Complex [d*d];
  RandCplxHerMat(zsite_random_mat, d);
  for (long i = 0; i < d; ++i) {
    for (long j = 0; j < d; ++j) {
      for (long k = 0; k < dh; ++k) {
        zlsite({i, k, j}) = zsite_random_mat[(i*d+j)];
        zrsite({k, i, j, k}) = zsite_random_mat[(i*d+j)];
      }
    }
  }
  delete[] zsite_random_mat;
  auto znull_ten = ZGQTensor();
  auto pzinit_state = new ZGQTensor({idx_dout, idx_dout, idx_Dout});

  srand(0);
  pzinit_state->Random(QN({QNNameVal("Sz", 0)}));
  RunTestLendLanczosSolverCase(
      {&znull_ten, &zlsite, &zrsite, &zrblock},
      pzinit_state,
      lanczos_params);
}
