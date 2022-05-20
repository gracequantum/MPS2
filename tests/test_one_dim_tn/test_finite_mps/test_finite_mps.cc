// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-21 09:13
*
* Description: GraceQ/MPS2 project. Unittests for MPS .
*/
#include "gqmps2/one_dim_tn/mps/finite_mps/finite_mps.h"
#include "gqten/gqten.h"
#include "gtest/gtest.h"

#include <utility>    // move

using namespace gqmps2;
using namespace gqten;

using U1QN = QN<U1QNVal>;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using Tensor = DGQTensor;

using SiteVecT = SiteVec<GQTEN_Double, U1QN>;
using MPST = FiniteMPS<GQTEN_Double, U1QN>;


struct TestMPS : public testing::Test {
  QNT qn0 = QNT({QNCard("N", U1QNVal(0))});
  QNT qn1 = QNT({QNCard("N", U1QNVal(1))});
  QNT qn2 = QNT({QNCard("N", U1QNVal(2))});
  IndexT vb0_in = IndexT({QNSctT(qn0, 1)}, IN);
  IndexT pb_out = IndexT({QNSctT(qn0, 1), QNSctT(qn1, 1)}, OUT);
  IndexT vb01_out = IndexT({QNSctT(qn0, 1), QNSctT(qn1, 1)}, OUT);
  IndexT vb01_in = InverseIndex(vb01_out);
  IndexT vb012_out = IndexT(
                        {QNSctT(qn0, 1), QNSctT(qn1, 2), QNSctT(qn2, 1)},
                        OUT
                    );
  IndexT vb012_in = InverseIndex(vb012_out);
  IndexT vb0_out = InverseIndex(vb0_in);
  Tensor t0 = Tensor({vb0_in, pb_out, vb01_out});
  Tensor t1 = Tensor({vb01_in, pb_out, vb012_out});
  Tensor t2 = Tensor({vb012_in, pb_out, vb012_out});
  Tensor t3 = Tensor({vb012_in, pb_out, vb01_out});
  Tensor t4 = Tensor({vb01_in, pb_out, vb0_out});

  SiteVecT site_vec = SiteVecT(5, pb_out);

  MPST mps = MPST(site_vec);

  void SetUp(void) {
    t0.Random(qn1);
    t1.Random(qn1);
    t2.Random(qn0);
    t3.Random(qn0);
    t4.Random(qn0);
    mps[0] = t0;
    mps[1] = t1;
    mps[2] = t2;
    mps[3] = t3;
    mps[4] = t4;
  }
};


// Helpers for testing MPS centralization.
template <typename TenT>
void CheckIsIdTen(const TenT &t) {
  auto shape = t.GetShape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], shape[1]);
  for (size_t i = 0; i < shape[0]; ++i) {
    for (size_t j = 0; j < shape[1]; ++j) {
      if (i == j) {
        EXPECT_NEAR(t(i, j), 1.0, 1E-15);
      } else {
        EXPECT_NEAR(t(i, j), 0.0, 1E-15);
      }
    }
  }
}


void CheckMPSTenCanonical(
    const MPST &mps,
    const size_t i,
    const int center
) {
  std::vector<std::vector<size_t>> ctrct_leg_idxs;
  if (i < center) {
    ctrct_leg_idxs = {{0, 1}, {0, 1}};
  } else if (i > center) {
    ctrct_leg_idxs = {{1, 2}, {1, 2}};
  }

  Tensor res;
  auto ten = mps[i];
  auto ten_dag = Dag(ten);
  Contract(&ten, &ten_dag, ctrct_leg_idxs, &res);
  CheckIsIdTen(res);
}


void CheckMPSCenter(const MPST &mps, const int center) {
  EXPECT_EQ(mps.GetCenter(), center);
  
  auto mps_size = mps.size();
  auto tens_cano_type = mps.GetTensCanoType();
  for (size_t i = 0; i < mps_size; ++i) {
    if (i < center) {
      EXPECT_EQ(tens_cano_type[i], MPSTenCanoType::LEFT);
      CheckMPSTenCanonical(mps, i, center);
    }
    if (i > center) {
      EXPECT_EQ(tens_cano_type[i], MPSTenCanoType::RIGHT);
      CheckMPSTenCanonical(mps, i, center);
    }
    if (i == center) {
      EXPECT_EQ(tens_cano_type[i], MPSTenCanoType::NONE);
    }
  }
}


void RunTestMPSCentralizeCase(MPST &mps, const int center) {
  mps.Centralize(center);
  CheckMPSCenter(mps, center);
  mkl_free_buffers();
}


void RunTestMPSCentralizeCase(MPST &mps) {
  for (size_t i = 0; i < mps.size(); ++i) {
    RunTestMPSCentralizeCase(mps, i);
  }
}


TEST_F(TestMPS, TestCentralize) {
  RunTestMPSCentralizeCase(mps, 0);
  RunTestMPSCentralizeCase(mps, 2);
  RunTestMPSCentralizeCase(mps, 4);
  RunTestMPSCentralizeCase(mps);

  mps[0].Random(qn0);
  RunTestMPSCentralizeCase(mps, 0);
  RunTestMPSCentralizeCase(mps, 2);
  RunTestMPSCentralizeCase(mps, 4);
  RunTestMPSCentralizeCase(mps);

  mps[1].Random(qn0);
  mps[2].Random(qn0);
  mps[4].Random(qn0);
  RunTestMPSCentralizeCase(mps, 0);
  RunTestMPSCentralizeCase(mps, 2);
  RunTestMPSCentralizeCase(mps, 4);
  RunTestMPSCentralizeCase(mps);
}


TEST_F(TestMPS, TestCopyAndMove) {
  mps.Centralize(2);
  const MPST &crmps = mps;

  MPST mps_copy(mps);
  const MPST &crmps_copy = mps_copy;
  EXPECT_EQ(mps_copy.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_copy.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_copy[i], crmps[i]);
    EXPECT_NE(crmps_copy(i), crmps(i));
  }

  MPST mps_copy2(mps.GetSitesInfo());
  mps_copy2 = mps;
  const MPST &crmps_copy2 = mps_copy2;
  EXPECT_EQ(mps_copy2.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_copy2.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_copy2[i], crmps[i]);
    EXPECT_NE(crmps_copy2(i), crmps(i));
  }

  auto craw_data_copy = mps_copy.cdata();
  MPST mps_move(std::move(mps_copy));
  const MPST &crmps_move = mps_move;
  EXPECT_EQ(mps_move.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_move.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_move[i], crmps[i]);
    EXPECT_EQ(crmps_move(i), craw_data_copy[i]);
  }

  auto craw_data_copy2 = mps_copy2.cdata();
  MPST mps_move2(mps_copy2.GetSitesInfo());
  mps_move2 = std::move(mps_copy2);
  const MPST &crmps_move2 = mps_move2;
  EXPECT_EQ(mps_move2.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_move2.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_move2[i], crmps[i]);
    EXPECT_EQ(crmps_move2(i), craw_data_copy2[i]);
  }
}


TEST_F(TestMPS, TestElemAccess) {
  mps.Centralize(2);

  const MPST &crmps = mps;
  Tensor ten = crmps[1];
  EXPECT_EQ(mps.GetTenCanoType(1), MPSTenCanoType::LEFT);
  ten = mps[1];
  EXPECT_EQ(mps.GetTenCanoType(1), MPSTenCanoType::NONE);
  EXPECT_EQ(crmps.GetTenCanoType(1), MPSTenCanoType::NONE);

  const MPST *cpmps = &mps;
  ten = (*cpmps)[3];
  EXPECT_EQ(mps.GetTenCanoType(3), MPSTenCanoType::RIGHT);
  MPST *pmps = &mps;
  ten = (*pmps)[3];
  EXPECT_EQ(mps.GetTenCanoType(3), MPSTenCanoType::NONE);
  EXPECT_EQ(crmps.GetTenCanoType(3), MPSTenCanoType::NONE);
}


TEST_F(TestMPS, TestIO) {
  MPST mps2(SiteVecT(5, pb_out));
  mps.Dump();
  mps2.Load();
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(mps2[i], mps[i]);
  }

  mps.Dump("mps2");
  mps2.Load("mps2");
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(mps2[i], mps[i]);
  }

  mps.Dump("mps3", true);
  EXPECT_TRUE(mps.empty());
}


TEST_F(TestMPS, TestTruncate) {
  TruncateMPS(mps, 0, 1, 3);

  TruncateMPS(mps, 0, 2, 2);

  mkl_free_buffers();
}
