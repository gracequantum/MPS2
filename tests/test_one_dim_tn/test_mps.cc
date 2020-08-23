// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-21 09:13
*
* Description: GraceQ/MPS2 project. Unittests for MPS .
*/
#include "gqmps2/one_dim_tn/mps/mps.h"
#include "gqten/gqten.h"
#include "gtest/gtest.h"

#include <utility>    // move

using namespace gqmps2;
using namespace gqten;

using Tensor = DGQTensor;


struct TestMPS : public testing::Test {
  QN qn0 = QN({QNNameVal("N", 0)});
  QN qn1 = QN({QNNameVal("N", 1)});
  QN qn2 = QN({QNNameVal("N", 2)});
  Index pb_out = Index({QNSector(qn0, 1), QNSector(qn1, 1)}, OUT);
  Index vb01_out = Index({QNSector(qn0, 3), QNSector(qn1, 3)}, OUT);
  Index vb01_in = InverseIndex(vb01_out);
  Index vb012_out = Index(
                        {QNSector(qn0, 3), QNSector(qn1, 3), QNSector(qn2, 3)},
                        OUT
                    );
  Index vb012_in = InverseIndex(vb012_out);
  Tensor t0 = Tensor({pb_out, vb01_out});
  Tensor t1 = Tensor({vb01_in, pb_out, vb012_out});
  Tensor t2 = Tensor({vb012_in, pb_out, vb012_out});
  Tensor t3 = Tensor({vb012_in, pb_out, vb01_out});
  Tensor t4 = Tensor({vb01_in, pb_out});

  MPS<Tensor> mps = MPS<Tensor>(5);

  void SetUp(void) {
    t0.Random(qn0);
    t1.Random(qn0);
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
  auto shape = t.shape;
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], shape[1]);
  for (long i = 0; i < shape[0]; ++i) {
    EXPECT_NEAR(t.Elem({i, i}), 1.0, 1E-15);
  }
}


template <typename ElemT>
void CheckMPSTenCanonical(
    const MPS<ElemT> &mps,
    const size_t i,
    const int center) {
  std::vector<std::vector<long>> ctrct_leg_idxs;
  if (i < center) {
    if (i == 0) {
      ctrct_leg_idxs = {{0}, {0}};
    } else {
      ctrct_leg_idxs = {{0, 1}, {0, 1}};
    }
  } else if (i > center) {
    if (i == mps.size() - 1) {
      ctrct_leg_idxs = {{1}, {1}};
    } else {
      ctrct_leg_idxs = {{1, 2}, {1, 2}};
    }
  }

  Tensor res;
  auto ten = mps[i];
  auto ten_dag = Dag(ten);
  Contract(&ten, &ten_dag, ctrct_leg_idxs, &res);
  CheckIsIdTen(res);
}


template <typename ElemT>
void CheckMPSCenter(const MPS<ElemT> &mps, const int center) {
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


template <typename ElemT>
void RunTestMPSCentralizeCase(MPS<ElemT> &mps, const int center) {
    mps.Centralize(center);
    CheckMPSCenter(mps, center);
}


template <typename ElemT>
void RunTestMPSCentralizeCase(MPS<ElemT> &mps) {
  for (int i = 0; i < mps.size(); ++i) {
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
  const MPS<Tensor> &crmps = mps;

  MPS<Tensor> mps_copy(mps);
  const MPS<Tensor> &crmps_copy = mps_copy;
  EXPECT_EQ(mps_copy.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_copy.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_copy[i], crmps[i]);
    EXPECT_NE(crmps_copy(i), crmps(i));
  }

  MPS<Tensor> mps_copy2 = mps;
  const MPS<Tensor> &crmps_copy2 = mps_copy2;
  EXPECT_EQ(mps_copy2.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_copy2.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_copy2[i], crmps[i]);
    EXPECT_NE(crmps_copy2(i), crmps(i));
  }

  auto craw_data_copy = mps_copy.cdata();
  MPS<Tensor> mps_move(std::move(mps_copy));
  const MPS<Tensor> &crmps_move = mps_move;
  EXPECT_EQ(mps_move.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_move.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_move[i], crmps[i]);
    EXPECT_EQ(crmps_move(i), craw_data_copy[i]);
  }

  auto craw_data_copy2 = mps_copy2.cdata();
  MPS<Tensor> mps_move2 = std::move(mps_copy2);
  const MPS<Tensor> &crmps_move2 = mps_move2;
  EXPECT_EQ(mps_move2.GetCenter(), mps.GetCenter());
  for (size_t i = 0; i < mps.size(); ++i) {
    EXPECT_EQ(mps_move2.GetTenCanoType(i), mps.GetTenCanoType(i));
    EXPECT_EQ(crmps_move2[i], crmps[i]);
    EXPECT_EQ(crmps_move2(i), craw_data_copy2[i]);
  }
}


TEST_F(TestMPS, TestElemAccess) {
  mps.Centralize(2);

  const MPS<Tensor> &crmps = mps;
  Tensor ten = crmps[1];
  EXPECT_EQ(mps.GetTenCanoType(1), MPSTenCanoType::LEFT);
  ten = mps[1];
  EXPECT_EQ(mps.GetTenCanoType(1), MPSTenCanoType::NONE);
  EXPECT_EQ(crmps.GetTenCanoType(1), MPSTenCanoType::NONE);

  const MPS<Tensor> *cpmps = &mps;
  ten = (*cpmps)[3];
  EXPECT_EQ(mps.GetTenCanoType(3), MPSTenCanoType::RIGHT);
  MPS<Tensor> *pmps = &mps;
  ten = (*pmps)[3];
  EXPECT_EQ(mps.GetTenCanoType(3), MPSTenCanoType::NONE);
  EXPECT_EQ(crmps.GetTenCanoType(3), MPSTenCanoType::NONE);
}
