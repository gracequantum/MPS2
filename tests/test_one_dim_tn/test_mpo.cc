// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-06 16:40
* 
* Description: GraceQ/MPS2 project. Unittests for MPO .
*/
#include "gqmps2/one_dim_tn/mpo.h"
#include "gqten/gqten.h"

#include "gtest/gtest.h"


using namespace gqmps2;
using namespace gqten;

using Tensor = DGQTensor;


struct TestMPO : public testing::Test {
  QN qn = QN({QNNameVal("N", 0)}); 
  Index idx_out = Index({QNSector(qn, 3)}, OUT);
  Index idx_in  = Index({QNSector(qn, 4)}, IN);
  Tensor ten1 = Tensor({idx_out});
  Tensor ten2 = Tensor({idx_in, idx_out});
  Tensor ten3 = Tensor({idx_in, idx_out, idx_out});

  void SetUp(void) {
    ten1.Random(qn);
    ten2.Random(qn);
    ten3.Random(qn);
  }
};


template <typename TenT>
void RunTestMPOConstructor1Case(const int N) {
  MPO<TenT> mpo(N);
  EXPECT_EQ(mpo.size(), N);
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(mpo[i], TenT()); 
  }
}


TEST_F(TestMPO, TestConstructors) {
  RunTestMPOConstructor1Case<Tensor>(0);
  RunTestMPOConstructor1Case<Tensor>(5);
}


TEST_F(TestMPO, TestElemAccess) {
  MPO<Tensor> mpo(3);
  mpo[0] = ten1;
  EXPECT_EQ(mpo[0], ten1);
  mpo[1] = ten2;
  EXPECT_EQ(mpo[1], ten2);
  mpo[2] = ten3;
  EXPECT_EQ(mpo[2], ten3);

  mpo[1] = ten3;
  EXPECT_EQ(mpo[1], ten3);
}


TEST_F(TestMPO, TestSharedCopy) {
  MPO<Tensor> mpo1(3);
  MPO<Tensor> mpo2(mpo1);

  mpo1[0] = ten1;
  EXPECT_EQ(mpo2[0], ten1);

  mpo2[1] = ten2;
  EXPECT_EQ(mpo1[1], ten2);

  mpo2[0] = ten3;
  EXPECT_EQ(mpo1[0], ten3);

  auto mpo3 = mpo2;
  EXPECT_EQ(mpo3[0], ten3);
  EXPECT_EQ(mpo3[1], ten2);
  EXPECT_EQ(mpo3[2], Tensor());
}
