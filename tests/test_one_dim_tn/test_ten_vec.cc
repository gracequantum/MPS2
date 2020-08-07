// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-06 16:40
* 
* Description: GraceQ/MPS2 project. Unittests for TenVec .
*/
#include "gqmps2/one_dim_tn/ten_vec.h"
#include "gqten/gqten.h"

#include "gtest/gtest.h"


using namespace gqmps2;
using namespace gqten;

using Tensor = DGQTensor;


struct TestTenVec : public testing::Test {
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
void RunTestTenVecConstructor1Case(const size_t N) {
  TenVec<TenT> ten_vec(N);
  EXPECT_EQ(ten_vec.size(), N);
  auto raw_data = ten_vec.cdata();
  for (auto &dat : raw_data) {
    EXPECT_EQ(dat, nullptr);
  }
}


template <typename TenT>
void RunTestTenVecConstructor2Case(const std::vector<TenT> &tens) {
  TenVec<TenT> ten_vec(tens);
  EXPECT_EQ(ten_vec.size(), tens.size());
  auto raw_data = ten_vec.cdata();
  for (size_t i = 0; i < tens.size(); ++i) {
    EXPECT_EQ(*raw_data[i], tens[i]);
  }
}


TEST_F(TestTenVec, TestConstructors) {
  RunTestTenVecConstructor1Case<Tensor>(0);
  RunTestTenVecConstructor1Case<Tensor>(5);

  RunTestTenVecConstructor2Case<Tensor>({});
  RunTestTenVecConstructor2Case<Tensor>({ten1});
  RunTestTenVecConstructor2Case<Tensor>({ten2, ten2});
  RunTestTenVecConstructor2Case<Tensor>({ten2, ten1, ten3});
}


TEST_F(TestTenVec, TestRawDataAccess) {
  TenVec<Tensor> ten_vec({ten1, ten2, ten3});
  auto raw_data = ten_vec.data();
  delete raw_data[0];
  raw_data[0] = new Tensor(ten3);
  auto craw_data = ten_vec.cdata();
  EXPECT_EQ(*craw_data[0], ten3);
}


TEST_F(TestTenVec, TestElemAccess) {
  TenVec<Tensor> ten_vec({ten1, ten2, ten3});
  auto elem0 = ten_vec(0);
  EXPECT_EQ(*elem0, ten1);

  delete ten_vec(1);
  ten_vec(1) = new Tensor(ten3);
  EXPECT_EQ(*ten_vec(1), ten3);
}
