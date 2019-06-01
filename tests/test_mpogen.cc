/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 10:26
* 
* Description: GraceQ/mps2 project. Unittests for MPO generation.
*/
#include "gqmps2/gqmps2.h"
#include "gtest/gtest.h"
#include "gqten/gqten.h"


using namespace gqmps2;
using namespace gqten;


struct TestMpoGenerator : public testing::Test {
  Index phys_idx_out = Index({
                           QNSector(QN({QNNameVal("Sz", -1)}), 1),
                           QNSector(QN({QNNameVal("Sz",  1)}), 1)}, OUT);
  Index phys_idx_in = InverseIndex(phys_idx_out);
  GQTensor sz = GQTensor({phys_idx_in, phys_idx_out});
  QN qn0 = QN({QNNameVal("Sz", 0)});

  void SetUp(void) {
    sz({0, 0}) = -0.5;
    sz({1, 1}) =  0.5;
  }
};


TEST_F(TestMpoGenerator, TestOneSiteOpCase) {
  long N = 4;
  auto mpo_gen = MPOGenerator(N, phys_idx_out, qn0);
  for (long i = 0; i < N; ++i) {
    mpo_gen.AddTerm(1., {OpIdx(sz, i)});
  }
  auto mpo = mpo_gen.Gen();
  auto lmpo_ten = *mpo[0];
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({1, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({0, 1, 0}), -0.5);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({1, 1, 1}), 0.5);
  auto cmpo_ten1 = *mpo[1];
  EXPECT_DOUBLE_EQ(cmpo_ten1.Elem({0, 0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_ten1.Elem({0, 1, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_ten1.Elem({1, 0, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_ten1.Elem({1, 1, 1, 1}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_ten1.Elem({0, 0, 0, 1}), -0.5);
  EXPECT_DOUBLE_EQ(cmpo_ten1.Elem({0, 1, 1, 1}), 0.5);
  auto cmpo_ten2 = *mpo[2];
  EXPECT_DOUBLE_EQ(cmpo_ten2.Elem({0, 0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_ten2.Elem({0, 1, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_ten2.Elem({1, 0, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_ten2.Elem({1, 1, 1, 1}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_ten2.Elem({0, 0, 0, 1}), -0.5);
  EXPECT_DOUBLE_EQ(cmpo_ten2.Elem({0, 1, 1, 1}), 0.5);
  auto rmpo_ten = *mpo[N-1];
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({0, 0, 0}), -0.5);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({1, 0, 1}), 0.5);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({0, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({1, 1, 1}), 1.);
}


TEST_F(TestMpoGenerator, TestTwoSiteOpCase) {
  long N = 3;
  auto mpo_gen = MPOGenerator(N, phys_idx_out, qn0);
  for (long i = 0; i < N-1; ++i) {
    mpo_gen.AddTerm(1., {OpIdx(sz, i), OpIdx(sz, i+1)});
  }
  auto mpo = mpo_gen.Gen();
  auto lmpo_ten = *mpo[0];
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({1, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({0, 1, 0}), 0.);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({1, 1, 1}), 0.);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({0, 2, 0}), -0.5);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({1, 2, 1}), 0.5);
  auto rmpo_ten = *mpo[N-1];
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({0, 0, 0}), 0.);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({1, 0, 1}), 0.);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({0, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({1, 1, 1}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({0, 2, 0}), -0.5);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({1, 2, 1}), 0.5);
}


TEST_F(TestMpoGenerator, TestOneTwoSiteMixCase) {
  long N = 10;
  auto h = 2.33;
  auto sx = GQTensor({phys_idx_in, phys_idx_out});
  sx({0, 1}) = 0.5;
  sx({1, 0}) = 0.5;
  auto mpo_gen = MPOGenerator(N, phys_idx_out, qn0);
  for (long i = 0; i < N; ++i) {
    mpo_gen.AddTerm(h, {OpIdx(sx, i)});
    if (i != N-1) {
      mpo_gen.AddTerm(1., {OpIdx(sz, i), OpIdx(sz, i+1)});
    }
  }
  auto mpo = mpo_gen.Gen();
  auto lmpo_ten = *mpo[0];
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({1, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({0, 1, 1}), h*0.5);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({1, 1, 0}), h*0.5);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({0, 2, 0}), -0.5);
  EXPECT_DOUBLE_EQ(lmpo_ten.Elem({1, 2, 1}), 0.5);
  auto rmpo_ten = *mpo[N-1];
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({0, 0, 1}), h*0.5);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({0, 0, 1}), h*0.5);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({0, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({1, 1, 1}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({0, 2, 0}), -0.5);
  EXPECT_DOUBLE_EQ(rmpo_ten.Elem({1, 2, 1}), 0.5);
  for (long i = 1; i < N-1; ++i) {
    auto cmpo_ten = *mpo[i];
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({0, 0, 0, 0}), 1.);
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({0, 1, 1, 0}), 1.);
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({0, 0, 1, 1}), h*0.5);
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({0, 1, 0, 1}), h*0.5);
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({0, 0, 0, 2}), -0.5);
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({0, 1, 1, 2}), 0.5);
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({0, 0, 0, 2}), -0.5);
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({0, 1, 1, 2}), 0.5);
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({1, 0, 0, 1}), 1.);
    EXPECT_DOUBLE_EQ(cmpo_ten.Elem({1, 1, 1, 1}), 1.);
  }
}
