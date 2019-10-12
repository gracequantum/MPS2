// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 10:26
* 
* Description: GraceQ/mps2 project. Unittests for MPO generation.
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"
#include "testing_utils.h"

#include "gtest/gtest.h"

using namespace gqmps2;
using namespace gqten;


struct TestMpoGenerator : public testing::Test {
  Index phys_idx_out = Index({
                           QNSector(QN({QNNameVal("Sz", -1)}), 1),
                           QNSector(QN({QNNameVal("Sz",  1)}), 1)}, OUT);
  Index phys_idx_in = InverseIndex(phys_idx_out);
  DGQTensor dsz = DGQTensor({phys_idx_in, phys_idx_out});
  ZGQTensor zsz = ZGQTensor({phys_idx_in, phys_idx_out});
  QN qn0 = QN({QNNameVal("Sz", 0)});

  void SetUp(void) {
    dsz({0, 0}) = -0.5;
    dsz({1, 1}) =  0.5;
    zsz({0, 0}) = -0.5;
    zsz({1, 1}) =  0.5;
  }
};


TEST_F(TestMpoGenerator, TestOneSiteOpCase) {
  long N = 4;
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, phys_idx_out, qn0);
  auto dcoef = Rand();
  for (long i = 0; i < N; ++i) {
    dmpo_gen.AddTerm(dcoef, {dsz}, {i});
  }
  auto dmpo = dmpo_gen.Gen();
  auto lmpo_dten = *dmpo[0];
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 1, 0}), -0.5*dcoef);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 1, 1}),  0.5*dcoef);
  auto cmpo_dten1 = *dmpo[1];
  EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({0, 0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({0, 1, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({1, 0, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({1, 1, 1, 1}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({0, 0, 0, 1}), -0.5*dcoef);
  EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({0, 1, 1, 1}),  0.5*dcoef);
  auto cmpo_dten2 = *dmpo[2];
  EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({0, 0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({0, 1, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({1, 0, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({1, 1, 1, 1}), 1.);
  EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({0, 0, 0, 1}), -0.5*dcoef);
  EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({0, 1, 1, 1}),  0.5*dcoef);
  auto rmpo_dten = *dmpo[N-1];
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 0, 0}), -0.5*dcoef);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 0, 1}),  0.5*dcoef);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 1, 1}), 1.);

  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, phys_idx_out, qn0);
  auto zcoef = GQTEN_Complex(Rand(), Rand());
  for (long i = 0; i < N; ++i) {
    zmpo_gen.AddTerm(zcoef, {zsz}, {i});
  }
  auto zmpo = zmpo_gen.Gen();
  auto lmpo_zten = *zmpo[0];
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 0, 0}), 1.);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 0, 1}), 1.);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 1, 0}), -0.5*zcoef);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 1, 1}),  0.5*zcoef);
  auto cmpo_zten1 = *zmpo[1];
  EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({0, 0, 0, 0}), 1.);
  EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({0, 1, 1, 0}), 1.);
  EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({1, 0, 0, 1}), 1.);
  EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({1, 1, 1, 1}), 1.);
  EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({0, 0, 0, 1}), -0.5*zcoef);
  EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({0, 1, 1, 1}),  0.5*zcoef);
  auto cmpo_zten2 = *zmpo[2];
  EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({0, 0, 0, 0}), 1.);
  EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({0, 1, 1, 0}), 1.);
  EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({1, 0, 0, 1}), 1.);
  EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({1, 1, 1, 1}), 1.);
  EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({0, 0, 0, 1}), -0.5*zcoef);
  EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({0, 1, 1, 1}),  0.5*zcoef);
  auto rmpo_zten = *zmpo[N-1];
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 0, 0}), -0.5*zcoef);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 0, 1}),  0.5*zcoef);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 1, 0}), 1.);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 1, 1}), 1.);
}


TEST_F(TestMpoGenerator, TestTwoSiteOpCase) {
  long N = 3;
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, phys_idx_out, qn0);
  auto dcoef = Rand();
  for (long i = 0; i < N-1; ++i) {
    dmpo_gen.AddTerm(dcoef, {dsz, dsz}, {i, i+1});
  }
  auto dmpo = dmpo_gen.Gen();
  auto lmpo_dten = *dmpo[0];
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 1, 0}), 0.);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 1, 1}), 0.);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 2, 0}), -0.5*dcoef);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 2, 1}),  0.5*dcoef);
  auto rmpo_dten = *dmpo[N-1];
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 0, 0}), 0.);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 0, 1}), 0.);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 1, 1}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 2, 0}), -0.5);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 2, 1}),  0.5);

  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, phys_idx_out, qn0);
  auto zcoef = GQTEN_Complex(Rand(), Rand());
  for (long i = 0; i < N-1; ++i) {
    zmpo_gen.AddTerm(zcoef, {zsz, zsz}, {i, i+1});
  }
  auto zmpo = zmpo_gen.Gen();
  auto lmpo_zten = *zmpo[0];
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 0, 0}), 1.);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 0, 1}), 1.);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 1, 0}), 0.);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 1, 1}), 0.);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 2, 0}), -0.5*zcoef);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 2, 1}),  0.5*zcoef);
  auto rmpo_zten = *zmpo[N-1];
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 0, 0}), 0.);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 0, 1}), 0.);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 1, 0}), 1.);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 1, 1}), 1.);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 2, 0}), -0.5);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 2, 1}),  0.5);
}


TEST_F(TestMpoGenerator, TestOneTwoSiteMixCase) {
  long N = 10;
  auto h = 2.33;
  auto dsx = DGQTensor({phys_idx_in, phys_idx_out});
  dsx({0, 1}) = 0.5;
  dsx({1, 0}) = 0.5;
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, phys_idx_out, qn0);
  for (long i = 0; i < N; ++i) {
    dmpo_gen.AddTerm(h, {dsx}, {i});
    if (i != N-1) {
      dmpo_gen.AddTerm(1., {dsz, dsz}, {i, i+1});
    }
  }
  auto dmpo = dmpo_gen.Gen();
  auto lmpo_dten = *dmpo[0];
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 0, 0}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 0, 1}), 1.);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 1, 1}), h*0.5);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 1, 0}), h*0.5);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 2, 0}), -0.5);
  EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 2, 1}), 0.5);
  auto rmpo_dten = *dmpo[N-1];
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 0, 1}), h*0.5);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 0, 1}), h*0.5);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 1, 0}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 1, 1}), 1.);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 2, 0}), -0.5);
  EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 2, 1}), 0.5);
  for (long i = 1; i < N-1; ++i) {
    auto cmpo_dten = *dmpo[i];
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 0, 0, 0}), 1.);
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 1, 1, 0}), 1.);
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 0, 1, 1}), h*0.5);
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 1, 0, 1}), h*0.5);
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 0, 0, 2}), -0.5);
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 1, 1, 2}), 0.5);
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 0, 0, 2}), -0.5);
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 1, 1, 2}), 0.5);
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({1, 0, 0, 1}), 1.);
    EXPECT_DOUBLE_EQ(cmpo_dten.Elem({1, 1, 1, 1}), 1.);
  }

  auto zsx = ZGQTensor({phys_idx_in, phys_idx_out});
  zsx({0, 1}) = 0.5;
  zsx({1, 0}) = 0.5;
  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, phys_idx_out, qn0);
  for (long i = 0; i < N; ++i) {
    zmpo_gen.AddTerm(h, {zsx}, {i});
    if (i != N-1) {
      zmpo_gen.AddTerm(1., {zsz, zsz}, {i, i+1});
    }
  }
  auto zmpo = zmpo_gen.Gen();
  auto lmpo_zten = *zmpo[0];
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 0, 0}), 1.);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 0, 1}), 1.);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 1, 1}), h*0.5);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 1, 0}), h*0.5);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 2, 0}), -0.5);
  EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 2, 1}), 0.5);
  auto rmpo_zten = *zmpo[N-1];
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 0, 1}), h*0.5);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 0, 1}), h*0.5);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 1, 0}), 1.);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 1, 1}), 1.);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 2, 0}), -0.5);
  EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 2, 1}), 0.5);
  for (long i = 1; i < N-1; ++i) {
    auto cmpo_zten = *zmpo[i];
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 0, 0, 0}), 1.);
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 1, 1, 0}), 1.);
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 0, 1, 1}), h*0.5);
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 1, 0, 1}), h*0.5);
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 0, 0, 2}), -0.5);
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 1, 1, 2}), 0.5);
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 0, 0, 2}), -0.5);
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 1, 1, 2}), 0.5);
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({1, 0, 0, 1}), 1.);
    EXPECT_COMPLEX_EQ(cmpo_zten.Elem({1, 1, 1, 1}), 1.);
  }
}
