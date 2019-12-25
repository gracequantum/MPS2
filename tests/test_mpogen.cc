// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 10:26
* 
* Description: GraceQ/mps2 project. Unittests for MPO generation.
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"
#include "gqmps2/detail/mpogen/coef_op_alg.h"
#include "testing_utils.h"

#include "gtest/gtest.h"

using namespace gqmps2;
using namespace gqten;

using DMPOGenerator = MPOGenerator<GQTEN_Double>;
using ZMPOGenerator = MPOGenerator<GQTEN_Complex>;

struct TestMpoGenerator : public testing::Test {
  Index phys_idx_out = Index({
                           QNSector(QN({QNNameVal("Sz", -1)}), 1),
                           QNSector(QN({QNNameVal("Sz",  1)}), 1)}, OUT);
  Index phys_idx_in = InverseIndex(phys_idx_out);
  DGQTensor dsz = DGQTensor({phys_idx_in, phys_idx_out});
  ZGQTensor zsz = ZGQTensor({phys_idx_in, phys_idx_out});
  ZGQTensor zsx = ZGQTensor({phys_idx_in, phys_idx_out});
  ZGQTensor zsy = ZGQTensor({phys_idx_in, phys_idx_out});
  DGQTensor did = DGQTensor({phys_idx_in, phys_idx_out});
  ZGQTensor zid = ZGQTensor({phys_idx_in, phys_idx_out});
  QN qn0 = QN({QNNameVal("Sz", 0)});

  void SetUp(void) {
    dsz({0, 0}) = -0.5;
    dsz({1, 1}) =  0.5;
    zsz({0, 0}) = -0.5;
    zsz({1, 1}) =  0.5;
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    zid({0, 0}) = 1;
    zid({1, 1}) = 1;
    zsx({0, 1}) = 1;
    zsx({1, 0}) = 1;
    zsy({0, 1}) = GQTEN_Complex(0, -1);
    zsy({1, 0}) = GQTEN_Complex(0,  1);
  }
};


TEST_F(TestMpoGenerator, TestInitialization) {
  DMPOGenerator mpo_generator(2, phys_idx_out, qn0);
}


TEST_F(TestMpoGenerator, TestAddTermCase1) {
  DMPOGenerator mpo_generator(2, phys_idx_out, qn0);
  mpo_generator.AddTerm(1., {dsz}, {0}, {});
  auto fsm = mpo_generator.GetFSM();
  SparOpReprMat bchmk_m0(1, 1), bchmk_m1(1, 1);
  bchmk_m0.SetElem(0, 0, OpRepr(1));
  bchmk_m1.SetElem(0, 0, kIdOpRepr);
  auto fsm_comp_mat_repr = fsm.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr[1], bchmk_m1);
}


TEST_F(TestMpoGenerator, TestAddTermCase2) {
  DMPOGenerator mpo_generator(2, phys_idx_out, qn0);
  mpo_generator.AddTerm(1., {dsz, dsz}, {0, 1}, {did});
  mpo_generator.AddTerm(1., {dsz, dsz}, {0, 1}, {did});
  auto fsm = mpo_generator.GetFSM();

  auto s = OpRepr(1);
  SparOpReprMat bchmk_m0(1, 1), bchmk_m1(1, 1);
  bchmk_m0.SetElem(0, 0, s);
  bchmk_m1.SetElem(0, 0, s+s);
  auto fsm_comp_mat_repr = fsm.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr[1], bchmk_m1);
}


TEST_F(TestMpoGenerator, TestAddTermCase3) {
  DMPOGenerator mpo_generator(4, phys_idx_out, qn0);
  mpo_generator.AddTerm(1., {dsz, dsz}, {0, 1}, {did});
  mpo_generator.AddTerm(1., {dsz, dsz}, {1, 2}, {did});
  mpo_generator.AddTerm(1., {dsz, dsz}, {2, 3}, {did});
  auto fsm = mpo_generator.GetFSM();

  auto s = OpRepr(1);
  SparOpReprMat bchmk_m0(1, 2), bchmk_m1(2, 3), bchmk_m2(3, 2), bchmk_m3(2, 1);
  bchmk_m0.SetElem(0, 0, s);
  bchmk_m0.SetElem(0, 1, kIdOpRepr);
  bchmk_m1.SetElem(0, 2, s);
  bchmk_m1.SetElem(1, 0, kIdOpRepr);
  bchmk_m1.SetElem(1, 1, s);
  bchmk_m2.SetElem(0, 0, s);
  bchmk_m2.SetElem(1, 1, s);
  bchmk_m2.SetElem(2, 1, kIdOpRepr);
  bchmk_m3.SetElem(0, 0, s);
  bchmk_m3.SetElem(1, 0, kIdOpRepr);
  auto fsm_comp_mat_repr = fsm.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr[1], bchmk_m1);
  EXPECT_EQ(fsm_comp_mat_repr[2], bchmk_m2);
  EXPECT_EQ(fsm_comp_mat_repr[3], bchmk_m3);
}


TEST_F(TestMpoGenerator, TestAddTermCase4) {
  DMPOGenerator mpo_generator(5, phys_idx_out, qn0);
  mpo_generator.AddTerm(1., {dsz, dsz}, {0, 4}, {did});
  mpo_generator.AddTerm(1., {dsz, dsz, dsz}, {1, 2, 4}, {did, did});
  mpo_generator.AddTerm(1., {dsz, dsz, dsz}, {1, 2, 3}, {did, did});
  auto fsm = mpo_generator.GetFSM();

  auto s = OpRepr(1);
  SparOpReprMat bchmk_m0(1, 2), bchmk_m1(2, 2), bchmk_m2(2, 2), bchmk_m3(2, 2), bchmk_m4(2, 1);
  bchmk_m0.SetElem(0, 0, kIdOpRepr);
  bchmk_m0.SetElem(0, 1, s);
  bchmk_m1.SetElem(0, 1, s);
  bchmk_m1.SetElem(1, 0, kIdOpRepr);
  bchmk_m2.SetElem(0, 0, kIdOpRepr);
  bchmk_m2.SetElem(1, 1, s);
  bchmk_m3.SetElem(0, 1, kIdOpRepr);
  bchmk_m3.SetElem(1, 0, s);
  bchmk_m3.SetElem(1, 1, kIdOpRepr);
  bchmk_m4.SetElem(0, 0, kIdOpRepr);
  bchmk_m4.SetElem(0, 1, s);
  auto fsm_comp_mat_repr = fsm.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr[1], bchmk_m1);
  EXPECT_EQ(fsm_comp_mat_repr[2], bchmk_m2);
  EXPECT_EQ(fsm_comp_mat_repr[3], bchmk_m3);
  EXPECT_EQ(fsm_comp_mat_repr[4], bchmk_m4);
}


TEST_F(TestMpoGenerator, TestAddTermCase5) {
  DMPOGenerator mpo_generator(4, phys_idx_out, qn0);
  GQTEN_Double ja = 0.5, jb = 2.0;
  mpo_generator.AddTerm(ja, {dsz, dsz}, {0, 1}, {did});
  mpo_generator.AddTerm(ja, {dsz, dsz}, {0, 2}, {did});
  mpo_generator.AddTerm(ja, {dsz, dsz}, {1, 3}, {did});
  mpo_generator.AddTerm(ja, {dsz, dsz}, {2, 3}, {did});
  mpo_generator.AddTerm(jb, {dsz, dsz}, {0, 3}, {did});
  mpo_generator.AddTerm(jb, {dsz, dsz}, {1, 2}, {did});
  auto fsm = mpo_generator.GetFSM();

  CoefLabel j1 = 1, j2 = 2;
  OpLabel s = 1;
  OpRepr op_s(s);
  SparOpReprMat bchmk_m0(1, 2), bchmk_m1(2, 4), bchmk_m2(4, 2), bchmk_m3(2, 1);
  bchmk_m0.SetElem(0, 0, kIdOpRepr);
  bchmk_m0.SetElem(0, 1, op_s);
  bchmk_m1.SetElem(0, 0, OpRepr(j1, kIdOpLabel));
  bchmk_m1.SetElem(0, 3, op_s);
  bchmk_m1.SetElem(1, 1, OpRepr(j1, s));
  bchmk_m1.SetElem(1, 2, kIdOpRepr);
  bchmk_m2.SetElem(0, 0, op_s);
  bchmk_m2.SetElem(1, 1, kIdOpRepr);
  bchmk_m2.SetElem(2, 0, OpRepr(j2, kIdOpLabel));
  bchmk_m2.SetElem(2, 1, OpRepr(j1, s));
  bchmk_m2.SetElem(3, 0, OpRepr(j1, kIdOpLabel));
  bchmk_m2.SetElem(3, 1, OpRepr(j2, s));
  bchmk_m3.SetElem(0, 0, op_s);
  bchmk_m3.SetElem(1, 0, kIdOpRepr);
  auto fsm_comp_mat_repr = fsm.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr[1], bchmk_m1);
  EXPECT_EQ(fsm_comp_mat_repr[2], bchmk_m2);
  EXPECT_EQ(fsm_comp_mat_repr[3], bchmk_m3);
}


TEST_F(TestMpoGenerator, TestAddTermCase6) {
  ZMPOGenerator mpo_generator(3, phys_idx_out, qn0);
  GQTEN_Complex J = 0.5, K = 2.0;
  mpo_generator.AddTerm(J, {zsx, zsx}, {0, 1}, {zid});
  mpo_generator.AddTerm(J, {zsy, zsy}, {0, 1}, {zid});
  mpo_generator.AddTerm(J, {zsz, zsz}, {0, 1}, {zid});
  mpo_generator.AddTerm(J, {zsx, zsx}, {1, 2}, {zid});
  mpo_generator.AddTerm(J, {zsy, zsy}, {1, 2}, {zid});
  mpo_generator.AddTerm(J, {zsz, zsz}, {1, 2}, {zid});
  mpo_generator.AddTerm(K, {zsx, zsx}, {0, 1}, {zid});
  mpo_generator.AddTerm(K, {zsz, zsz}, {1, 2}, {zid});
  auto fsm = mpo_generator.GetFSM();

  CoefLabel j = 1, k = 2;
  OpLabel sx = 1, sy = 2, sz = 3;
  SparOpReprMat bchmk_m0(1, 4), bchmk_m1(4, 4), bchmk_m2(4, 1);
  bchmk_m0.SetElem(0, 0, OpRepr({k, j}, {sx, sx}));
  bchmk_m0.SetElem(0, 1, OpRepr(j, sy));
  bchmk_m0.SetElem(0, 2, OpRepr(j, sz));
  bchmk_m0.SetElem(0, 3, kIdOpRepr);
  bchmk_m1.SetElem(0, 3, OpRepr(sx));
  bchmk_m1.SetElem(1, 3, OpRepr(sy));
  bchmk_m1.SetElem(2, 3, OpRepr(sz));
  bchmk_m1.SetElem(3, 0, OpRepr(j, sx));
  bchmk_m1.SetElem(3, 1, OpRepr(j, sy));
  bchmk_m1.SetElem(3, 2, OpRepr({j, k}, {sz, sz}));
  bchmk_m2.SetElem(0, 0, OpRepr(sx));
  bchmk_m2.SetElem(1, 0, OpRepr(sy));
  bchmk_m2.SetElem(2, 0, OpRepr(sz));
  bchmk_m2.SetElem(3, 0, kIdOpRepr);
  auto fsm_comp_mat_repr = fsm.GenCompressedMatRepr();
  EXPECT_EQ(fsm_comp_mat_repr[0], bchmk_m0);
  EXPECT_EQ(fsm_comp_mat_repr[1], bchmk_m1);
  EXPECT_EQ(fsm_comp_mat_repr[2], bchmk_m2);
}



//struct TestMpoGenerator : public testing::Test {
  //Index phys_idx_out = Index({
                           //QNSector(QN({QNNameVal("Sz", -1)}), 1),
                           //QNSector(QN({QNNameVal("Sz",  1)}), 1)}, OUT);
  //Index phys_idx_in = InverseIndex(phys_idx_out);
  //DGQTensor dsz = DGQTensor({phys_idx_in, phys_idx_out});
  //ZGQTensor zsz = ZGQTensor({phys_idx_in, phys_idx_out});
  //QN qn0 = QN({QNNameVal("Sz", 0)});

  //void SetUp(void) {
    //dsz({0, 0}) = -0.5;
    //dsz({1, 1}) =  0.5;
    //zsz({0, 0}) = -0.5;
    //zsz({1, 1}) =  0.5;
  //}
//};


//TEST_F(TestMpoGenerator, TestOneSiteOpCase) {
  //long N = 4;
  //auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, phys_idx_out, qn0);
  //auto dcoef = Rand();
  //for (long i = 0; i < N; ++i) {
    //dmpo_gen.AddTerm(dcoef, {dsz}, {i});
  //}
  //auto dmpo = dmpo_gen.Gen();
  //auto lmpo_dten = *dmpo[0];
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 0, 0}), 1.);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 0, 1}), 1.);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 1, 0}), -0.5*dcoef);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 1, 1}),  0.5*dcoef);
  //auto cmpo_dten1 = *dmpo[1];
  //EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({0, 0, 0, 0}), 1.);
  //EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({0, 1, 1, 0}), 1.);
  //EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({1, 0, 0, 1}), 1.);
  //EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({1, 1, 1, 1}), 1.);
  //EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({0, 0, 0, 1}), -0.5*dcoef);
  //EXPECT_DOUBLE_EQ(cmpo_dten1.Elem({0, 1, 1, 1}),  0.5*dcoef);
  //auto cmpo_dten2 = *dmpo[2];
  //EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({0, 0, 0, 0}), 1.);
  //EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({0, 1, 1, 0}), 1.);
  //EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({1, 0, 0, 1}), 1.);
  //EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({1, 1, 1, 1}), 1.);
  //EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({0, 0, 0, 1}), -0.5*dcoef);
  //EXPECT_DOUBLE_EQ(cmpo_dten2.Elem({0, 1, 1, 1}),  0.5*dcoef);
  //auto rmpo_dten = *dmpo[N-1];
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 0, 0}), -0.5*dcoef);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 0, 1}),  0.5*dcoef);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 1, 0}), 1.);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 1, 1}), 1.);

  //auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, phys_idx_out, qn0);
  //auto zcoef = GQTEN_Complex(Rand(), Rand());
  //for (long i = 0; i < N; ++i) {
    //zmpo_gen.AddTerm(zcoef, {zsz}, {i});
  //}
  //auto zmpo = zmpo_gen.Gen();
  //auto lmpo_zten = *zmpo[0];
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 0, 0}), 1.);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 0, 1}), 1.);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 1, 0}), -0.5*zcoef);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 1, 1}),  0.5*zcoef);
  //auto cmpo_zten1 = *zmpo[1];
  //EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({0, 0, 0, 0}), 1.);
  //EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({0, 1, 1, 0}), 1.);
  //EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({1, 0, 0, 1}), 1.);
  //EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({1, 1, 1, 1}), 1.);
  //EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({0, 0, 0, 1}), -0.5*zcoef);
  //EXPECT_COMPLEX_EQ(cmpo_zten1.Elem({0, 1, 1, 1}),  0.5*zcoef);
  //auto cmpo_zten2 = *zmpo[2];
  //EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({0, 0, 0, 0}), 1.);
  //EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({0, 1, 1, 0}), 1.);
  //EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({1, 0, 0, 1}), 1.);
  //EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({1, 1, 1, 1}), 1.);
  //EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({0, 0, 0, 1}), -0.5*zcoef);
  //EXPECT_COMPLEX_EQ(cmpo_zten2.Elem({0, 1, 1, 1}),  0.5*zcoef);
  //auto rmpo_zten = *zmpo[N-1];
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 0, 0}), -0.5*zcoef);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 0, 1}),  0.5*zcoef);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 1, 0}), 1.);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 1, 1}), 1.);
//}


//TEST_F(TestMpoGenerator, TestTwoSiteOpCase) {
  //long N = 3;
  //auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, phys_idx_out, qn0);
  //auto dcoef = Rand();
  //for (long i = 0; i < N-1; ++i) {
    //dmpo_gen.AddTerm(dcoef, {dsz, dsz}, {i, i+1});
  //}
  //auto dmpo = dmpo_gen.Gen();
  //auto lmpo_dten = *dmpo[0];
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 0, 0}), 1.);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 0, 1}), 1.);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 1, 0}), 0.);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 1, 1}), 0.);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 2, 0}), -0.5*dcoef);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 2, 1}),  0.5*dcoef);
  //auto rmpo_dten = *dmpo[N-1];
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 0, 0}), 0.);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 0, 1}), 0.);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 1, 0}), 1.);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 1, 1}), 1.);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 2, 0}), -0.5);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 2, 1}),  0.5);

  //auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, phys_idx_out, qn0);
  //auto zcoef = GQTEN_Complex(Rand(), Rand());
  //for (long i = 0; i < N-1; ++i) {
    //zmpo_gen.AddTerm(zcoef, {zsz, zsz}, {i, i+1});
  //}
  //auto zmpo = zmpo_gen.Gen();
  //auto lmpo_zten = *zmpo[0];
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 0, 0}), 1.);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 0, 1}), 1.);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 1, 0}), 0.);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 1, 1}), 0.);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 2, 0}), -0.5*zcoef);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 2, 1}),  0.5*zcoef);
  //auto rmpo_zten = *zmpo[N-1];
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 0, 0}), 0.);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 0, 1}), 0.);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 1, 0}), 1.);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 1, 1}), 1.);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 2, 0}), -0.5);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 2, 1}),  0.5);
//}


//TEST_F(TestMpoGenerator, TestOneTwoSiteMixCase) {
  //long N = 10;
  //auto h = 2.33;
  //auto dsx = DGQTensor({phys_idx_in, phys_idx_out});
  //dsx({0, 1}) = 0.5;
  //dsx({1, 0}) = 0.5;
  //auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, phys_idx_out, qn0);
  //for (long i = 0; i < N; ++i) {
    //dmpo_gen.AddTerm(h, {dsx}, {i});
    //if (i != N-1) {
      //dmpo_gen.AddTerm(1., {dsz, dsz}, {i, i+1});
    //}
  //}
  //auto dmpo = dmpo_gen.Gen();
  //auto lmpo_dten = *dmpo[0];
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 0, 0}), 1.);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 0, 1}), 1.);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 1, 1}), h*0.5);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 1, 0}), h*0.5);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({0, 2, 0}), -0.5);
  //EXPECT_DOUBLE_EQ(lmpo_dten.Elem({1, 2, 1}), 0.5);
  //auto rmpo_dten = *dmpo[N-1];
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 0, 1}), h*0.5);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 0, 1}), h*0.5);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 1, 0}), 1.);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 1, 1}), 1.);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({0, 2, 0}), -0.5);
  //EXPECT_DOUBLE_EQ(rmpo_dten.Elem({1, 2, 1}), 0.5);
  //for (long i = 1; i < N-1; ++i) {
    //auto cmpo_dten = *dmpo[i];
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 0, 0, 0}), 1.);
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 1, 1, 0}), 1.);
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 0, 1, 1}), h*0.5);
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 1, 0, 1}), h*0.5);
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 0, 0, 2}), -0.5);
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 1, 1, 2}), 0.5);
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 0, 0, 2}), -0.5);
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({0, 1, 1, 2}), 0.5);
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({1, 0, 0, 1}), 1.);
    //EXPECT_DOUBLE_EQ(cmpo_dten.Elem({1, 1, 1, 1}), 1.);
  //}

  //auto zsx = ZGQTensor({phys_idx_in, phys_idx_out});
  //zsx({0, 1}) = 0.5;
  //zsx({1, 0}) = 0.5;
  //auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, phys_idx_out, qn0);
  //for (long i = 0; i < N; ++i) {
    //zmpo_gen.AddTerm(h, {zsx}, {i});
    //if (i != N-1) {
      //zmpo_gen.AddTerm(1., {zsz, zsz}, {i, i+1});
    //}
  //}
  //auto zmpo = zmpo_gen.Gen();
  //auto lmpo_zten = *zmpo[0];
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 0, 0}), 1.);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 0, 1}), 1.);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 1, 1}), h*0.5);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 1, 0}), h*0.5);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({0, 2, 0}), -0.5);
  //EXPECT_COMPLEX_EQ(lmpo_zten.Elem({1, 2, 1}), 0.5);
  //auto rmpo_zten = *zmpo[N-1];
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 0, 1}), h*0.5);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 0, 1}), h*0.5);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 1, 0}), 1.);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 1, 1}), 1.);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({0, 2, 0}), -0.5);
  //EXPECT_COMPLEX_EQ(rmpo_zten.Elem({1, 2, 1}), 0.5);
  //for (long i = 1; i < N-1; ++i) {
    //auto cmpo_zten = *zmpo[i];
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 0, 0, 0}), 1.);
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 1, 1, 0}), 1.);
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 0, 1, 1}), h*0.5);
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 1, 0, 1}), h*0.5);
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 0, 0, 2}), -0.5);
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 1, 1, 2}), 0.5);
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 0, 0, 2}), -0.5);
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({0, 1, 1, 2}), 0.5);
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({1, 0, 0, 1}), 1.);
    //EXPECT_COMPLEX_EQ(cmpo_zten.Elem({1, 1, 1, 1}), 1.);
  //}
//}
