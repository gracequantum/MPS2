// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 16:08
* 
* Description: GraceQ/mps2 project. Unittest for two sites algorithm.
*/
#include "gqmps2/gqmps2.h"
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <vector>


using namespace gqmps2;
using namespace gqten;
using DTenPtrVec = std::vector<DGQTensor *>;


struct TestTwoSiteAlgorithmSpinSystem : public testing::Test {
  long N = 6;

  QN qn0 = QN({QNNameVal("Sz", 0)});
  Index pb_out = Index({
                     QNSector(QN({QNNameVal("Sz", 1)}), 1),
                     QNSector(QN({QNNameVal("Sz", -1)}), 1)}, OUT);
  Index pb_in = InverseIndex(pb_out);
  DGQTensor dsz = DGQTensor({pb_in, pb_out});
  DGQTensor dsp = DGQTensor({pb_in, pb_out});
  DGQTensor dsm = DGQTensor({pb_in, pb_out});
  DTenPtrVec dmps = DTenPtrVec(N);

  void SetUp(void) {
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;
  }
};


template <typename TenType>
void RunTestTwoSiteAlgorithmCase(
    std::vector<TenType *> &mps, const std::vector<TenType *> &mpo,
    const SweepParams &sweep_params,
    const double benmrk_e0, const double precision) {
  auto e0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(e0, benmrk_e0, precision);
}


TEST_F(TestTwoSiteAlgorithmSpinSystem, 1DIsing) {
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, pb_out, qn0);
  for (long i = 0; i < N-1; ++i) {
    dmpo_gen.AddTerm(1, {dsz, dsz}, {i, i+1});
  }
  auto dmpo = dmpo_gen.Gen();


  auto sweep_params = SweepParams(
                          4,
                          1, 10, 1.0E-5,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-7));
  RandomInitMps(dmps, pb_out, qn0, qn0, 2);
  RunTestTwoSiteAlgorithmCase(dmps, dmpo, sweep_params, -0.25*(N-1), 1.0E-10);

  // No file I/O case.
  sweep_params = SweepParams(
                     2,
                     1, 10, 1.0E-5,
                     false,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7));

  RandomInitMps(dmps, pb_out, qn0, qn0, 2);
  RunTestTwoSiteAlgorithmCase(dmps, dmpo, sweep_params, -0.25*(N-1), 1.0E-10);
}


TEST_F(TestTwoSiteAlgorithmSpinSystem, 1DHeisenberg) {
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, pb_out, qn0);
  for (long i = 0; i < N-1; ++i) {
    dmpo_gen.AddTerm(1,   {dsz, dsz}, {i, i+1});
    dmpo_gen.AddTerm(0.5, {dsp, dsm}, {i, i+1});
    dmpo_gen.AddTerm(0.5, {dsm, dsp}, {i, i+1});
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = SweepParams(
                     4,
                     8, 8, 1.0E-9,
                     true,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7));
  RandomInitMps(dmps, pb_out, qn0, qn0, 4);
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -2.493577133888, 1.0E-12);

  // Continue simulation test.
  DumpMps(dmps);
  for (auto &mps_ten : dmps) { delete mps_ten; }
  LoadMps(dmps);

  sweep_params = SweepParams(
                     4,
                     8, 8, 1.0E-9,
                     true,
                     kTwoSiteAlgoWorkflowContinue,
                     LanczosParams(1.0E-7));
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -2.493577133888, 1.0E-12);
}


TEST_F(TestTwoSiteAlgorithmSpinSystem, 2DHeisenberg) {
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, pb_out, qn0);
  std::vector<std::pair<long, long>> nn_pairs = {
      std::make_pair(0, 1), 
      std::make_pair(0, 2), 
      std::make_pair(1, 3), 
      std::make_pair(2, 3), 
      std::make_pair(2, 4), 
      std::make_pair(3, 5), 
      std::make_pair(4, 5)
  };
  for (auto &p : nn_pairs) {
    dmpo_gen.AddTerm(1,   {dsz, dsz}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {dsp, dsm}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5, {dsm, dsp}, {p.first, p.second});
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = SweepParams(
                     4,
                     8, 8, 1.0E-9,
                     true,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7));
  RandomInitMps(dmps, pb_out, qn0, qn0, 4);
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -3.129385241572, 1.0E-12);

  // Test direct product state initialization.
  std::vector<long> stat_labs;
  for (int i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(dmps, stat_labs, pb_out, qn0);

  sweep_params = SweepParams(
                     4,
                     8, 8, 1.0E-9,
                     true,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7));
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -3.129385241572, 1.0E-12);
}


// Test Fermion models.
struct TestTwoSiteAlgorithmTjSystem : public testing::Test {
  long N = 4;
  double t = 3.;
  double J = 1.;
  QN qn0 = QN({QNNameVal("N", 0), QNNameVal("Sz", 0)});
  Index pb_out = Index({
      QNSector(QN({QNNameVal("N", 1), QNNameVal("Sz",  1)}), 1),
      QNSector(QN({QNNameVal("N", 1), QNNameVal("Sz", -1)}), 1),
      QNSector(QN({QNNameVal("N", 0), QNNameVal("Sz",  0)}), 1)}, OUT);
  Index pb_in = InverseIndex(pb_out);

  DGQTensor df =  DGQTensor({pb_in, pb_out});
  DGQTensor dsz = DGQTensor({pb_in, pb_out});
  DGQTensor dsp = DGQTensor({pb_in, pb_out});
  DGQTensor dsm = DGQTensor({pb_in, pb_out});
  DGQTensor dcup =    DGQTensor({pb_in, pb_out});
  DGQTensor dcdagup = DGQTensor({pb_in, pb_out});
  DGQTensor dcdn =    DGQTensor({pb_in, pb_out});
  DGQTensor dcdagdn = DGQTensor({pb_in, pb_out});
  DTenPtrVec dmps = DTenPtrVec(N);

  void SetUp(void) {
    df({0, 0})  = -1;
    df({1, 1})  = -1;
    df({2, 2})  = 1;
    dsz({0, 0}) =  0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;
    dcup({2, 0}) = 1;
    dcdagup({0, 2}) = 1;
    dcdn({2, 1}) = 1;
    dcdagdn({1, 2}) = 1;
  }
};


TEST_F(TestTwoSiteAlgorithmTjSystem, 1DCase) {
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, pb_out, qn0);
  for (long i = 0; i < N-1; ++i) {
    dmpo_gen.AddTerm(-t, {dcdagup, dcup}, {i, i+1}, df);
    dmpo_gen.AddTerm(-t, {dcdagdn, dcdn}, {i, i+1}, df);
    dmpo_gen.AddTerm(-t, {dcup, dcdagup}, {i, i+1}, df);
    dmpo_gen.AddTerm(-t, {dcdn, dcdagdn}, {i, i+1}, df);
    dmpo_gen.AddTerm(J,     {dsz, dsz}, {i, i+1});
    dmpo_gen.AddTerm(0.5*J, {dsp, dsm}, {i, i+1});
    dmpo_gen.AddTerm(0.5*J, {dsm, dsp}, {i, i+1});
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = SweepParams(
                          11,
                          8, 8, 1.0E-9,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-8, 20));
  auto total_div = QN({QNNameVal("N", N-2), QNNameVal("Sz", 0)});
  auto zero_div = QN({QNNameVal("N", 0), QNNameVal("Sz", 0)});
  RandomInitMps(dmps, pb_out, total_div, zero_div, 5);
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -6.947478526233, 1.0E-10);
}


TEST_F(TestTwoSiteAlgorithmTjSystem, 2DCase) {
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, pb_out, qn0);
  std::vector<std::pair<long, long>> nn_pairs = {
      std::make_pair(0, 1), 
      std::make_pair(0, 2), 
      std::make_pair(2, 3), 
      std::make_pair(1, 3)};
  for (auto &p : nn_pairs) {
    dmpo_gen.AddTerm(-t, {dcdagup, dcup}, {p.first, p.second}, df);
    dmpo_gen.AddTerm(-t, {dcdagdn, dcdn}, {p.first, p.second}, df);
    dmpo_gen.AddTerm(-t, {dcup, dcdagup}, {p.first, p.second}, df);
    dmpo_gen.AddTerm(-t, {dcdn, dcdagdn}, {p.first, p.second}, df);
    dmpo_gen.AddTerm(J,     {dsz, dsz}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5*J, {dsp, dsm}, {p.first, p.second});
    dmpo_gen.AddTerm(0.5*J, {dsm, dsp}, {p.first, p.second});
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = SweepParams(
                          10,
                          8, 8, 1.0E-9,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-8, 20));

  auto total_div = QN({QNNameVal("N", N-2), QNNameVal("Sz", 0)});
  auto zero_div = QN({QNNameVal("N", 0), QNNameVal("Sz", 0)});
  RandomInitMps(dmps, pb_out, total_div, zero_div, 5);
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -8.868563739680, 1.0E-10);

  // Direct product state initialization.
  DirectStateInitMps(dmps, {2, 0, 1, 2}, pb_out, zero_div);
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -8.868563739680, 1.0E-10);
}


struct TestTwoSiteAlgorithmHubbardSystem : public testing::Test {
  long Nx = 2;
  long Ny = 2;
  long N = Nx * Ny;
  double t0 = 1.0;
  double t1 = 0.5;
  double U = 2.0;

  QN qn0 = QN({QNNameVal("Nup", 0), QNNameVal("Ndn", 0)});
  Index pb_out = Index({
      QNSector(QN({QNNameVal("Nup", 0), QNNameVal("Ndn", 0)}), 1),
      QNSector(QN({QNNameVal("Nup", 1), QNNameVal("Ndn", 0)}), 1),
      QNSector(QN({QNNameVal("Nup", 0), QNNameVal("Ndn", 1)}), 1),
      QNSector(QN({QNNameVal("Nup", 1), QNNameVal("Ndn", 1)}), 1)}, OUT);
  Index pb_in = InverseIndex(pb_out);

  DGQTensor df       = DGQTensor({pb_in, pb_out});
  DGQTensor dnupdn   = DGQTensor({pb_in, pb_out});    // n_up*n_dn
  DGQTensor dadagupf = DGQTensor({pb_in, pb_out});    // a^+_up*f
  DGQTensor daup     = DGQTensor({pb_in, pb_out});
  DGQTensor dadagdn  = DGQTensor({pb_in, pb_out});
  DGQTensor dfadn    = DGQTensor({pb_in, pb_out});
  DGQTensor dnaupf   = DGQTensor({pb_in, pb_out});    // -a_up*f
  DGQTensor dadagup  = DGQTensor({pb_in, pb_out});
  DGQTensor dnadn    = DGQTensor({pb_in, pb_out});
  DGQTensor dfadagdn = DGQTensor({pb_in, pb_out});    // f*a^+_dn
  DTenPtrVec dmps = DTenPtrVec(N);

  void SetUp(void) {
    df({0, 0})  = 1;
    df({1, 1})  = -1;
    df({2, 2})  = -1;
    df({3, 3})  = 1;

    dnupdn({3, 3}) = 1;

    dadagupf({1, 0}) = 1;
    dadagupf({3, 2}) = -1;
    daup({0, 1}) = 1;
    daup({2, 3}) = 1;
    dadagdn({2, 0}) = 1;
    dadagdn({3, 1}) = 1;
    dfadn({0, 2}) = 1;
    dfadn({1, 3}) = -1;
    dnaupf({0, 1}) = 1;
    dnaupf({2, 3}) = -1;
    dadagup({1, 0}) = 1;
    dadagup({3, 2}) = 1;
    dnadn({0, 2}) = -1;
    dnadn({1, 3}) = -1;
    dfadagdn({2, 0}) = -1;
    dfadagdn({3, 1}) = 1;
  }
};


inline long coors2idx(
    const long x, const long y, const long Nx, const long Ny) {
  return x * Ny + y;
}


inline void KeepOrder(long &x, long &y) {
  if (x > y) {
    auto temp = y;
    y = x;
    x = temp;
  }
}


TEST_F(TestTwoSiteAlgorithmHubbardSystem, 2Dcase) {
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, pb_out, qn0);
  for (long i = 0; i < Nx; ++i) {
    for (long j = 0; j < Ny; ++j) {
      auto s0 = coors2idx(i, j, Nx, Ny);
      dmpo_gen.AddTerm(U, {dnupdn}, {s0});

      if (i != Nx-1) {
        auto s1 = coors2idx(i+1, j, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        dmpo_gen.AddTerm(1, {-t0*dadagupf, daup},  {s0, s1}, df);
        dmpo_gen.AddTerm(1, {-t0*dadagdn, dfadn},  {s0, s1}, df);
        dmpo_gen.AddTerm(1, {dnaupf, -t0*dadagup}, {s0, s1}, df);
        dmpo_gen.AddTerm(1, {dnadn, -t0*dfadagdn}, {s0, s1}, df);
      }
      if (j != Ny-1) {
        auto s1 = coors2idx(i, j+1, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        dmpo_gen.AddTerm(1, {-t0*dadagupf, daup},  {s0, s1}, df);
        dmpo_gen.AddTerm(1, {-t0*dadagdn, dfadn},  {s0, s1}, df);
        dmpo_gen.AddTerm(1, {dnaupf, -t0*dadagup}, {s0, s1}, df);
        dmpo_gen.AddTerm(1, {dnadn, -t0*dfadagdn}, {s0, s1}, df);
      }

      if (j != Ny-1) {
        if (i != 0) {
          auto s2 = coors2idx(i-1, j+1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          dmpo_gen.AddTerm(1, {-t1*dadagupf, daup},  {temp_s0, s2}, df);
          dmpo_gen.AddTerm(1, {-t1*dadagdn, dfadn},  {temp_s0, s2}, df);
          dmpo_gen.AddTerm(1, {dnaupf, -t1*dadagup}, {temp_s0, s2}, df);
          dmpo_gen.AddTerm(1, {dnadn, -t1*dfadagdn}, {temp_s0, s2}, df);
        } 
        if (i != Nx-1) {
          auto s2 = coors2idx(i+1, j+1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          dmpo_gen.AddTerm(1, {-t1*dadagupf, daup},  {temp_s0, s2}, df);
          dmpo_gen.AddTerm(1, {-t1*dadagdn, dfadn},  {temp_s0, s2}, df);
          dmpo_gen.AddTerm(1, {dnaupf, -t1*dadagup}, {temp_s0, s2}, df);
          dmpo_gen.AddTerm(1, {dnadn, -t1*dfadagdn}, {temp_s0, s2}, df);
        } 
      }
    }
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = SweepParams(
                          10,
                          16, 16, 1.0E-9,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-8, 20));
  auto qn0 = QN({QNNameVal("Nup", 0), QNNameVal("Ndn", 0)});
  std::vector<long> stat_labs(N);
  for (int i = 0; i < N; ++i) { stat_labs[i] = (i % 2 == 0 ? 1 : 2); }
  DirectStateInitMps(dmps, stat_labs, pb_out, qn0);
  RunTestTwoSiteAlgorithmCase(
      dmps, dmpo, sweep_params,
      -2.828427124746, 1.0E-10);
}
