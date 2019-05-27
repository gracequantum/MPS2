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


struct TestTwoSiteAlgorithmSpinSystem : public testing::Test {
  Index phys_idx_out = Index({
                           QNSector(QN({QNNameVal("Sz", 1)}), 1),
                           QNSector(QN({QNNameVal("Sz", -1)}), 1)}, OUT);
  Index phys_idx_in = InverseIndex(phys_idx_out);
  GQTensor sz = GQTensor({phys_idx_in, phys_idx_out});
  GQTensor sp = GQTensor({phys_idx_in, phys_idx_out});
  GQTensor sm = GQTensor({phys_idx_in, phys_idx_out});

  long init_vdim = 1;
  Index mps_init_virt_idx_in =
      Index({
          QNSector(QN({QNNameVal("Sz", 1)}), init_vdim),
          QNSector(QN({QNNameVal("Sz", 0)}), init_vdim),
          QNSector(QN({QNNameVal("Sz", -1)}), init_vdim)}, IN);
  Index mps_init_virt_idx_out = InverseIndex(mps_init_virt_idx_in);

  void SetUp(void) {
    sz({0, 0}) = 0.5;
    sz({1, 1}) = -0.5;
    sp({0, 1}) = 1;
    sm({1, 0}) = 1;
  }
};


TEST_F(TestTwoSiteAlgorithmSpinSystem, Cases) {
  long N = 6;
  
  // 1D Ising model.
  auto mpo_gen = MPOGenerator(N, phys_idx_out);
  for (long i = 0; i < N-1; ++i) {
    mpo_gen.AddTerm(1, {OpIdx(sz, i), OpIdx(sz, i+1)});
  }
  auto mpo = mpo_gen.Gen();
  std::vector<GQTensor *> mps(N);
  auto qn0 = QN({QNNameVal("Sz", 0)});
  srand(0);
  RandomInitMps(mps, phys_idx_out, qn0, qn0);
  auto sweep_params = SweepParams(
                          4,
                          1, 10, 1.0E-5,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-7),
                          N/2-1);
  auto energy0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(energy0, -0.25*(N-1), 1.0E-12);
  // No file I/O case.
  sweep_params = SweepParams(
                     4,
                     1, 10, 1.0E-5,
                     false,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7),
                     N/2-1);
  energy0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(energy0, -0.25*(N-1), 1.0E-12);

  // 1D AFM Heisenberg model.
  mpo_gen = MPOGenerator(N, phys_idx_out);
  for (long i = 0; i < N-1; ++i) {
    mpo_gen.AddTerm(1, {OpIdx(sz, i), OpIdx(sz, i+1)});
    mpo_gen.AddTerm(0.5, {OpIdx(sp, i), OpIdx(sm, i+1)});
    mpo_gen.AddTerm(0.5, {OpIdx(sm, i), OpIdx(sp, i+1)});
  }
  mpo = mpo_gen.Gen();
  sweep_params = SweepParams(
                     4,
                     8, 8, 1.0E-9,
                     true,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7),
                     N/2-1);
  energy0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(energy0, -2.493577133888, 1.0E-12);

  // Restart test.
  DumpMps(mps);
  sweep_params = SweepParams(
                     4,
                     8, 8, 1.0E-9,
                     true,
                     kTwoSiteAlgoWorkflowContinue,
                     LanczosParams(1.0E-7),
                     N/2-1);
  for (auto &mps_ten : mps) { delete mps_ten; }
  LoadMps(mps);
  energy0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(energy0, -2.493577133888, 1.0E-12);

  // 2D AFM Heisenberg model.
  mpo_gen = MPOGenerator(N, phys_idx_out);
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
    mpo_gen.AddTerm(1, {OpIdx(sz, p.first), OpIdx(sz, p.second)});
    mpo_gen.AddTerm(0.5, {OpIdx(sp, p.first), OpIdx(sm, p.second)});
    mpo_gen.AddTerm(0.5, {OpIdx(sm, p.first), OpIdx(sp, p.second)});
  }
  mpo = mpo_gen.Gen();
  energy0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(energy0, -3.129385241572, 1.0E-12);

  // Test direct product state initialization.
  std::vector<long> stat_labs;
  for (int i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  DirectStateInitMps(mps, stat_labs, phys_idx_out, qn0);
  sweep_params = SweepParams(
                     5,
                     8, 8, 1.0E-9,
                     true,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7),
                     N/2-1);
  energy0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(energy0, -3.129385241572, 1.0E-12);
}


// Test Fermion model.
struct TestTwoSiteAlgorithmFermionSystem : public testing::Test {
  long N = 4;
  double t = 3.;
  double J = 1.;
  Index phys_idx_out = Index({
      QNSector(QN({QNNameVal("N", 1), QNNameVal("Sz",  1)}), 1),
      QNSector(QN({QNNameVal("N", 1), QNNameVal("Sz", -1)}), 1),
      QNSector(QN({QNNameVal("N", 0), QNNameVal("Sz",  0)}), 1)}, OUT);
  Index phys_idx_in = InverseIndex(phys_idx_out);
  GQTensor f = GQTensor({phys_idx_in, phys_idx_out});
  GQTensor sz = GQTensor({phys_idx_in, phys_idx_out});
  GQTensor sp = GQTensor({phys_idx_in, phys_idx_out});
  GQTensor sm = GQTensor({phys_idx_in, phys_idx_out});
  GQTensor cup = GQTensor({phys_idx_in, phys_idx_out});
  GQTensor cdagup = GQTensor({phys_idx_in, phys_idx_out});
  GQTensor cdn = GQTensor({phys_idx_in, phys_idx_out});
  GQTensor cdagdn = GQTensor({phys_idx_in, phys_idx_out});

  void SetUp(void) {
    f({0, 0})  = -1;
    f({1, 1})  = -1;
    f({2, 2})  = 1;
    sz({0, 0}) =  0.5;
    sz({1, 1}) = -0.5;
    sp({0, 1}) = 1;
    sm({1, 0}) = 1;
    cup({2, 0}) = 1;
    cdagup({0, 2}) = 1;
    cdn({2, 1}) = 1;
    cdagdn({1, 2}) = 1;
  }
};


TEST_F(TestTwoSiteAlgorithmFermionSystem, Cases) {
  // 1D t-J chain.
  auto mpo_gen = MPOGenerator(N, phys_idx_out);
  for (long i = 0; i < N-1; ++i) {
    mpo_gen.AddTerm(-t, {OpIdx(cdagup, i), OpIdx(cup, i+1)});
    mpo_gen.AddTerm(-t, {OpIdx(cdagdn, i), OpIdx(cdn, i+1)});
    mpo_gen.AddTerm(-t, {OpIdx(cup, i), OpIdx(cdagup, i+1)});
    mpo_gen.AddTerm(-t, {OpIdx(cdn, i), OpIdx(cdagdn, i+1)});
    mpo_gen.AddTerm(J, {OpIdx(sz, i), OpIdx(sz, i+1)});
    mpo_gen.AddTerm(0.5*J, {OpIdx(sp, i), OpIdx(sm, i+1)});
    mpo_gen.AddTerm(0.5*J, {OpIdx(sm, i), OpIdx(sp, i+1)});
  }
  auto mpo = mpo_gen.Gen();

  std::vector<GQTensor *> mps(N);
  auto total_div = QN({QNNameVal("N", N-2), QNNameVal("Sz", 0)});
  auto zero_div = QN({QNNameVal("N", 0), QNNameVal("Sz", 0)});
  srand(0);
  RandomInitMps(mps, phys_idx_out, total_div, zero_div);
  auto sweep_params = SweepParams(
                          12,
                          8, 8, 1.0E-9,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-8, 20),
                          N/2-1);
  auto energy0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(energy0, -6.947478526233, 1.0E-10);

  // 2D t-J model.
  mpo_gen = MPOGenerator(N, phys_idx_out);
  std::vector<std::pair<long, long>> nn_pairs = {
      std::make_pair(0, 1), 
      std::make_pair(0, 2), 
      std::make_pair(2, 3), 
      std::make_pair(1, 3)};
  for (auto &p : nn_pairs) {
    mpo_gen.AddTerm(-t, {OpIdx(cdagup, p.first), OpIdx(cup, p.second)}, f);
    mpo_gen.AddTerm(-t, {OpIdx(cdagdn, p.first), OpIdx(cdn, p.second)}, f);
    mpo_gen.AddTerm(-t, {OpIdx(cup, p.first), OpIdx(cdagup, p.second)}, f);
    mpo_gen.AddTerm(-t, {OpIdx(cdn, p.first), OpIdx(cdagdn, p.second)}, f);
    mpo_gen.AddTerm(J, {OpIdx(sz, p.first), OpIdx(sz, p.second)});
    mpo_gen.AddTerm(0.5*J, {OpIdx(sp, p.first), OpIdx(sm, p.second)});
    mpo_gen.AddTerm(0.5*J, {OpIdx(sm, p.first), OpIdx(sp, p.second)});
  }
  mpo = mpo_gen.Gen();
  energy0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(energy0, -8.868563739680, 1.0E-10);
  // Direct product state initialization.
  DirectStateInitMps(mps, {2, 0, 1, 2}, phys_idx_out, zero_div);
  energy0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(energy0, -8.868563739680, 1.0E-10);
}
