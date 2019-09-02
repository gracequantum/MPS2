// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-07-01 15:38
* 
* Description: GraceQ/mps2 project. MPI parallel two site algorithm implementation unittests.
*/
#include "gqmps2/gqmps2.h"
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <vector>

#include "mpi.h"


using namespace gqmps2;
using namespace gqten;


const int kGemmBatchWorkers =  3;


struct TestDistributedTwoSiteAlgorithmTjSystem : public testing::Test {
  long N = 4;
  double t = 3.;
  double J = 1.;
  QN qn0 = QN({QNNameVal("N", 0), QNNameVal("Sz", 0)});
  Index pb_out = Index({
      QNSector(QN({QNNameVal("N", 1), QNNameVal("Sz",  1)}), 1),
      QNSector(QN({QNNameVal("N", 1), QNNameVal("Sz", -1)}), 1),
      QNSector(QN({QNNameVal("N", 0), QNNameVal("Sz",  0)}), 1)}, OUT);
  Index pb_in = InverseIndex(pb_out);
  GQTensor f = GQTensor({pb_in, pb_out});
  GQTensor sz = GQTensor({pb_in, pb_out});
  GQTensor sp = GQTensor({pb_in, pb_out});
  GQTensor sm = GQTensor({pb_in, pb_out});
  GQTensor cup = GQTensor({pb_in, pb_out});
  GQTensor cdagup = GQTensor({pb_in, pb_out});
  GQTensor cdn = GQTensor({pb_in, pb_out});
  GQTensor cdagdn = GQTensor({pb_in, pb_out});

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


TEST_F(TestDistributedTwoSiteAlgorithmTjSystem, 1DCase) {
  auto mpo_gen = MPOGenerator(N, pb_out, qn0);
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
  RandomInitMps(mps, pb_out, total_div, zero_div, 5);

  auto sweep_params = SweepParams(
                          11,
                          8, 8, 1.0E-9,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-8, 20));

  auto energy0 = GQMPS2_MPI_TwoSiteAlgorithm(
                     mps, mpo, sweep_params,
                     MPI_COMM_WORLD, kGemmBatchWorkers);
  EXPECT_NEAR(energy0, -6.947478526233, 1.0E-10);
}


int main(int argc, char *argv[]) {
  int result = 0;
  testing::InitGoogleTest(&argc, argv); 
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  result = RUN_ALL_TESTS();
  for (int i = 1; i <= kGemmBatchWorkers; ++i) {
    MPI_SendGemmWorkerStat(kGemmWorkerStatStop, i, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return result;
}
