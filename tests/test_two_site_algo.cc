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
using ZTenPtrVec = std::vector<ZGQTensor *>;


template <typename TenType>
void RunTestTwoSiteAlgorithmCase(
    std::vector<TenType *> &mps, const std::vector<TenType *> &mpo,
    const SweepParams &sweep_params,
    const double benmrk_e0, const double precision) {
  auto e0 = TwoSiteAlgorithm(mps, mpo, sweep_params);
  EXPECT_NEAR(e0, benmrk_e0, precision);
}


// Helpers
inline void KeepOrder(long &x, long &y) {
  if (x > y) {
    auto temp = y;
    y = x;
    x = temp;
  }
}


inline long coors2idx(
    const long x, const long y, const long Nx, const long Ny) {
	return x * Ny + y;
}



inline long coors2idxSquare(
    const long x, const long y, const long Nx, const long Ny) {
  return x * Ny + y;
}


inline long coors2idxHoneycomb(
    const long x, const long y, const long Nx, const long Ny) {
  return Ny * (x%Nx) + y%Ny;
}


// Test spin systems
struct TestTwoSiteAlgorithmSpinSystem : public testing::Test {
  long N = 6;

  QN qn0 = QN({QNNameVal("Sz", 0)});
  Index pb_out = Index({
                     QNSector(QN({QNNameVal("Sz", 1)}), 1),
                     QNSector(QN({QNNameVal("Sz", -1)}), 1)}, OUT);
  Index pb_in = InverseIndex(pb_out);

  DGQTensor  dsz  = DGQTensor({pb_in, pb_out});
  DGQTensor  dsp  = DGQTensor({pb_in, pb_out});
  DGQTensor  dsm  = DGQTensor({pb_in, pb_out});
  DTenPtrVec dmps = DTenPtrVec(N);

  ZGQTensor  zsz  = ZGQTensor({pb_in, pb_out});
  ZGQTensor  zsp  = ZGQTensor({pb_in, pb_out});
  ZGQTensor  zsm  = ZGQTensor({pb_in, pb_out});
  ZTenPtrVec zmps = ZTenPtrVec(N);

  void SetUp(void) {
    dsz({0, 0}) = 0.5;
    dsz({1, 1}) = -0.5;
    dsp({0, 1}) = 1;
    dsm({1, 0}) = 1;

    zsz({0, 0}) = 0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;
  }
};


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

  // Complex Hamiltonian.
  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, pb_out, qn0);
  for (long i = 0; i < N-1; ++i) {
    zmpo_gen.AddTerm(1, {zsz, zsz}, {i, i+1});
  }
  auto zmpo = zmpo_gen.Gen();
  sweep_params = SweepParams(
                     4,
                     1, 10, 1.0E-5,
                     true,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7));
  RandomInitMps(zmps, pb_out, qn0, qn0, 2);
  RunTestTwoSiteAlgorithmCase(zmps, zmpo, sweep_params, -0.25*(N-1), 1.0E-10);
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

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, pb_out, qn0);
  for (long i = 0; i < N-1; ++i) {
    zmpo_gen.AddTerm(1,   {zsz, zsz}, {i, i+1});
    zmpo_gen.AddTerm(0.5, {zsp, zsm}, {i, i+1});
    zmpo_gen.AddTerm(0.5, {zsm, zsp}, {i, i+1});
  }
  auto zmpo = zmpo_gen.Gen();

  sweep_params = SweepParams(
                     4,
                     8, 8, 1.0E-9,
                     true,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7));
  RandomInitMps(zmps, pb_out, qn0, qn0, 4);
  RunTestTwoSiteAlgorithmCase(
      zmps, zmpo, sweep_params,
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

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, pb_out, qn0);
  for (auto &p : nn_pairs) {
    zmpo_gen.AddTerm(1,   {zsz, zsz}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zsp, zsm}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5, {zsm, zsp}, {p.first, p.second});
  }
  auto zmpo = zmpo_gen.Gen();

  sweep_params = SweepParams(
                     4,
                     8, 8, 1.0E-9,
                     true,
                     kTwoSiteAlgoWorkflowInitial,
                     LanczosParams(1.0E-7));
  DirectStateInitMps(zmps, stat_labs, pb_out, qn0);
  RunTestTwoSiteAlgorithmCase(
      zmps, zmpo, sweep_params,
      -3.129385241572, 1.0E-12);
}


TEST_F(TestTwoSiteAlgorithmSpinSystem, 2DKitaevSimpleCase) {
  long Nx = 4;
  long Ny = 2;
  long N1 = Nx*Ny;
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N1, pb_out, qn0);
  for (long x = 0; x < Nx; ++x) {
    for (long y = 0; y < Ny; ++y) {
      if (x % 2 == 1) {
        auto s0 = coors2idxHoneycomb(x, y, Nx, Ny);
        auto s1 = coors2idxHoneycomb(x-1, y+1, Nx, Ny);
        KeepOrder(s0, s1);
        dmpo_gen.AddTerm(1, {dsz, dsz}, {s0, s1});
      }
    }
  }
  auto dmpo = dmpo_gen.Gen();

  auto sweep_params = SweepParams(
                          4,
                          8, 8, 1.0E-4,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-10));
  // Test extend direct product state random initialization.
  std::vector<long> stat_labs1, stat_labs2;
  for (long i = 0; i < N1; ++i) {
    stat_labs1.push_back(i%2);
    stat_labs2.push_back((i+1)%2);
  }
  auto dmps_8sites = DTenPtrVec(N1);
  ExtendDirectRandomInitMps(
      dmps_8sites, {stat_labs1, stat_labs2}, pb_out, qn0, 2);
  RunTestTwoSiteAlgorithmCase(
      dmps_8sites, dmpo, sweep_params,
      -1.0, 1.0E-12);

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N1, pb_out, qn0);
  for (long x = 0; x < Nx; ++x) {
    for (long y = 0; y < Ny; ++y) {
      if (x % 2 == 1) {
        auto s0 = coors2idxHoneycomb(x, y, Nx, Ny);
        auto s1 = coors2idxHoneycomb(x-1, y+1, Nx, Ny);
        KeepOrder(s0, s1);
        zmpo_gen.AddTerm(1, {zsz, zsz}, {s0, s1});
      }
    }
  }
  auto zmpo = zmpo_gen.Gen();
  auto zmps_8sites = ZTenPtrVec(N1);
  ExtendDirectRandomInitMps(
      zmps_8sites, {stat_labs1, stat_labs2}, pb_out, qn0, 2);
  RunTestTwoSiteAlgorithmCase(
      zmps_8sites, zmpo, sweep_params,
      -1.0, 1.0E-12);
}


TEST(TestTwoSiteAlgorithmNoSymmetrySpinSystem, 2DKitaevComplexCase) {
	using TenElemType = GQTEN_Complex;
	using Tensor = GQTensor<TenElemType>;
	//-------------Set quantum numbers-----------------
	auto zero_div = QN({QNNameVal("N",0)});
	auto idx_out = Index({QNSector(QN({QNNameVal("N",1)}), 2)}, OUT);
	auto idx_in = InverseIndex(idx_out);
	//--------------Single site operators-----------------
	// define the structure of operators
	auto sz = Tensor({ idx_in, idx_out });
	auto sx = Tensor({ idx_in, idx_out });
	auto sy = Tensor({ idx_in, idx_out });
	auto id = Tensor({ idx_in, idx_out });
	// define the contents of operators
  sz({0, 0}) = GQTEN_Complex(0.5, 0);
  sz({1, 1}) = GQTEN_Complex(-0.5, 0);
  sx({0, 1}) = GQTEN_Complex(0.5, 0);
  sx({1, 0}) = GQTEN_Complex(0.5, 0);
	sy({0, 1}) = GQTEN_Complex(0, -0.5);
	sy({1, 0}) = GQTEN_Complex(0, 0.5);
  id({0, 0}) = GQTEN_Complex(1, 0);
  id({1, 1}) = GQTEN_Complex(1, 0);
	//---------------Generate the MPO-----------------
	double J = -1.0;
	double K = 1.0;
	double Gm = 0.1;
	double h = 0.1;
  long Nx = 3, Ny = 4;
  long N = Nx*Ny;
	auto mpo_gen = MPOGenerator<TenElemType>(N, idx_out, zero_div);
	// H =   J * \Sigma_{<ij>} S_i*S_j
	//       K * \Sigma_{<ij>,c-link} Sc_i*Sc_j
	//		 Gm * \Sigma_{<ij>,c-link} Sa_i*Sb_j + Sb_i*Sa_j
	//     - h * \Sigma_i{S^z}
	for (long x = 0; x < Nx; ++x) {
		for (long y = 0; y < Ny; ++y) {
			// use the configuration '/|\' to traverse the square lattice
			// single site operator
			auto site0_num = coors2idx(x, y, Nx, Ny);
			mpo_gen.AddTerm(-h, sz, site0_num);
			mpo_gen.AddTerm(-h, sx, site0_num);
			mpo_gen.AddTerm(-h, sy, site0_num);
			// the '/' part: x-link
			// note that x and y start from 0
			if (y % 2 == 0) {
				auto site0_num = coors2idx(x, y, Nx, Ny);
				auto site1_num = coors2idx(x, y + 1, Nx, Ny);
				std::cout << site0_num << " " << site1_num << std::endl;
				mpo_gen.AddTerm(J,  {sz, sz}, {site0_num, site1_num});
				mpo_gen.AddTerm(J,  {sx, sx}, {site0_num, site1_num});
				mpo_gen.AddTerm(J,  {sy, sy}, {site0_num, site1_num});
        mpo_gen.AddTerm(K,  {sx, sx}, {site0_num, site1_num});
        mpo_gen.AddTerm(Gm, {sz, sy}, {site0_num, site1_num});
        mpo_gen.AddTerm(Gm, {sy, sz}, {site0_num, site1_num});
			}
			// the '|' part: z-link
			if (y % 2 == 1) {
				auto site0_num = coors2idx(x, y, Nx, Ny);
				auto site1_num = coors2idx(x, (y + 1) % Ny, Nx, Ny);
				KeepOrder(site0_num, site1_num);
				std::cout << site0_num << " " << site1_num << std::endl;
				mpo_gen.AddTerm(J,  {sz, sz}, {site0_num, site1_num});
				mpo_gen.AddTerm(J,  {sx, sx}, {site0_num, site1_num});
				mpo_gen.AddTerm(J,  {sy, sy}, {site0_num, site1_num});
        mpo_gen.AddTerm(K,  {sz, sz}, {site0_num, site1_num});
        mpo_gen.AddTerm(Gm, {sx, sy}, {site0_num, site1_num});
        mpo_gen.AddTerm(Gm, {sy, sx}, {site0_num, site1_num});
			}
			// the '\' part: y-link
			// if (y % 2 == 1) {																	// torus
			if ((y % 2 == 1)&&(x != Nx - 1)) {
				auto site0_num = coors2idx(x, y, Nx, Ny);
				auto site1_num = coors2idx(x + 1, y - 1, Nx, Ny);						// cylinder
				KeepOrder(site0_num, site1_num);
				std::cout << site0_num << " " << site1_num << std::endl;
				mpo_gen.AddTerm(J,  {sz, sz}, {site0_num, site1_num});
				mpo_gen.AddTerm(J,  {sx, sx}, {site0_num, site1_num});
				mpo_gen.AddTerm(J,  {sy, sy}, {site0_num, site1_num});
        mpo_gen.AddTerm(K,  {sy, sy}, {site0_num, site1_num});
        mpo_gen.AddTerm(Gm, {sz, sx}, {site0_num, site1_num});
        mpo_gen.AddTerm(Gm, {sx, sz}, {site0_num, site1_num});
			}
		}
  }
	auto mpo = mpo_gen.Gen();

	std::vector<Tensor*> mps(N);
  std::vector<long> stat_labs(N);
  auto was_up = false;
  for (long i = 0; i < N; ++i) {
    if (was_up) {
      stat_labs[i] = 1;
      was_up = false;
    }
    else if (!was_up) {
      stat_labs[i] = 0;
      was_up = true;
    }
  }
  DirectStateInitMps(mps, stat_labs, idx_out, zero_div);

  auto sweep_params = SweepParams(
                          4,
                          128, 128, 1.0E-4,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-10));
  RunTestTwoSiteAlgorithmCase(mps, mpo, sweep_params, -4.57509167674, 1.0E-10);
}


// Test fermion models.
struct TestTwoSiteAlgorithmTjSystem2U1Symm : public testing::Test {
  long N = 4;
  double t = 3.;
  double J = 1.;
  QN qn0 = QN({QNNameVal("N", 0), QNNameVal("Sz", 0)});
  Index pb_out = Index({
      QNSector(QN({QNNameVal("N", 1), QNNameVal("Sz",  1)}), 1),
      QNSector(QN({QNNameVal("N", 1), QNNameVal("Sz", -1)}), 1),
      QNSector(QN({QNNameVal("N", 0), QNNameVal("Sz",  0)}), 1)}, OUT);
  Index pb_in = InverseIndex(pb_out);

  DGQTensor df      = DGQTensor({pb_in, pb_out});
  DGQTensor dsz     = DGQTensor({pb_in, pb_out});
  DGQTensor dsp     = DGQTensor({pb_in, pb_out});
  DGQTensor dsm     = DGQTensor({pb_in, pb_out});
  DGQTensor dcup    = DGQTensor({pb_in, pb_out});
  DGQTensor dcdagup = DGQTensor({pb_in, pb_out});
  DGQTensor dcdn    = DGQTensor({pb_in, pb_out});
  DGQTensor dcdagdn = DGQTensor({pb_in, pb_out});
  DTenPtrVec dmps   = DTenPtrVec(N);

  ZGQTensor zf      = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsz     = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsp     = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsm     = ZGQTensor({pb_in, pb_out});
  ZGQTensor zcup    = ZGQTensor({pb_in, pb_out});
  ZGQTensor zcdagup = ZGQTensor({pb_in, pb_out});
  ZGQTensor zcdn    = ZGQTensor({pb_in, pb_out});
  ZGQTensor zcdagdn = ZGQTensor({pb_in, pb_out});
  ZTenPtrVec zmps   = ZTenPtrVec(N);

  void SetUp(void) {
    df({0, 0})  = -1;
    df({1, 1})  = -1;
    df({2, 2})  = 1;
    dsz({0, 0}) =  0.5;
    dsz({1, 1}) = -0.5;
    dsp({1, 0}) = 1;
    dsm({0, 1}) = 1;
    dcup({0, 2}) = 1;
    dcdagup({2, 0}) = 1;
    dcdn({1, 2}) = 1;
    dcdagdn({2, 1}) = 1;

    zf({0, 0})  = -1;
    zf({1, 1})  = -1;
    zf({2, 2})  = 1;
    zsz({0, 0}) =  0.5;
    zsz({1, 1}) = -0.5;
    zsp({1, 0}) = 1;
    zsm({0, 1}) = 1;
    zcup({0, 2}) = 1;
    zcdagup({2, 0}) = 1;
    zcdn({1, 2}) = 1;
    zcdagdn({2, 1}) = 1;
  }
};


TEST_F(TestTwoSiteAlgorithmTjSystem2U1Symm, 1DCase) {
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, pb_out, qn0);
  for (long i = 0; i < N-1; ++i) {
    dmpo_gen.AddTerm(-t,    {dcdagup, dcup}, {i, i+1}, df);
    dmpo_gen.AddTerm(-t,    {dcdagdn, dcdn}, {i, i+1}, df);
    dmpo_gen.AddTerm(-t,    {dcup, dcdagup}, {i, i+1}, df);
    dmpo_gen.AddTerm(-t,    {dcdn, dcdagdn}, {i, i+1}, df);
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

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, pb_out, qn0);
  for (long i = 0; i < N-1; ++i) {
    zmpo_gen.AddTerm(-t,    {zcdagup, zcup}, {i, i+1}, zf);
    zmpo_gen.AddTerm(-t,    {zcdagdn, zcdn}, {i, i+1}, zf);
    zmpo_gen.AddTerm(-t,    {zcup, zcdagup}, {i, i+1}, zf);
    zmpo_gen.AddTerm(-t,    {zcdn, zcdagdn}, {i, i+1}, zf);
    zmpo_gen.AddTerm(J,     {zsz, zsz}, {i, i+1});
    zmpo_gen.AddTerm(0.5*J, {zsp, zsm}, {i, i+1});
    zmpo_gen.AddTerm(0.5*J, {zsm, zsp}, {i, i+1});
  }
  auto zmpo = zmpo_gen.Gen();
  RandomInitMps(zmps, pb_out, total_div, zero_div, 5);
  RunTestTwoSiteAlgorithmCase(
      zmps, zmpo, sweep_params,
      -6.947478526233, 1.0E-10);
}


TEST_F(TestTwoSiteAlgorithmTjSystem2U1Symm, 2DCase) {
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

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, pb_out, qn0);
  for (auto &p : nn_pairs) {
    zmpo_gen.AddTerm(-t, {zcdagup, zcup}, {p.first, p.second}, zf);
    zmpo_gen.AddTerm(-t, {zcdagdn, zcdn}, {p.first, p.second}, zf);
    zmpo_gen.AddTerm(-t, {zcup, zcdagup}, {p.first, p.second}, zf);
    zmpo_gen.AddTerm(-t, {zcdn, zcdagdn}, {p.first, p.second}, zf);
    zmpo_gen.AddTerm(J,     {zsz, zsz}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5*J, {zsp, zsm}, {p.first, p.second});
    zmpo_gen.AddTerm(0.5*J, {zsm, zsp}, {p.first, p.second});
  }
  auto zmpo = zmpo_gen.Gen();
  DirectStateInitMps(zmps, {2, 0, 1, 2}, pb_out, zero_div);
  RunTestTwoSiteAlgorithmCase(
      zmps, zmpo, sweep_params,
      -8.868563739680, 1.0E-10);
}


struct TestTwoSiteAlgorithmTjSystem1U1Symm : public testing::Test {
  QN qn0 = QN({QNNameVal("N", 0)});
  Index pb_out = Index({
                     QNSector(QN({QNNameVal("N", 1)}), 2),
                     QNSector(QN({QNNameVal("N", 0)}), 1)
                     }, OUT);
  Index pb_in = InverseIndex(pb_out);
  ZGQTensor zf = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsz = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsp = ZGQTensor({pb_in, pb_out});
  ZGQTensor zsm = ZGQTensor({pb_in, pb_out});
  ZGQTensor zcup = ZGQTensor({pb_in, pb_out});
  ZGQTensor zcdagup = ZGQTensor({pb_in, pb_out});
  ZGQTensor zcdn = ZGQTensor({pb_in, pb_out});
  ZGQTensor zcdagdn = ZGQTensor({pb_in, pb_out});
  ZGQTensor zntot = ZGQTensor({pb_in, pb_out});
  ZGQTensor zid = ZGQTensor({pb_in, pb_out});

  void SetUp(void) {
    zf({0, 0})  = -1;
    zf({1, 1})  = -1;
    zf({2, 2})  = 1;
    zsz({0, 0}) =  0.5;
    zsz({1, 1}) = -0.5;
    zsp({0, 1}) = 1;
    zsm({1, 0}) = 1;
    zcup({2, 0}) = 1;
    zcdagup({0, 2}) = 1;
    zcdn({2, 1}) = 1;
    zcdagdn({1, 2}) = 1;
    zntot({0, 0}) = 1;
    zntot({1, 1}) = 1;
    zid({0,0})=1;
    zid({1,1})=1;
    zid({2,2})=1;
  }
};


TEST_F(TestTwoSiteAlgorithmTjSystem1U1Symm, RashbaTermCase) {
  double t = 3.0;
  double J = 1.0;
  double lamb = 0.03;
  auto ilamb = GQTEN_Complex(0, lamb);
  long Nx = 3;
  long Ny = 2;
  long Ntot = Nx * Ny;
  char BCx = 'p';
  char BCy = 'o';
  auto mpo_gen = MPOGenerator<GQTEN_Complex>(Ntot, pb_out, qn0);
  for (long x = 0; x < Nx; ++x) {
    for (long y = 0; y < Ny; ++y) {

      if (!((BCx == 'o') && (x == Nx-1))) {
        auto s0 = coors2idxSquare(x, y, Nx, Ny);
        auto s1 = coors2idxSquare((x+1)%Nx, y, Nx, Ny);
        KeepOrder(s0, s1);
        std::cout << s0 << " " << s1 << std::endl;
        mpo_gen.AddTerm(-t, {zcdagup, zcup}, {s0, s1}, zf);
        mpo_gen.AddTerm(-t, {zcdagdn, zcdn}, {s0, s1}, zf);
        mpo_gen.AddTerm(-t, {zcup, zcdagup}, {s0, s1}, zf);
        mpo_gen.AddTerm(-t, {zcdn, zcdagdn}, {s0, s1}, zf);
        mpo_gen.AddTerm(J, {zsz, zsz}, {s0, s1});
        mpo_gen.AddTerm(0.5*J, {zsp, zsm}, {s0, s1});
        mpo_gen.AddTerm(0.5*J, {zsm, zsp}, {s0, s1});
        mpo_gen.AddTerm(-0.25*J, {zntot, zntot}, {s0, s1});
        // SO term
        if (x != Nx-1) {
          mpo_gen.AddTerm(lamb, {zcdagup, zcdn}, {s0, s1}, zf);
          mpo_gen.AddTerm(lamb, {zcup, zcdagdn}, {s0, s1}, zf);
          mpo_gen.AddTerm(-lamb, {zcdagdn, zcup}, {s0, s1}, zf);
          mpo_gen.AddTerm(-lamb, {zcdn, zcdagup}, {s0, s1}, zf);
        } else {    // At the boundary
          mpo_gen.AddTerm(lamb, {zcdn, zcdagup}, {s0, s1}, zf);
          mpo_gen.AddTerm(lamb, {zcdagdn, zcup}, {s0, s1}, zf);
          mpo_gen.AddTerm(-lamb, {zcup, zcdagdn}, {s0, s1}, zf);
          mpo_gen.AddTerm(-lamb, {zcdagup, zcdn}, {s0, s1}, zf);
        }
      }

      if (!((BCy == 'o') && (y == Ny-1))) {
        auto s0 = coors2idxSquare(x, y, Nx, Ny);
        auto s2 = coors2idxSquare(x, (y+1)%Ny, Nx, Ny);
        KeepOrder(s0, s2);
        std::cout << s0 << " " << s2 << std::endl;
        mpo_gen.AddTerm(-t, {zcdagup, zcup}, {s0, s2}, zf);
        mpo_gen.AddTerm(-t, {zcdagdn, zcdn}, {s0, s2}, zf);
        mpo_gen.AddTerm(-t, {zcup, zcdagup}, {s0, s2}, zf);
        mpo_gen.AddTerm(-t, {zcdn, zcdagdn}, {s0, s2}, zf);
        mpo_gen.AddTerm(J, {zsz, zsz}, {s0, s2});
        mpo_gen.AddTerm(0.5*J, {zsp, zsm}, {s0, s2});
        mpo_gen.AddTerm(0.5*J, {zsm, zsp}, {s0, s2});
        mpo_gen.AddTerm(-0.25*J, {zntot, zntot}, {s0, s2});
        if (y != Ny-1) {
          mpo_gen.AddTerm(ilamb,  {zcdagup, zcdn}, {s0, s2}, zf);
          mpo_gen.AddTerm(ilamb,  {zcdagdn, zcup}, {s0, s2}, zf);
          mpo_gen.AddTerm(-ilamb, {zcup, zcdagdn}, {s0, s2}, zf);
          mpo_gen.AddTerm(-ilamb, {zcdn, zcdagup}, {s0, s2}, zf);
        } else {    // At the boundary
          mpo_gen.AddTerm(ilamb, {zcup, zcdagdn}, {s0, s2}, zf);
          mpo_gen.AddTerm(ilamb, {zcdn, zcdagup}, {s0, s2}, zf);
          mpo_gen.AddTerm(-ilamb, {zcdagup, zcdn}, {s0, s2}, zf);
          mpo_gen.AddTerm(-ilamb, {zcdagdn, zcup}, {s0, s2}, zf);
        }
      }
    }
  }
  auto mpo = mpo_gen.Gen();
  auto mps = ZTenPtrVec(Ntot);
  DirectStateInitMps(mps, {0, 1, 0, 2, 0, 1}, pb_out, qn0);
  auto sweep_params = SweepParams(
                          8,
                          30, 30, 1.0E-4,
                          true,
                          kTwoSiteAlgoWorkflowInitial,
                          LanczosParams(1.0E-14, 100));
  RunTestTwoSiteAlgorithmCase(
      mps, mpo, sweep_params,
      -11.018692166942165, 1.0E-10);
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
  DTenPtrVec dmps    = DTenPtrVec(N);

  ZGQTensor zf       = ZGQTensor({pb_in, pb_out});
  ZGQTensor znupdn   = ZGQTensor({pb_in, pb_out});    // n_up*n_dn
  ZGQTensor zadagupf = ZGQTensor({pb_in, pb_out});    // a^+_up*f
  ZGQTensor zaup     = ZGQTensor({pb_in, pb_out});
  ZGQTensor zadagdn  = ZGQTensor({pb_in, pb_out});
  ZGQTensor zfadn    = ZGQTensor({pb_in, pb_out});
  ZGQTensor znaupf   = ZGQTensor({pb_in, pb_out});    // -a_up*f
  ZGQTensor zadagup  = ZGQTensor({pb_in, pb_out});
  ZGQTensor znadn    = ZGQTensor({pb_in, pb_out});
  ZGQTensor zfadagdn = ZGQTensor({pb_in, pb_out});    // f*a^+_dn
  ZTenPtrVec zmps    = ZTenPtrVec(N);

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

    zf({0, 0})  = 1;
    zf({1, 1})  = -1;
    zf({2, 2})  = -1;
    zf({3, 3})  = 1;

    znupdn({3, 3}) = 1;

    zadagupf({1, 0}) = 1;
    zadagupf({3, 2}) = -1;
    zaup({0, 1}) = 1;
    zaup({2, 3}) = 1;
    zadagdn({2, 0}) = 1;
    zadagdn({3, 1}) = 1;
    zfadn({0, 2}) = 1;
    zfadn({1, 3}) = -1;
    znaupf({0, 1}) = 1;
    znaupf({2, 3}) = -1;
    zadagup({1, 0}) = 1;
    zadagup({3, 2}) = 1;
    znadn({0, 2}) = -1;
    znadn({1, 3}) = -1;
    zfadagdn({2, 0}) = -1;
    zfadagdn({3, 1}) = 1;
  }
};


TEST_F(TestTwoSiteAlgorithmHubbardSystem, 2Dcase) {
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(N, pb_out, qn0);
  for (long i = 0; i < Nx; ++i) {
    for (long j = 0; j < Ny; ++j) {
      auto s0 = coors2idxSquare(i, j, Nx, Ny);
      dmpo_gen.AddTerm(U, dnupdn, s0);

      if (i != Nx-1) {
        auto s1 = coors2idxSquare(i+1, j, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        dmpo_gen.AddTerm(1, {-t0*dadagupf, daup},  {s0, s1}, df);
        dmpo_gen.AddTerm(1, {-t0*dadagdn, dfadn},  {s0, s1}, df);
        dmpo_gen.AddTerm(1, {dnaupf, -t0*dadagup}, {s0, s1}, df);
        dmpo_gen.AddTerm(1, {dnadn, -t0*dfadagdn}, {s0, s1}, df);
      }
      if (j != Ny-1) {
        auto s1 = coors2idxSquare(i, j+1, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        dmpo_gen.AddTerm(1, {-t0*dadagupf, daup},  {s0, s1}, df);
        dmpo_gen.AddTerm(1, {-t0*dadagdn, dfadn},  {s0, s1}, df);
        dmpo_gen.AddTerm(1, {dnaupf, -t0*dadagup}, {s0, s1}, df);
        dmpo_gen.AddTerm(1, {dnadn, -t0*dfadagdn}, {s0, s1}, df);
      }

      if (j != Ny-1) {
        if (i != 0) {
          auto s2 = coors2idxSquare(i-1, j+1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          dmpo_gen.AddTerm(1, {-t1*dadagupf, daup},  {temp_s0, s2}, df);
          dmpo_gen.AddTerm(1, {-t1*dadagdn, dfadn},  {temp_s0, s2}, df);
          dmpo_gen.AddTerm(1, {dnaupf, -t1*dadagup}, {temp_s0, s2}, df);
          dmpo_gen.AddTerm(1, {dnadn, -t1*dfadagdn}, {temp_s0, s2}, df);
        } 
        if (i != Nx-1) {
          auto s2 = coors2idxSquare(i+1, j+1, Nx, Ny);
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

  // Complex Hamiltonian
  auto zmpo_gen = MPOGenerator<GQTEN_Complex>(N, pb_out, qn0);
  for (long i = 0; i < Nx; ++i) {
    for (long j = 0; j < Ny; ++j) {
      auto s0 = coors2idxSquare(i, j, Nx, Ny);
      zmpo_gen.AddTerm(U, znupdn, s0);

      if (i != Nx-1) {
        auto s1 = coors2idxSquare(i+1, j, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        zmpo_gen.AddTerm(-t0, {zadagupf, zaup},  {s0, s1}, zf);
        zmpo_gen.AddTerm(-t0, {zadagdn, zfadn},  {s0, s1}, zf);
        zmpo_gen.AddTerm(-t0, {znaupf, zadagup}, {s0, s1}, zf);
        zmpo_gen.AddTerm(-t0, {znadn, zfadagdn}, {s0, s1}, zf);
      }
      if (j != Ny-1) {
        auto s1 = coors2idxSquare(i, j+1, Nx, Ny);
        std::cout << s0 << " " << s1 << std::endl;
        zmpo_gen.AddTerm(-t0, {zadagupf, zaup},  {s0, s1}, zf);
        zmpo_gen.AddTerm(-t0, {zadagdn, zfadn},  {s0, s1}, zf);
        zmpo_gen.AddTerm(-t0, {znaupf, zadagup}, {s0, s1}, zf);
        zmpo_gen.AddTerm(-t0, {znadn, zfadagdn}, {s0, s1}, zf);
      }

      if (j != Ny-1) {
        if (i != 0) {
          auto s2 = coors2idxSquare(i-1, j+1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          zmpo_gen.AddTerm(-t1, {zadagupf, zaup},  {temp_s0, s2}, zf);
          zmpo_gen.AddTerm(-t1, {zadagdn, zfadn},  {temp_s0, s2}, zf);
          zmpo_gen.AddTerm(-t1, {znaupf, zadagup}, {temp_s0, s2}, zf);
          zmpo_gen.AddTerm(-t1, {znadn, zfadagdn}, {temp_s0, s2}, zf);
        } 
        if (i != Nx-1) {
          auto s2 = coors2idxSquare(i+1, j+1, Nx, Ny);
          auto temp_s0 = s0;
          KeepOrder(temp_s0, s2);
          std::cout << temp_s0 << " " << s2 << std::endl;
          zmpo_gen.AddTerm(-t1, {zadagupf, zaup},  {temp_s0, s2}, zf);
          zmpo_gen.AddTerm(-t1, {zadagdn, zfadn},  {temp_s0, s2}, zf);
          zmpo_gen.AddTerm(-t1, {znaupf, zadagup}, {temp_s0, s2}, zf);
          zmpo_gen.AddTerm(-t1, {znadn, zfadagdn}, {temp_s0, s2}, zf);
        } 
      }
    }
  }
  auto zmpo = zmpo_gen.Gen();
  DirectStateInitMps(zmps, stat_labs, pb_out, qn0);
  RunTestTwoSiteAlgorithmCase(
      zmps, zmpo, sweep_params,
      -2.828427124746, 1.0E-10);
}


///Ref: 10.1103/PhysRevB.97.245119
struct TestKondoInsulatorSystem : public testing::Test {
  long N = 4;
  double t = 0.25;
  double Jk = 1.0;
  double Jz = 0.5;

  QN qn0 = QN({QNNameVal("Sz", 0)});
  Index pb_outE = Index({QNSector(QN({ QNNameVal("Sz",  0)}), 4)},OUT );//extended electron
  Index pb_inE = InverseIndex(pb_outE);
  Index pb_outL = Index({QNSector(QN({ QNNameVal("Sz",  0)}), 2)},OUT );//localized electron
  Index pb_inL = InverseIndex(pb_outL);
  DGQTensor sz = DGQTensor({pb_inE, pb_outE});
  DGQTensor sp = DGQTensor({pb_inE, pb_outE});
  DGQTensor sm = DGQTensor({pb_inE, pb_outE});
  DGQTensor bupcF =DGQTensor({pb_inE,pb_outE});
  DGQTensor bupaF = DGQTensor({pb_inE,pb_outE});
  DGQTensor Fbdnc = DGQTensor({pb_inE,pb_outE});
  DGQTensor Fbdna = DGQTensor({pb_inE,pb_outE});
  DGQTensor bupc =DGQTensor({pb_inE,pb_outE});
  DGQTensor bupa = DGQTensor({pb_inE,pb_outE});
  DGQTensor bdnc = DGQTensor({pb_inE,pb_outE});
  DGQTensor bdna = DGQTensor({pb_inE,pb_outE});


  DGQTensor Sz =DGQTensor({pb_inL, pb_outL});
  DGQTensor Sp =DGQTensor({pb_inL, pb_outL});
  DGQTensor Sm =DGQTensor({pb_inL, pb_outL});
  DTenPtrVec dmps    = DTenPtrVec(2*N);
  std::vector<Index> pb_set = std::vector<Index>(2*N);

  void SetUp(void) {
    sz({0,0}) = 0.5;  sz({1,1}) = -0.5;
    sp({0,1}) = 1.0;
    sm({1,0}) = 1.0;
    bupcF({2,1}) = -1;  bupcF({0,3}) = 1;
    Fbdnc({2,0}) = 1;   Fbdnc({1,3}) = -1;
    bupaF({1,2}) = 1;   bupaF({3,0}) = -1;
    Fbdna({0,2}) = -1;  Fbdna({3,1}) = 1;

    bupc({2,1}) = 1;  bupc({0,3}) = 1;
    bdnc({2,0}) = 1;  bdnc({1,3}) = 1;
    bupa({1,2}) = 1;  bupa({3,0}) = 1;
    bdna({0,2}) = 1;  bdna({3,1}) = 1;

    Sz({0,0}) = 0.5;  Sz({1,1}) = -0.5;
    Sp({0,1}) = 1.0;
    Sm({1,0}) = 1.0;
    for(long i =0;i < 2*N; ++i){
      if(i%2==0) pb_set[i] = pb_outE; // even site is extended electron
      if(i%2==1) pb_set[i] = pb_outL; // odd site is localized electron
    }
  }
};

TEST_F(TestKondoInsulatorSystem, doublechain) {
  auto dmpo_gen = MPOGenerator<GQTEN_Double>(pb_set, qn0);
  for (long i = 0; i < 2*N-2; i=i+2){
    dmpo_gen.AddTerm( -t, {bupcF,bupa},{i,i+2});
    dmpo_gen.AddTerm( -t, {bdnc,Fbdna},{i,i+2});
    dmpo_gen.AddTerm(  t, {bupaF,bupc},{i,i+2});
    dmpo_gen.AddTerm(  t, {bdna,Fbdnc},{i,i+2});
    dmpo_gen.AddTerm( Jz , {Sz,Sz}, {i+1,i+3});
  }
  for (long i = 0; i < 2*N; i=i+2){
    dmpo_gen.AddTerm( Jk, {sz,Sz},{i, i+1} );
    dmpo_gen.AddTerm(Jk/2,{sp,Sm},{i, i+1} );
    dmpo_gen.AddTerm(Jk/2,{sm,Sp},{i, i+1} );
  }
  auto dmpo = dmpo_gen.Gen();
  auto sweep_params = SweepParams(5,64, 64, 1.0E-9,true,//Sweep, Dmin, Dmax,Cutoff, FileIO
    kTwoSiteAlgoWorkflowInitial, //mode
    LanczosParams(1.0E-8, 20));  //LanczosParams

    std::vector<long> stat_labs(2*N,0);
    DirectStateInitMps(dmps, stat_labs, pb_set, qn0);
    ///Benchmark with ED's results
    RunTestTwoSiteAlgorithmCase(dmps, dmpo, sweep_params,-3.180025784229132, 1.0E-10);
}

