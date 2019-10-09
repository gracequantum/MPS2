// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-10-08 09:38
* 
* Description: GraceQ/MPS2 project. Unittest for MPS measurements.
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include "gtest/gtest.h"


using namespace gqmps2;
using namespace gqten;
using DTenPtrVec = std::vector<DGQTensor *>;
using ZTenPtrVec = std::vector<ZGQTensor *>;


inline void ExpectDoubleEq(const double lhs, const double rhs) {
  EXPECT_DOUBLE_EQ(lhs, rhs);
}


inline void ExpectDoubleEq(const GQTEN_Complex lhs, const GQTEN_Complex rhs) {
  EXPECT_DOUBLE_EQ(lhs.real(), rhs.real());
  EXPECT_DOUBLE_EQ(lhs.imag(), rhs.imag());
}


struct TestMpsMeasurement : public testing::Test {
  long N = 6;

  QN qn0 = QN({QNNameVal("N", 0)});
  Index pb_out = Index({
                     QNSector(QN({QNNameVal("N", 0)}), 1),
                     QNSector(QN({QNNameVal("N", 1)}), 1)}, OUT);
  Index pb_in = InverseIndex(pb_out);

  DGQTensor dntot = DGQTensor({pb_in, pb_out});
  ZGQTensor zntot = ZGQTensor({pb_in, pb_out});
  DTenPtrVec dmps = DTenPtrVec(N);
  ZTenPtrVec zmps = ZTenPtrVec(N);

  void SetUp(void) {
    dntot({0, 0}) = 0;
    dntot({1, 1}) = 1;
    zntot({0, 0}) = 0;
    zntot({1, 1}) = 1;
  }
};


template <typename MpsType, typename TenElemType>
void RunTestMeasureOneSiteOpCase(
    MpsType &mps,
    const GQTensor<TenElemType> &op, const std::vector<TenElemType> &res) {
  auto measu_res = MeasureOneSiteOp(mps, op, "op1");
  assert(measu_res.size() == res.size());
  for (size_t i = 0; i < res.size(); ++i) {
    ExpectDoubleEq(measu_res[i].avg, res[i]);
  }
}


TEST_F(TestMpsMeasurement, TestMeasureOneSiteOp) {
  // Double case 1
  std::vector<long> stat_labs1;
  std::vector<GQTEN_Double> dres1;
  for (long i = 0; i < N; ++i) {
    stat_labs1.push_back(1);
    dres1.push_back(1.0);
  }
  auto dmps1 = dmps;
  DirectStateInitMps(dmps1, stat_labs1, pb_out, qn0);
  auto dmps_for_measu1 = MPS<DGQTensor>(dmps1, -1); 
  RunTestMeasureOneSiteOpCase(dmps_for_measu1, dntot, dres1);
  // Double case 2
  std::vector<long> stat_labs2;
  std::vector<GQTEN_Double> dres2;
  for (long i = 0; i < N; ++i) {
    auto lab = i % 2;
    stat_labs2.push_back(lab);
    dres2.push_back(lab);
  }
  auto dmps2 = dmps;
  DirectStateInitMps(dmps2, stat_labs2, pb_out, qn0);
  auto dmps_for_measu2 = MPS<DGQTensor>(dmps2, -1); 
  RunTestMeasureOneSiteOpCase(dmps_for_measu2, dntot, dres2);
  // Complex case 1
  std::vector<GQTEN_Complex> zres1;
  for (auto d : dres1) { zres1.push_back(d); }
  auto zmps1 = zmps;
  DirectStateInitMps(zmps1, stat_labs1, pb_out, qn0);
  auto zmps_for_measu1 = MPS<ZGQTensor>(zmps1, -1); 
  RunTestMeasureOneSiteOpCase(zmps_for_measu1, zntot, zres1);
  // Complex case 2
  std::vector<GQTEN_Complex> zres2;
  for (auto d : dres2) { zres2.push_back(d); }
  auto zmps2 = zmps;
  DirectStateInitMps(zmps2, stat_labs2, pb_out, qn0);
  auto zmps_for_measu2 = MPS<ZGQTensor>(zmps2, -1); 
  RunTestMeasureOneSiteOpCase(zmps_for_measu2, zntot, zres2);
}
