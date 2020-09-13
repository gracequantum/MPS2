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
#include <stdlib.h>

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
  DGQTensor did = DGQTensor({pb_in, pb_out});
  ZGQTensor zid = ZGQTensor({pb_in, pb_out});
  DTenPtrVec dmps = DTenPtrVec(N);
  ZTenPtrVec zmps = ZTenPtrVec(N);

  std::vector<long> stat_labs1;
  std::vector<long> stat_labs2;

  void SetUp(void) {
    dntot({0, 0}) = 0;
    dntot({1, 1}) = 1;
    zntot({0, 0}) = 0;
    zntot({1, 1}) = 1;
    did({0, 0}) = 1;
    did({1, 1}) = 1;
    zid({0, 0}) = 1;
    zid({1, 1}) = 1;

    for (long i = 0; i < N; ++i) {
      stat_labs1.push_back(1);
      stat_labs2.push_back(i % 2);
    }
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

///support for non-uniform index
template <typename MpsType, typename TenElemType>
void RunTestMeasureOneSiteOpCase(
  MpsType &mps,
  const GQTensor<TenElemType> &op, const std::vector<long> site_set,
  const std::vector<TenElemType> &res) {
  auto measu_res = MeasureOneSiteOp(mps, op,site_set, "op1");
  assert(measu_res.size() == res.size());
  for (size_t i = 0; i < res.size(); ++i) {
    ExpectDoubleEq(measu_res[i].avg, res[i]);
  }
}

TEST_F(TestMpsMeasurement, TestMeasureOneSiteOp) {
  // Double case 1
  std::vector<GQTEN_Double> dres1;
  for (long i = 0; i < N; ++i) { dres1.push_back(stat_labs1[i]); }
  auto dmps1 = dmps;
  DirectStateInitMps(dmps1, stat_labs1, pb_out, qn0);
  auto dmps_for_measu1 = MPS<DGQTensor>(N);
  for (long i = 0; i < N; ++i) { dmps_for_measu1[i] = *dmps1[i]; }
  RunTestMeasureOneSiteOpCase(dmps_for_measu1, dntot, dres1);
  MpsFree(dmps1);
  // Double case 2
  std::vector<GQTEN_Double> dres2;
  for (long i = 0; i < N; ++i) { dres2.push_back(stat_labs2[i]); }
  auto dmps2 = dmps;
  DirectStateInitMps(dmps2, stat_labs2, pb_out, qn0);
  auto dmps_for_measu2 = MPS<DGQTensor>(N);
  for (long i = 0; i < N; ++i) { dmps_for_measu2[i] = *dmps2[i]; }
  RunTestMeasureOneSiteOpCase(dmps_for_measu2, dntot, dres2);
  MpsFree(dmps2);
  // Complex case 1
  std::vector<GQTEN_Complex> zres1;
  for (auto d : dres1) { zres1.push_back(d); }
  auto zmps1 = zmps;
  DirectStateInitMps(zmps1, stat_labs1, pb_out, qn0);
  auto zmps_for_measu1 = MPS<ZGQTensor>(N);
  for (long i = 0; i < N; ++i) { zmps_for_measu1[i] = *zmps1[i]; }
  RunTestMeasureOneSiteOpCase(zmps_for_measu1, zntot, zres1);
  MpsFree(zmps1);
  // Complex case 2
  std::vector<GQTEN_Complex> zres2;
  for (auto d : dres2) { zres2.push_back(d); }
  auto zmps2 = zmps;
  DirectStateInitMps(zmps2, stat_labs2, pb_out, qn0);
  auto zmps_for_measu2 = MPS<ZGQTensor>(N);
  for (long i = 0; i < N; ++i) { zmps_for_measu2[i] = *zmps2[i]; }
  RunTestMeasureOneSiteOpCase(zmps_for_measu2, zntot, zres2);
  MpsFree(zmps2);
}


//template <typename MpsType, typename TenElemType>
//void RunTestMeasureTwoSiteOpCase(
    //MpsType &mps,
    //const std::vector<GQTensor<TenElemType>> &phys_ops,
    //const GQTensor<TenElemType> &inst_op,
    //const GQTensor<TenElemType> &id_op,
    //const std::vector<std::vector<long>> &sites_set,
    //const std::vector<TenElemType> &res) {
  //auto measu_res = MeasureTwoSiteOp(
                       //mps, phys_ops, inst_op, id_op, sites_set, "op1op2");
  //assert(measu_res.size() == res.size());
  //for (size_t i = 0; i < res.size(); ++i) {
    //ExpectDoubleEq(measu_res[i].avg, res[i]);
  //}
//}


//TEST_F(TestMpsMeasurement, TestMeasureTwoSiteOp) {
  //std::vector<std::vector<long>> sites_set = {
                                               //{0, 1}, {0, 2}, {0, 5},
                                               //{1, 2}, {1, 3},
                                               //{4, 5}
                                             //};

  //// Double case 1
  //std::vector<GQTEN_Double> dres1(sites_set.size(), 1.0);
  //auto dmps1 = dmps;
  //DirectStateInitMps(dmps1, stat_labs1, pb_out, qn0);
  //auto dmps_for_measu1 = MPS<DGQTensor>(dmps1, -1); 
  //RunTestMeasureTwoSiteOpCase(
      //dmps_for_measu1, {did, did}, did, did, sites_set, dres1);
  //RunTestMeasureTwoSiteOpCase(
      //dmps_for_measu1, {dntot, dntot}, did, did, sites_set, dres1);
  //MpsFree(dmps1);
  //// Double case 2
  //auto dmps2 = dmps;
  //DirectStateInitMps(dmps2, stat_labs2, pb_out, qn0);
  //auto dmps_for_measu2 = MPS<DGQTensor>(dmps2, -1); 
  //RunTestMeasureTwoSiteOpCase(
      //dmps_for_measu2, {did, did}, did, did, sites_set, dres1);
  //std::vector<GQTEN_Double> dres2 = {0, 0, 0, 0, 1, 0};
  //RunTestMeasureTwoSiteOpCase(
      //dmps_for_measu2, {dntot, dntot}, did, did, sites_set, dres2);
  //MpsFree(dmps2);

  //// Complex case 1
  //std::vector<GQTEN_Complex> zres1(sites_set.size(), 1.0);
  //auto zmps1 = zmps;
  //DirectStateInitMps(zmps1, stat_labs1, pb_out, qn0);
  //auto zmps_for_measu1 = MPS<ZGQTensor>(zmps1, -1); 
  //RunTestMeasureTwoSiteOpCase(
      //zmps_for_measu1, {zid, zid}, zid, zid, sites_set, zres1);
  //RunTestMeasureTwoSiteOpCase(
      //zmps_for_measu1, {zntot, zntot}, zid, zid, sites_set, zres1);
  //MpsFree(zmps1);
  //// Double case 2
  //auto zmps2 = zmps;
  //DirectStateInitMps(zmps2, stat_labs2, pb_out, qn0);
  //auto zmps_for_measu2 = MPS<ZGQTensor>(zmps2, -1); 
  //RunTestMeasureTwoSiteOpCase(
      //zmps_for_measu2, {zid, zid}, zid, zid, sites_set, zres1);
  //std::vector<GQTEN_Complex> zres2 = {0, 0, 0, 0, 1, 0};
  //RunTestMeasureTwoSiteOpCase(
      //zmps_for_measu2, {zntot, zntot}, zid, zid, sites_set, zres2);
  //MpsFree(zmps2);
//}


//struct TestNonUniformMpsMeasurement : public testing::Test {
  //long N = 4; //unit cell number, A-B-A-B-A-B-A-B

  //QN qn0 = QN({QNNameVal("N", 0)});
  //Index pb_outA = Index({QNSector(QN({QNNameVal("N", 0)}), 1),
                        //QNSector(QN({QNNameVal("N", 1)}), 1)}, OUT);
  //Index pb_inA = InverseIndex(pb_outA);
  //Index pb_outB = Index({QNSector(QN({QNNameVal("N", 0)}), 2)}, OUT);
  //Index pb_inB = InverseIndex(pb_outB);

  //DGQTensor dntotA = DGQTensor({pb_inA, pb_outA});
  //DGQTensor didA = DGQTensor({pb_inA, pb_outA});
  //DGQTensor didB = DGQTensor({pb_inB, pb_outB});
  //DTenPtrVec dmps = DTenPtrVec(2*N);
  //std::vector<Index> pb_out_set = std::vector<Index>(2*N);

  //std::vector<long> stat_labs1 = std::vector<long>(2*N,1);
  //std::vector<long> A_sublattice=std::vector<long>(N);

  //void SetUp(void) {
    //dntotA({0, 0}) = 0;
    //dntotA({1, 1}) = 1;
    //didA({0, 0}) = 1;
    //didA({1, 1}) = 1;

    //didB({0, 0}) = 1;
    //didB({1, 1}) = 1;
    //srand((unsigned)time(NULL));
    //for (long i = 0; i < 2*N; ++i) {
      //if(i%2==0) pb_out_set[i]=pb_outA;
      //else pb_out_set[i]=pb_outB;
    //}
    //for (long i=0;i<N;++i){
      //A_sublattice[i]=2*i;
      //stat_labs1[2*i]= rand()%2;
    //}
  //}
//};


//TEST_F(TestNonUniformMpsMeasurement, TestNoUniformMeasureOneSiteOp) {
  //std::vector<GQTEN_Double> dres1(N);
  //for (long i = 0; i < N; ++i) { dres1[i]=stat_labs1[2*i]; }
  //auto dmps1 = dmps;
  //DirectStateInitMps(dmps1, stat_labs1, pb_out_set, qn0);
  //auto dmps_for_measu1 = MPS<DGQTensor>(dmps1, -1);
  //RunTestMeasureOneSiteOpCase(dmps_for_measu1, dntotA,A_sublattice, dres1);
  //MpsFree(dmps1);
//}

/////No insertion operator case, can used for uniform or non-uniform indices
//template <typename MpsType, typename TenElemType>
//void RunTestMeasureTwoSiteOpCase(
  //MpsType &mps,
  //const std::vector<GQTensor<TenElemType>> &phys_ops,
  //const std::vector<std::vector<long>> &sites_set,
  //const std::vector<TenElemType> &res) {
  //auto measu_res = MeasureTwoSiteOp(
    //mps, phys_ops, sites_set, "op1op2");
  //assert(measu_res.size() == res.size());
  //for (size_t i = 0; i < res.size(); ++i) {
    //ExpectDoubleEq(measu_res[i].avg, res[i]);
  //}
//}

//TEST_F(TestNonUniformMpsMeasurement, TestNoUniformMeasureTwoSiteOp) {
//std::vector<std::vector<long>> sites_set = {
  //{0, 2}, {0, 4},{2, 4}
//};
//// Double case 1
//std::vector<GQTEN_Double> dres1(sites_set.size());
//for(int i = 0;i<sites_set.size();i++){
  //long site1 = sites_set[i][0];
  //long site2 = sites_set[i][1];
  //dres1[i]=stat_labs1[site1]*stat_labs1[site2];
//}
//auto dmps1 = dmps;
//DirectStateInitMps(dmps1, stat_labs1, pb_out_set, qn0);
//auto dmps_for_measu1 = MPS<DGQTensor>(dmps1, -1);
//RunTestMeasureTwoSiteOpCase(
  //dmps_for_measu1, {dntotA, dntotA}, sites_set, dres1);
//MpsFree(dmps1);
//}

