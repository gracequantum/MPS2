/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-06-05 22:52
* 
* Description: GraceQ/mps2 project. Implement MPS observation measurements.
*/
#include "mps_measu.h"
#include "mps_ops.h"
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <string>
#include <fstream>
#include <iomanip>


namespace gqmps2 {
using namespace gqten;


void MeasureOneSiteOp(
    MPS &mps,
    const GQTensor &op,
    const std::string &res_file_basename) {
  auto N = mps.N;
  std::vector<MeasuRes> measu_res(N);
  for (std::size_t i = 0; i < N; ++i) {
    CentralizeMps(mps, i);
    measu_res[i] = OneSiteOpAvg(*mps.tens[i], op, i, N);
  }
  DumpMeasuRes(measu_res, res_file_basename);
}


void MeasureOneSiteOp(
    MPS &mps,
    const std::vector<GQTensor> &ops,
    const std::vector<std::string> &res_file_basenames) {
  auto op_num = ops.size();
  assert(op_num == res_file_basenames.size());
  auto N = mps.N;
  std::vector<std::vector<MeasuRes>> measu_res_set(op_num);
  for (auto &measu_res : measu_res_set) {
    measu_res = std::vector<MeasuRes>(N);
  }
  for (std::size_t i = 0; i < N; ++i) {
    CentralizeMps(mps, i);
    for (std::size_t j = 0; j < op_num; ++j) {
      measu_res_set[j][i] = OneSiteOpAvg(*mps.tens[i], ops[j], i, N);
    }
  }
  for (std::size_t i = 0; i < op_num; ++i) {
    DumpMeasuRes(measu_res_set[i], res_file_basenames[i]);
  }
}


void MeasureTwoSiteOp(
    MPS &mps,
    const std::vector<GQTensor> &phys_ops,
    const GQTensor &inst_op,
    const GQTensor &id_op,
    const std::vector<std::vector<long>> &sites_set,
    const std::string &res_file_basename) {
  assert(phys_ops.size() == 2);
  auto measu_event_num = sites_set.size();
  std::vector<std::vector<GQTensor>> phys_ops_set(measu_event_num, phys_ops);
  std::vector<std::vector<GQTensor>> inst_ops_set(measu_event_num, {inst_op});
  MeasureMultiSiteOp(
      mps,
      phys_ops_set,
      inst_ops_set,
      id_op,
      sites_set,
      res_file_basename);
}


void MeasureMultiSiteOp(
    MPS &mps,
    const std::vector<std::vector<GQTensor>> &phys_ops_set,
    const std::vector<std::vector<GQTensor>> &inst_ops_set,
    const GQTensor &id_op,
    const std::vector<std::vector<long>> &sites_set,
    const std::string &res_file_basename) {
  auto measu_event_num = sites_set.size();
  std::vector<MeasuRes> measu_res(measu_event_num);
  for (std::size_t i = 0; i < measu_event_num; ++i) {
    auto &phys_ops = phys_ops_set[i];
    auto &inst_ops = inst_ops_set[i];
    auto &sites = sites_set[i];
    assert(sites.size() > 1);
    assert(IsOrderKept(sites));
    CentralizeMps(mps, sites[0]);
    measu_res[i] = MultiSiteOpAvg(mps, phys_ops, inst_ops, id_op, sites);
  }
  DumpMeasuRes(measu_res, res_file_basename);
}


MeasuRes OneSiteOpAvg(
    const GQTensor &cent_ten, const GQTensor &op,
    const long site, const long N) {
  double avg;
  std::vector<long> ta_ctrct_axes1, tb_ctrct_axes1;
  std::vector<long> ta_ctrct_axes2, tb_ctrct_axes2;
  if (site == 0) {
    ta_ctrct_axes1 = {0};
    tb_ctrct_axes1 = {1};
    ta_ctrct_axes2 = {0, 1};
    tb_ctrct_axes2 = {1, 0};
  } else if (site == (N-1)) {
    ta_ctrct_axes1 = {1};
    tb_ctrct_axes1 = {0};
    ta_ctrct_axes2 = {1, 0};
    tb_ctrct_axes2 = {0, 1};
  } else {
    ta_ctrct_axes1 = {1};
    tb_ctrct_axes1 = {0};
    ta_ctrct_axes2 = {0, 2, 1};
    tb_ctrct_axes2 = {0, 1, 2};
  }
  auto temp_ten = Contract(cent_ten, op, {ta_ctrct_axes1, tb_ctrct_axes1});
  auto res_ten = Contract(
                     *temp_ten, MockDag(cent_ten),
                     {ta_ctrct_axes2, tb_ctrct_axes2});
  delete temp_ten;
  avg = res_ten->scalar;
  delete res_ten;
  return MeasuRes({site}, avg);
}


MeasuRes MultiSiteOpAvg(
    MPS &mps,
    const std::vector<GQTensor> &phys_ops,
    const std::vector<GQTensor> &inst_ops,
    const GQTensor &id_op,
    const std::vector<long> &sites) {
  // Deal with head tensor.
  std::vector<long> head_mps_ten_ctrct_axes1;
  std::vector<long> head_mps_ten_ctrct_axes2;
  std::vector<long> head_mps_ten_ctrct_axes3;
  if (sites[0] == 0) {
    head_mps_ten_ctrct_axes1 = {0};
    head_mps_ten_ctrct_axes2 = {1};
    head_mps_ten_ctrct_axes3 = {0};
  } else {
    head_mps_ten_ctrct_axes1 = {1};
    head_mps_ten_ctrct_axes2 = {0, 2};
    head_mps_ten_ctrct_axes3 = {0, 1};
  }
  auto temp_ten0 = Contract(
                       *mps.tens[sites[0]], phys_ops[0],
                       {head_mps_ten_ctrct_axes1, {0}});
  auto temp_ten = Contract(
                       *temp_ten0, MockDag(*mps.tens[sites[0]]),
                       {head_mps_ten_ctrct_axes2, head_mps_ten_ctrct_axes3});
  delete temp_ten0;

  // Deal with middle tensors.
  auto inst_op_num = inst_ops.size();
  auto phys_op_num = phys_ops.size();
  assert(phys_op_num == (inst_op_num+1));
  for (std::size_t i = 0; i < inst_op_num; ++i) {
    for (long j = sites[i]+1; j < sites[i+1]; j++) {
      CtrctMidTen(mps, j, inst_ops[i], id_op, temp_ten);
    } 
    if (i != inst_op_num-1) {
      CtrctMidTen(mps, sites[i+1], phys_ops[i+1], id_op, temp_ten); 
    }
  }

  // Deal with tail tensor.
  std::vector<long> tail_mps_ten_ctrct_axes1;
  std::vector<long> tail_mps_ten_ctrct_axes2;
  if (sites.back() == mps.N-1) {
    tail_mps_ten_ctrct_axes1 = {0, 1}; 
    tail_mps_ten_ctrct_axes2 = {1, 0};
  } else {
    tail_mps_ten_ctrct_axes1 = {0, 1, 2};
    tail_mps_ten_ctrct_axes2 = {2, 0, 1};
  }
  auto temp_ten2 = Contract(
                       *mps.tens[sites[inst_op_num]], *temp_ten, 
                       {{0}, {0}});
  delete temp_ten;
  auto temp_ten3 = Contract(*temp_ten2, phys_ops[phys_op_num-1], {{0}, {0}});
  delete temp_ten2;
  auto res_ten = Contract(
                     *temp_ten3, MockDag(*mps.tens[sites[phys_op_num-1]]),
                     {tail_mps_ten_ctrct_axes1, tail_mps_ten_ctrct_axes2});
  delete temp_ten3;
  auto avg = res_ten->scalar;
  delete res_ten;
  return MeasuRes(sites, avg);
}


void CtrctMidTen(
    const MPS &mps, const long site,
    const GQTensor &op, const GQTensor &id_op, GQTensor * &t) {
  if (op == id_op) {
    auto temp_ten = Contract(*mps.tens[site], *t, {{0}, {0}});
    delete t;
    t = Contract(*temp_ten, MockDag(*mps.tens[site]), {{0, 2}, {1, 0}});
    delete temp_ten;
  } else {
    auto temp_ten1 = Contract(*mps.tens[site], *t, {{0}, {0}});
    delete t;
    auto temp_ten2 = Contract(*temp_ten1, op, {{0}, {0}});
    delete temp_ten1;
    t = Contract(*temp_ten2, MockDag(*mps.tens[site]), {{1, 2}, {0, 1}});
    delete temp_ten2;
  }
}


void DumpMeasuRes(
    const std::vector<MeasuRes> &res, const std::string &basename) {
  auto file = basename + ".json";
  std::ofstream ofs(file);
  ofs << "[\n";
  for (auto it = res.begin(); it != res.end()-1; ++it) {
    auto &measu_res = *it;
    ofs << "  [[";
    for (auto it = measu_res.sites.begin();
         it != measu_res.sites.end()-1;
         ++it) {
      ofs << *it << ", ";
    }
    ofs << measu_res.sites.back() << "], ";
    ofs << std::setw(14) << std::setprecision(12) << measu_res.avg << "],\n";
  }
  auto &measu_res = *(res.end()-1);
  ofs << "  [[";
  for (auto it = measu_res.sites.begin();
       it != measu_res.sites.end()-1;
       ++it) {
    ofs << *it << ", ";
  }
  ofs << measu_res.sites.back() << "], ";
  ofs << std::setw(14) << std::setprecision(12) << measu_res.avg << "]\n";
  ofs << "]";
  ofs.close();
}
} /* gqmps2 */ 
