// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-10-08 22:18
* 
* Description: GraceQ/MPS2 project. Implementation details for MPS observation measurements.
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <string>
#include <fstream>
#include <iomanip>
#include <algorithm>


namespace gqmps2 {
using namespace gqten;


// Forward declaration.

template <typename TenType>
void ReadMps(const int i, std::vector<TenType *> & mps);
template <typename TenType>
void ReadMps(const int i, std::vector<TenType *> & mps);

template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const GQTensor<TenElemType> &,
    const std::vector<long> &);

template <typename TenType>
void CtrctMidTen(
    const MPS<TenType> &, const long,
    const TenType &, const TenType &,
    TenType * &);

template <typename AvgType>
void DumpMeasuRes(const MeasuRes<AvgType> &, const std::string &);


inline void DumpSites(std::ofstream &ofs, const std::vector<long> &sites) {
  ofs << "[";
  for (auto it = sites.begin(); it != sites.end()-1; ++it) {
    ofs << *it << ", ";
  }
  ofs << sites.back();
  ofs << "], ";
}


inline void DumpAvgVal(std::ofstream &ofs, const GQTEN_Double avg) {
  ofs << std::setw(14) << std::setprecision(12) << avg;
}


inline void DumpAvgVal(std::ofstream &ofs, const GQTEN_Complex avg) {
  ofs << "[";  
  ofs << std::setw(14) << std::setprecision(12) << avg.real();
  ofs << ", ";
  ofs << std::setw(14) << std::setprecision(12) << avg.imag();
  ofs << "]";
}


// Measure one-site operator.
template <typename TenElemType>
MeasuRes<TenElemType> MeasureOneSiteOp(
    MPS<GQTensor<TenElemType>> &mps,
    const GQTensor<TenElemType> &op, const std::string &res_file_basename) {
  auto N = mps.N;
  MeasuRes<TenElemType> measu_res(N);
  for (std::size_t i = 0; i < N; ++i) {
    CentralizeMps(mps, i);
    measu_res[i] = OneSiteOpAvg(*mps.tens[i], op, i, N);
  }
  DumpMeasuRes(measu_res, res_file_basename);
  return measu_res;
}


template <typename TenElemType>
MeasuResSet<TenElemType> MeasureOneSiteOp(
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<GQTensor<TenElemType>> &ops,
    const std::vector<std::string> &res_file_basenames) {
  auto op_num = ops.size();
  assert(op_num == res_file_basenames.size());
  auto N = mps.N;
  MeasuResSet<TenElemType> measu_res_set(op_num);
  for (auto &measu_res : measu_res_set) {
    measu_res = MeasuRes<TenElemType>(N);
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
  return measu_res_set;
}


/** MeasureTwoSiteOp
 * Measure two-site operator. For the differences with previous version, see MeasureMultiSiteOp
 * @tparam TenElemType
 * @param mps
 * @param phys_ops
 * @param inst_op
 * @param sites_set
 * @param res_file_basename
 * @return
 */
template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<GQTensor<TenElemType>> &phys_ops,
    const GQTensor<TenElemType> &inst_op,
    const std::vector<std::vector<long>> &sites_set,
    const std::string &res_file_basename) {
  assert(phys_ops.size() == 2);
  auto measu_event_num = sites_set.size();
  std::vector<std::vector<GQTensor<TenElemType>>> phys_ops_set(
                                                      measu_event_num,
                                                      phys_ops);
  std::vector<std::vector<GQTensor<TenElemType>>> inst_ops_set(
                                                      measu_event_num,
                                                      {inst_op});
  return MeasureMultiSiteOp(
             mps,
             phys_ops_set,
             inst_ops_set,
             sites_set,
             res_file_basename);
}


/** MeasureMultiSiteOp
 * Measure multi-site operator. Differences with previous version:
 * - No input identity operator. So can work for non-uniform lattices.
 * - mps.tens point to NULL. All of the tensors are saved in disks.
 * @tparam TenElemType
 * @param mps
 * @param phys_ops_set
 * @param inst_ops_set
 * @param sites_set
 * @param res_file_basename
 * @return
 */
template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(// Here mps point to nothing
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<std::vector<GQTensor<TenElemType>>> &phys_ops_set,
    const std::vector<std::vector<GQTensor<TenElemType>>> &inst_ops_set,
    const std::vector<std::vector<long>> &sites_set,
    const std::string &res_file_basename) {
  auto measu_event_num = sites_set.size();
  MeasuRes<TenElemType> measu_res(measu_event_num);
  for (std::size_t i = 0; i < measu_event_num; ++i) {
    auto &phys_ops = phys_ops_set[i];
    auto &inst_ops = inst_ops_set[i];
    auto &sites = sites_set[i];
    assert(sites.size() > 1);
    assert(std::is_sorted(sites.begin(),sites.end()));
    CentralizeMps(mps, sites[0]);
    measu_res[i] = MultiSiteOpAvg(mps, phys_ops, inst_ops, sites);
  }
  DumpMeasuRes(measu_res, res_file_basename);
  return measu_res;
}


template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg( //Difference with previous version: No identity parameter
    MPS<GQTensor<TenElemType>> &mps,  // Actually, saving all the mps tensors to disk
    const std::vector<GQTensor<TenElemType>> &phys_ops, // and mps parameter points to nothing.
    const std::vector<GQTensor<TenElemType>> &inst_ops,
    const std::vector<long> &sites) {
  // Deal with head tensor.
  std::vector<long> head_mps_ten_ctrct_axes1;
  std::vector<long> head_mps_ten_ctrct_axes2;
  std::vector<long> head_mps_ten_ctrct_axes3;
  if (sites[0] == 0) { // MPS have left canonical form to sites[0]
    head_mps_ten_ctrct_axes1 = {0};
    head_mps_ten_ctrct_axes2 = {1};
    head_mps_ten_ctrct_axes3 = {0};
  } else {
    head_mps_ten_ctrct_axes1 = {1};
    head_mps_ten_ctrct_axes2 = {0, 2};
    head_mps_ten_ctrct_axes3 = {0, 1};
  }
  ReadMps(sites[0], mps.tens);
  auto temp_ten0 = Contract(
                       *mps.tens[sites[0]], phys_ops[0],
                       {head_mps_ten_ctrct_axes1, {0}});
  auto temp_ten = Contract(
                       *temp_ten0, Dag(*mps.tens[sites[0]]),
                       {head_mps_ten_ctrct_axes2, head_mps_ten_ctrct_axes3});
  SaveMps(sites[0], mps.tens);
  delete temp_ten0;

  // Deal with middle tensors.
  auto inst_op_num = inst_ops.size();
  auto phys_op_num = phys_ops.size();
  assert(phys_op_num == (inst_op_num+1));
  for (std::size_t i = 0; i < inst_op_num; ++i) {
    for (long j = sites[i]+1; j < sites[i+1]; j++) {
      CtrctMidTen(mps, j, inst_ops[i], temp_ten);
    } 
    if (i != inst_op_num-1) {
      CtrctMidTen(mps, sites[i+1], phys_ops[i+1], temp_ten);
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
  ReadMps(sites[inst_op_num], mps.tens);
  auto temp_ten2 = Contract(
                       *mps.tens[sites[inst_op_num]], *temp_ten, 
                       {{0}, {0}});
  delete temp_ten;
  auto temp_ten3 = Contract(*temp_ten2, phys_ops[phys_op_num-1], {{0}, {0}});
  delete temp_ten2;
  auto res_ten = Contract(// I think below phys_op_num-1 == inst_op_num? wanghx June 25 2020
                     *temp_ten3, Dag(*mps.tens[sites[phys_op_num-1]]),
                     {tail_mps_ten_ctrct_axes1, tail_mps_ten_ctrct_axes2});
  SaveMps(sites[inst_op_num], mps.tens);
  delete temp_ten3;
  auto avg = res_ten->scalar;
  delete res_ten;
  return MeasuResElem<TenElemType>(sites, avg);
}


template <typename TenType>
void CtrctMidTen(// Here mps can point to nothing
    const MPS<TenType> &mps, const long site,
    const TenType &op, TenType * &t) {
  auto id_op = TenType({op.indexes[0],op.indexes[1]});
  for(int i = 0;i<id_op.shape[0];i++){
    id_op({i,i})=1.0;
  }
  int disk_save_flag = 0;
  if (mps.tens[i] == NULL){// if point to nothing, using the read-saving disk mode
    ReadMps(site, mps.tens);
    disk_save_flag = 1;
  }
  if (op == id_op) {
    auto temp_ten = Contract(*mps.tens[site], *t, {{0}, {0}});
    delete t;
    t = Contract(*temp_ten, Dag(*mps.tens[site]), {{0, 2}, {1, 0}});
    delete temp_ten;
  } else {
    auto temp_ten1 = Contract(*mps.tens[site], *t, {{0}, {0}});
    delete t;
    auto temp_ten2 = Contract(*temp_ten1, op, {{0}, {0}});
    delete temp_ten1;
    t = Contract(*temp_ten2, Dag(*mps.tens[site]), {{1, 2}, {0, 1}});
    delete temp_ten2;
  }
  if (disk_save_flag == 1) SaveMps(site, mps.tens);
}


// Date dump.
template <typename AvgType>
void DumpMeasuRes(
    const MeasuRes<AvgType> &res, const std::string &basename) {
  auto file = basename + ".json";
  std::ofstream ofs(file);

  ofs << "[\n";

  for (auto it = res.begin(); it != res.end(); ++it) {
    auto &measu_res_elem = *it;

    ofs << "  [";

    DumpSites(ofs, measu_res_elem.sites); DumpAvgVal(ofs, measu_res_elem.avg);

    if (it == res.end()-1) {
      ofs << "]\n";
    } else {
      ofs << "],\n";
    }
  }

  ofs << "]";

  ofs.close();
}
} /* gqmps2 */ 
