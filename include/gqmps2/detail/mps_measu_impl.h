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
template <typename TenElemType>
MeasuResElem<TenElemType> OneSiteOpAvg(
    const GQTensor<TenElemType> &, const GQTensor<TenElemType> &,
    const long, const long);

template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const std::vector<std::vector<GQTensor<TenElemType>>> &,
    const GQTensor<TenElemType> &,
    const std::vector<long> &);

template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const GQTensor<TenElemType> &,
    const std::vector<long> &);

template <typename TenElemType>
TenElemType OpsVecAvg(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const size_t,
    const size_t,
    const GQTensor<TenElemType> &
);

template <typename TenType>
void CtrctMidTen(
    const MPS<TenType> &, const long,
    const TenType &, const TenType &,
    TenType * &);

template <typename AvgType>
void DumpMeasuRes(const MeasuRes<AvgType> &, const std::string &);


// Helpers.
inline bool IsOrderKept(const std::vector<long> &sites) {
  auto ordered_sites = sites;
  std::sort(ordered_sites.begin(), ordered_sites.end());
  for (std::size_t i = 0; i < sites.size(); ++i) {
    if (sites[i] != ordered_sites[i]) { return false; }
  }
  return true;
}


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


// Measure two-site operator.
template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<GQTensor<TenElemType>> &phys_ops,
    const std::vector<std::vector<GQTensor<TenElemType>>> &inst_ops_set,  ///< inset operators for each measure event
    const GQTensor<TenElemType> &id_op,
    const std::vector<std::vector<long>> &sites_set,
    const std::string &res_file_basename
) {
  // Deal with two physical operators
  assert(phys_ops.size() == 2);
  auto measu_event_num = sites_set.size();
  std::vector<std::vector<GQTensor<TenElemType>>> phys_ops_set(
                                                      measu_event_num,
                                                      phys_ops);

  // Deal with inset operators for each measure event
  assert(inst_ops_set.size() == measu_event_num);
  std::vector<std::vector<std::vector<GQTensor<TenElemType>>>> inst_ops_set_set;
  for (size_t i = 0; i < measu_event_num; ++i) {
    assert(sites_set[i].size() == 2);
    assert((sites_set[i][1] - sites_set[i][0] - 1) == inst_ops_set[i].size());
    inst_ops_set_set.push_back({inst_ops_set[i]});
  }

  return MeasureMultiSiteOp(
      mps,
      phys_ops_set,
      inst_ops_set_set,
      id_op,
      sites_set,
      res_file_basename
  );
}


template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<GQTensor<TenElemType>> &phys_ops,
    const GQTensor<TenElemType> &inst_op,
    const GQTensor<TenElemType> &id_op,
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
             id_op,
             sites_set,
             res_file_basename);
}


// Measure multi-site operator.
template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<std::vector<GQTensor<TenElemType>>> &phys_ops_set,
    const std::vector<
        std::vector<std::vector<GQTensor<TenElemType>>>
    >                                                     &inst_ops_set_set,
    const GQTensor<TenElemType> &id_op,
    const std::vector<std::vector<long>> &sites_set,
    const std::string &res_file_basename
) {
  auto measu_event_num = sites_set.size();
  MeasuRes<TenElemType> measu_res(measu_event_num);
  for (std::size_t i = 0; i < measu_event_num; ++i) {
    auto &phys_ops = phys_ops_set[i];
    auto &inst_ops_set = inst_ops_set_set[i];
    auto &sites = sites_set[i];
    assert(sites.size() > 1);
    assert(IsOrderKept(sites));
    CentralizeMps(mps, sites[0]);
    measu_res[i] = MultiSiteOpAvg(mps, phys_ops, inst_ops_set, id_op, sites);
  }
  DumpMeasuRes(measu_res, res_file_basename);
  return measu_res;
}


template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<std::vector<GQTensor<TenElemType>>> &phys_ops_set,
    const std::vector<std::vector<GQTensor<TenElemType>>> &inst_ops_set,
    const GQTensor<TenElemType> &id_op,
    const std::vector<std::vector<long>> &sites_set,
    const std::string &res_file_basename) {
  auto measu_event_num = sites_set.size();
  MeasuRes<TenElemType> measu_res(measu_event_num);
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
  return measu_res;
}


// Averages.
template <typename TenElemType>
MeasuResElem<TenElemType> OneSiteOpAvg(
    const GQTensor<TenElemType> &cent_ten, const GQTensor<TenElemType> &op,
    const long site, const long N) {
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
    ta_ctrct_axes2 = {0, 1};
    tb_ctrct_axes2 = {0, 1};
  } else {
    ta_ctrct_axes1 = {1};
    tb_ctrct_axes1 = {0};
    ta_ctrct_axes2 = {0, 2, 1};
    tb_ctrct_axes2 = {0, 1, 2};
  }
  auto temp_ten = Contract(cent_ten, op, {ta_ctrct_axes1, tb_ctrct_axes1});
  auto res_ten = Contract(
                     *temp_ten, Dag(cent_ten),
                     {ta_ctrct_axes2, tb_ctrct_axes2});
  delete temp_ten;
  auto avg = res_ten->scalar;
  delete res_ten;
  return MeasuResElem<TenElemType>({site}, avg);
}


template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg(
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<GQTensor<TenElemType>> &phys_ops,
    const std::vector<std::vector<GQTensor<TenElemType>>> &inst_ops_set,
    const GQTensor<TenElemType> &id_op,
    const std::vector<long> &sites) {
  auto inst_ops_num = inst_ops_set.size();
  auto phys_op_num = phys_ops.size();
  assert(phys_op_num == (inst_ops_num + 1));
  std::vector<GQTensor<TenElemType>> ops;
  for (size_t i = 0; i < inst_ops_num; ++i) {
    ops.push_back(phys_ops[i]);
    for (long j = sites[i] + 1; j < sites[i+1]; ++j) {
      ops.push_back(inst_ops_set[i][j - sites[i] - 1]);
    }
  }
  ops.push_back(phys_ops.back());

  auto avg = OpsVecAvg(mps, ops, sites.front(), sites.back(), id_op);

  return MeasuResElem<TenElemType>(sites, avg);
}


template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg(
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<GQTensor<TenElemType>> &phys_ops,
    const std::vector<GQTensor<TenElemType>> &inst_ops,
    const GQTensor<TenElemType> &id_op,
    const std::vector<long> &sites) {
  auto inst_op_num = inst_ops.size();
  auto phys_op_num = phys_ops.size();
  assert(phys_op_num == (inst_op_num + 1));
  std::vector<std::vector<GQTensor<TenElemType>>> inst_ops_set;
  for (size_t i = 0; i < inst_op_num; ++i) {
    inst_ops_set.push_back(
        std::vector<GQTensor<TenElemType>>(
            sites[i+1] - sites[i] - 1,
            inst_ops[i]
        )
    );
  }

  return MultiSiteOpAvg(mps, phys_ops, inst_ops_set, id_op, sites);
}


template <typename TenElemType>
TenElemType OpsVecAvg(
    MPS<GQTensor<TenElemType>> &mps,      // Has been centralized to head_site
    const std::vector<GQTensor<TenElemType>> &ops,
    const size_t head_site,
    const size_t tail_site,
    const GQTensor<TenElemType> &id_op
) {
  // Deal with head tensor.
  std::vector<long> head_mps_ten_ctrct_axes1;
  std::vector<long> head_mps_ten_ctrct_axes2;
  std::vector<long> head_mps_ten_ctrct_axes3;
  if (head_site == 0) {
    head_mps_ten_ctrct_axes1 = {0};
    head_mps_ten_ctrct_axes2 = {1};
    head_mps_ten_ctrct_axes3 = {0};
  } else {
    head_mps_ten_ctrct_axes1 = {1};
    head_mps_ten_ctrct_axes2 = {0, 2};
    head_mps_ten_ctrct_axes3 = {0, 1};
  }
  auto temp_ten0 = Contract(
                       *mps.tens[head_site], ops[0],
                       {head_mps_ten_ctrct_axes1, {0}});
  auto temp_ten = Contract(
                       *temp_ten0, Dag(*mps.tens[head_site]),
                       {head_mps_ten_ctrct_axes2, head_mps_ten_ctrct_axes3});
  delete temp_ten0;

  // Deal with middle tensors.
  assert(ops.size() == (tail_site - head_site + 1));
  for (size_t i = head_site + 1; i < tail_site; ++i) {
    CtrctMidTen(mps, i, ops[i - head_site], id_op, temp_ten);
  }

  // Deal with tail tensor.
  std::vector<long> tail_mps_ten_ctrct_axes1;
  std::vector<long> tail_mps_ten_ctrct_axes2;
  if (tail_site == mps.N-1) {
    tail_mps_ten_ctrct_axes1 = {0, 1}; 
    tail_mps_ten_ctrct_axes2 = {1, 0};
  } else {
    tail_mps_ten_ctrct_axes1 = {0, 1, 2};
    tail_mps_ten_ctrct_axes2 = {2, 0, 1};
  }
  auto temp_ten2 = Contract(
                       *mps.tens[tail_site], *temp_ten, 
                       {{0}, {0}});
  delete temp_ten;
  auto temp_ten3 = Contract(*temp_ten2, ops.back(), {{0}, {0}});
  delete temp_ten2;
  auto res_ten = Contract(
                     *temp_ten3, Dag(*mps.tens[tail_site]),
                     {tail_mps_ten_ctrct_axes1, tail_mps_ten_ctrct_axes2});
  delete temp_ten3;
  auto avg = res_ten->scalar;
  delete res_ten;

  return avg;
}


template <typename TenType>
void CtrctMidTen(
    const MPS<TenType> &mps, const long site,
    const TenType &op, const TenType &id_op,
    TenType * &t) {
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
