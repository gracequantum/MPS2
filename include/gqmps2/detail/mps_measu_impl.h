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
void SaveMps(const int i, std::vector<TenType *> & mps);


///Below two functions are added by wanghx 25 June 2020, note don't conflict in the future.
template <typename TenType>
void ReadMps(const int i, std::vector<TenType *> & mps){
  assert(mps[i]==NULL);
  mps[i] = new TenType();
  std::string file;
  file = kMpsPath + "/" + kMpsTenBaseName + std::to_string(i) + "." + kGQTenFileSuffix;
  std::ifstream ifs(file, std::ifstream::binary);
  bfread(ifs, *mps[i]);
  ifs.close();
  return;
}
template <typename TenType>
void SaveMps(const int i, std::vector<TenType *> & mps){
  std::string file;
  file = kMpsPath + "/" + kMpsTenBaseName + std::to_string(i) + "." + kGQTenFileSuffix;
  std::ofstream ofs(file, std::ifstream::binary);
  bfwrite(ofs, *mps[i]);
  ofs.close();
  delete mps[i];
  mps[i] = NULL;
  return;
}

template <typename TenElemType>
MeasuResElem<TenElemType> OneSiteOpAvg(
  const GQTensor<TenElemType> &, // mps tensor
  const GQTensor<TenElemType> &, // operator being measured
  const long, //site number of the operator
  const long);// total system size

template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const std::vector<long> &);

template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg(
  MPS<GQTensor<TenElemType>> &,
  const std::vector<GQTensor<TenElemType>> &,
  const std::vector<GQTensor<TenElemType>> &,
  const std::vector<long> &,
  const std::vector<long> &);

template <typename TenType>
void CtrctMidTen(
    const MPS<TenType> &, const long,
    const TenType &,TenType * &);

template <typename TenType>
void CtrctMidTen(
  const MPS<TenType> &mps, const long site, TenType * &t);
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


/// Measure one-site operator.
/// Modified by wanghx 25 June 2020, to add support for disk operation.
template <typename TenElemType>
MeasuRes<TenElemType> MeasureOneSiteOp(
    MPS<GQTensor<TenElemType>> &mps,
    const GQTensor<TenElemType> &op, const std::string &res_file_basename) {
  auto N = mps.N;
  MeasuRes<TenElemType> measu_res(N);
  bool disk_flag =false;
  if(mps.tens[0] == NULL) disk_flag = true;
  for (std::size_t i = 0; i < N; ++i) {
    CentralizeMps(mps, i);
    if (disk_flag) ReadMps(i,mps.tens);
    measu_res[i] = OneSiteOpAvg(*mps.tens[i], op, i, N);
    if (disk_flag) SaveMps(i,mps.tens);
  }
  DumpMeasuRes(measu_res, res_file_basename);
  return measu_res;
}

/** MeasuRes<TenElemType> MeasureOneSiteOp
 * To specify which site to be measured
 * @tparam TenElemType
 * @param mps
 * @param op
 * @param res_file_basename
 * @return
 */
template <typename TenElemType>
MeasuRes<TenElemType> MeasureOneSiteOp(
  MPS<GQTensor<TenElemType>> &mps,
  const GQTensor<TenElemType> &op,
  const std::vector<long> &site_set,
  const std::string &res_file_basename) {
  auto N = mps.N;
  MeasuRes<TenElemType> measu_res(N);
  bool disk_flag =false;
  if(mps.tens[0] == NULL) disk_flag = true;
  for (auto iter=site_set.begin(); iter<site_set.end();iter++) {
    long i = *iter;
    CentralizeMps(mps, i);
    if (disk_flag) ReadMps(i,mps.tens);
    measu_res[i] = OneSiteOpAvg(*mps.tens[i], op, i, N);
    if (disk_flag) SaveMps(i,mps.tens);
  }
  DumpMeasuRes(measu_res, res_file_basename);
  return measu_res;
}

/// Modified by wanghx 25 June 2020, to add support for disk operation.
template <typename TenElemType>
MeasuResSet<TenElemType> MeasureOneSiteOp(
    MPS<GQTensor<TenElemType>> &mps,
    const std::vector<GQTensor<TenElemType>> &ops,
    const std::vector<std::string> &res_file_basenames) {
  auto op_num = ops.size();
  assert(op_num == res_file_basenames.size());
  auto N = mps.N;
  bool disk_flag =false;
  if(mps.tens[0] == NULL) disk_flag = true;
  MeasuResSet<TenElemType> measu_res_set(op_num);
  for (auto &measu_res : measu_res_set) {
    measu_res = MeasuRes<TenElemType>(N);
  }
  for (std::size_t i = 0; i < N; ++i) {
    CentralizeMps(mps, i);
    if (disk_flag) ReadMps(i,mps.tens);
    for (std::size_t j = 0; j < op_num; ++j) {
      measu_res_set[j][i] = OneSiteOpAvg(*mps.tens[i], ops[j], i, N);
    }
    if (disk_flag) SaveMps(i,mps.tens);
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


/** MeasuRes<TenElemType> MeasureTwoSiteOp
 * No insertion operators
 * @tparam TenElemType
 * @param mps
 * @param phys_ops
 * @param sites_set
 * @param res_file_basename
 * @return
 */
template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
  MPS<GQTensor<TenElemType>> &mps,
  const std::vector<GQTensor<TenElemType>> &phys_ops,
  const std::vector<std::vector<long>> &sites_set,
  const std::string &res_file_basename) {
  assert(phys_ops.size() == 2);
  auto measu_event_num = sites_set.size();
  std::vector<std::vector<GQTensor<TenElemType>>> phys_ops_set(
    measu_event_num,
    phys_ops);
  auto inst_op = phys_ops.front();
  std::vector<std::vector<GQTensor<TenElemType>>> inst_ops_set(
    measu_event_num,
    {inst_op});
  std::vector<long> NullVector;
  std::vector<std::vector<long>> insertsite_set(measu_event_num, NullVector);
  return MeasureMultiSiteOp(
    mps,
    phys_ops_set,
    inst_ops_set,
    sites_set,
    insertsite_set,
    res_file_basename);
}


// Measure multi-site operator.
template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(
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


/** MeasuRes<TenElemType> MeasureMultiSiteOp
 * Specify which sites are inserted
 * @tparam TenElemType
 * @param mps
 * @param phys_ops_set
 * @param inst_ops_set
 * @param sites_set
 * @param insertsites_set
 * @param res_file_basename
 * @return
 */
template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(
  MPS<GQTensor<TenElemType>> &mps,
  const std::vector<std::vector<GQTensor<TenElemType>>> &phys_ops_set,
  const std::vector<std::vector<GQTensor<TenElemType>>> &inst_ops_set,
  const std::vector<std::vector<long>> &sites_set,
  const std::vector<std::vector<long>> &insertsites_set,
  const std::string &res_file_basename) {
  auto measu_event_num = sites_set.size();
  MeasuRes<TenElemType> measu_res(measu_event_num);
  for (std::size_t i = 0; i < measu_event_num; ++i) {
    auto &phys_ops = phys_ops_set[i];
    auto &inst_ops = inst_ops_set[i];
    auto &sites = sites_set[i];
    auto &insert_sites=insertsites_set[i];
    assert(sites.size() > 1);
    assert(std::is_sorted(sites.begin(),sites.end()));
    CentralizeMps(mps, sites[0]);
    measu_res[i] = MultiSiteOpAvg(mps, phys_ops, inst_ops, sites, insert_sites);
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

/** MeasuResElem<TenElemType> MultiSiteOpAvg
 * Add by wanghx 25 June 2020.Difference with previous version: No identity parameter;
 * And mps can be saved into disk
 * @tparam TenElemType
 * @param mps
 * @param phys_ops
 * @param inst_ops
 * @param sites on which sites physical operators are acting
 * @return
 */
template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg(
  MPS<GQTensor<TenElemType>> &mps,
  const std::vector<GQTensor<TenElemType>> &phys_ops,
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
  bool disk_save_flag = false;
  if (mps.tens[0] == NULL){// if point to nothing, using the read-saving disk mode
    ReadMps(sites[0], mps.tens);
    disk_save_flag = true;
  }
  auto temp_ten0 = Contract(
    *mps.tens[sites[0]], phys_ops[0],
    {head_mps_ten_ctrct_axes1, {0}});
  auto temp_ten = Contract(
    *temp_ten0, Dag(*mps.tens[sites[0]]),
    {head_mps_ten_ctrct_axes2, head_mps_ten_ctrct_axes3});
  if(disk_save_flag) SaveMps(sites[0], mps.tens);
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
  if(disk_save_flag) ReadMps(sites[inst_op_num], mps.tens);
  auto temp_ten2 = Contract(
    *mps.tens[sites[inst_op_num]], *temp_ten,
    {{0}, {0}});
  delete temp_ten;
  auto temp_ten3 = Contract(*temp_ten2, phys_ops[phys_op_num-1], {{0}, {0}});
  delete temp_ten2;
  auto res_ten = Contract(// I think below phys_op_num-1 == inst_op_num? wanghx June 25 2020
    *temp_ten3, Dag(*mps.tens[sites[phys_op_num-1]]),
    {tail_mps_ten_ctrct_axes1, tail_mps_ten_ctrct_axes2});
  if(disk_save_flag) SaveMps(sites[inst_op_num], mps.tens);
  delete temp_ten3;
  auto avg = res_ten->scalar;
  delete res_ten;
  return MeasuResElem<TenElemType>(sites, avg);
}

/** MeasuResElem<TenElemType> MultiSiteOpAvg
 * Add by wanghx 30 June 2020. Can specify which sites are inserted. And mps can be saved into disk
 * @tparam TenElemType GQTEN_Double or GQTEN_Complex
 * @param mps MPS struct, including mps tensor pointers, center and mps size
 * @param phys_ops physical operators, a vector
 * @param inst_ops insertion operators, a vector
 * @param phys_sites on which sites physical operators are acting
 * @param inst_sites on which sites insertion operators are inserted
 * @return Measure's results
 */
template <typename TenElemType>
MeasuResElem<TenElemType> MultiSiteOpAvg(
  MPS<GQTensor<TenElemType>> &mps,
  const std::vector<GQTensor<TenElemType>> &phys_ops,
  const std::vector<GQTensor<TenElemType>> &inst_ops,
  const std::vector<long> &phys_sites,
  const std::vector<long> &inst_sites) {
  // Deal with head tensor.
  std::vector<long> head_mps_ten_ctrct_axes1;
  std::vector<long> head_mps_ten_ctrct_axes2;
  std::vector<long> head_mps_ten_ctrct_axes3;
  if (phys_sites[0] == 0) { // MPS have left canonical form to  phys_sites[0]
    head_mps_ten_ctrct_axes1 = {0};
    head_mps_ten_ctrct_axes2 = {1};
    head_mps_ten_ctrct_axes3 = {0};
  } else {
    head_mps_ten_ctrct_axes1 = {1};
    head_mps_ten_ctrct_axes2 = {0, 2};
    head_mps_ten_ctrct_axes3 = {0, 1};
  }
  bool disk_save_flag = false;
  if (mps.tens[0] == NULL){// if point to nothing, using the read-saving disk mode
    ReadMps(phys_sites[0], mps.tens);
    disk_save_flag = true;
  }
  auto temp_ten0 = Contract(
    *mps.tens[phys_sites[0]], phys_ops[0],
    {head_mps_ten_ctrct_axes1, {0}});
  auto temp_ten = Contract(
    *temp_ten0, Dag(*mps.tens[phys_sites[0]]),
    {head_mps_ten_ctrct_axes2, head_mps_ten_ctrct_axes3});
  if(disk_save_flag) SaveMps(phys_sites[0], mps.tens);
  delete temp_ten0;

  // Deal with middle tensors.
  auto inst_op_num = inst_ops.size();
  auto phys_op_num = phys_ops.size();
  assert(phys_op_num == (inst_op_num+1));
  for (std::size_t i = 0; i < inst_op_num; ++i) {
    for (long j = phys_sites[i]+1; j < phys_sites[i+1]; j++) {
      if(std::find(inst_sites.begin(), inst_sites.end(),j) != inst_sites.end())
        CtrctMidTen(mps, j, inst_ops[i], temp_ten);
      else CtrctMidTen(mps, j, temp_ten);
    }
    if (i != inst_op_num-1) {
      CtrctMidTen(mps, phys_sites[i+1], phys_ops[i+1], temp_ten);
    }
  }

  // Deal with tail tensor.
  std::vector<long> tail_mps_ten_ctrct_axes1;
  std::vector<long> tail_mps_ten_ctrct_axes2;
  if (phys_sites.back() == mps.N-1) {
    tail_mps_ten_ctrct_axes1 = {0, 1};
    tail_mps_ten_ctrct_axes2 = {1, 0};
  } else {
    tail_mps_ten_ctrct_axes1 = {0, 1, 2};
    tail_mps_ten_ctrct_axes2 = {2, 0, 1};
  }
  if(disk_save_flag) ReadMps(phys_sites[inst_op_num], mps.tens);
  auto temp_ten2 = Contract(
    *mps.tens[phys_sites[inst_op_num]], *temp_ten,
    {{0}, {0}});
  delete temp_ten;
  auto temp_ten3 = Contract(*temp_ten2, phys_ops[phys_op_num-1], {{0}, {0}});
  delete temp_ten2;
  auto res_ten = Contract(// I think below phys_op_num-1 == inst_op_num? wanghx June 25 2020
    *temp_ten3, Dag(*mps.tens[phys_sites[phys_op_num-1]]),
    {tail_mps_ten_ctrct_axes1, tail_mps_ten_ctrct_axes2});
  if(disk_save_flag) SaveMps(phys_sites[inst_op_num], mps.tens);
  delete temp_ten3;
  auto avg = res_ten->scalar;
  delete res_ten;
  return MeasuResElem<TenElemType>(phys_sites, avg);
}


/// Modified by hxwang June 27, 2020, to support non-uniform lattice.
template <typename TenType>
void CtrctMidTen(
  const MPS<TenType> &mps, const long site,
  const TenType &op, TenType * &t) {
  auto id_op = TenType({op.indexes[0],op.indexes[1]});
  for(int i = 0;i<id_op.shape[0];i++){
    id_op({i,i})=1.0;
  }
  bool disk_save_flag = false;
  if (mps.tens[0] == NULL){// if point to nothing, using the read-saving disk mode
    ReadMps(site, mps.tens);
    disk_save_flag = true;
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
  if (disk_save_flag) SaveMps(site, mps.tens);
}

/** void CtrctMidTen
 * No &op parameter, we suppose it equals identity.
 * @tparam TenType
 * @param mps
 * @param site
 * @param t
 */
template <typename TenType>
void CtrctMidTen(
  const MPS<TenType> &mps, const long site, TenType * &t) {
  bool disk_save_flag = false;
  if (mps.tens[0] == NULL){// if point to nothing, using the read-saving disk mode
    ReadMps(site, mps.tens);
    disk_save_flag = true;
  }
  auto temp_ten = Contract(*mps.tens[site], *t, {{0}, {0}});
  delete t;
  t = Contract(*temp_ten, Dag(*mps.tens[site]), {{0, 2}, {1, 0}});
  delete temp_ten;
  if (disk_save_flag) SaveMps(site, mps.tens);
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
