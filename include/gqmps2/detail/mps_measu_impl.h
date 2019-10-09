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


namespace gqmps2 {
using namespace gqten;


// Forward declaration.
template <typename TenElemType>
MeasuResElem<TenElemType> OneSiteOpAvg(
    const GQTensor<TenElemType> &, const GQTensor<TenElemType> &,
    const long, const long);


template <typename AvgType>
void DumpMeasuRes(const MeasuRes<AvgType> &, const std::string &);


// Helpers.
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


template <typename AvgType>
void DumpMeasuRes(
    const MeasuRes<AvgType> &res, const std::string &basename) {
  auto file = basename + ".json";
  std::ofstream ofs(file);

  ofs << "[\n";

  for (auto it = res.begin(); it != res.end()-1; ++it) {
    auto &measu_res_elem = *it;
    ofs << "  [";
    DumpSites(ofs, measu_res_elem.sites);
    DumpAvgVal(ofs, measu_res_elem.avg);
    ofs << "],\n";
  }

  auto &measu_res_elem = *(res.end()-1);
  ofs << "  [";
  DumpSites(ofs, measu_res_elem.sites);
  DumpAvgVal(ofs, measu_res_elem.avg);
  ofs << "]\n";

  ofs << "]";

  ofs.close();
}
} /* gqmps2 */ 
