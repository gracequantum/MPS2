// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-06-06 09:37
* 
* Description: GraceQ/mps2 project. MPS observation measurements.
*/
#ifndef GQMPS2_MPS_MEASU_H
#define GQMPS2_MPS_MEASU_H


#include "gqten/gqten.h"
#include "gqmps2/gqmps2.h"

#include <vector>
#include <algorithm>


namespace gqmps2 {
using namespace gqten;


struct MeasuRes {
  MeasuRes(void) = default;
  MeasuRes(const std::vector<long> &sites, const double avg) :
    sites(sites), avg(avg) {}

  std::vector<long> sites;
  double avg;
};

MeasuRes OneSiteOpAvg(
    const GQTensor &, const GQTensor &, const long, const long);

MeasuRes MultiSiteOpAvg(
    MPS &,
    const std::vector<GQTensor> &,
    const std::vector<GQTensor> &,
    const GQTensor &,
    const std::vector<long> &);

void CtrctMidTen(
    const MPS &, const long, const GQTensor &, const GQTensor &, GQTensor *&);

void DumpMeasuRes(const std::vector<MeasuRes> &, const std::string &);


// Inline functions.
inline bool IsOrderKept(const std::vector<long> &sites) {
  auto ordered_sites = sites;
  std::sort(ordered_sites.begin(), ordered_sites.end());
  for (std::size_t i = 0; i < sites.size(); ++i) {
    if (sites[i] != ordered_sites[i]) { return false; }
  }
  return true;
}

} /* gqmps2 */ 
#endif /* ifndef GQMPS2_MPS_MEASU_H */
