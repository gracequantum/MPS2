/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 12:15
* 
* Description: GraceQ/mps2 project. Private objects for two sites algorithm.
*/
#ifndef GQMPS2_TWO_SITE_ALGO_H
#define GQMPS2_TWO_SITE_ALGO_H


#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <vector>
#include <cmath>
#include <cstdio>


namespace gqmps2 {
using namespace gqten;


std::pair<std::vector<GQTensor *>, std::vector<GQTensor *>> InitBlocks(
    const std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    const SweepParams &);

double TwoSiteSweep(
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    std::vector<GQTensor *> &, std::vector<GQTensor *> &,
    const SweepParams &);

double TwoSiteUpdate(
    const long,
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    std::vector<GQTensor *> &, std::vector<GQTensor *> &,
    const SweepParams &, const char);


// Helpers.
inline double MeasureEE(const GQTensor *s, const long sdim) {
  double ee = 0;
  double p;
  for (long i = 0; i < sdim; ++i) {
    p = std::pow(s->Elem({i, i}), 2.0);
    ee += -p * std::log(p);
  }
  return ee;
}


inline void RemoveFile(const std::string &file) {
  if (std::remove(file.c_str())) {
    std::cout << "Unable to delete " << file << std::endl;
    exit(1);
  }
}


inline std::string GenBlockFileName(
    const std::string &dir, const long blk_len) {
  return kRuntimeTempPath + "/" +
         dir + kBlockFileBaseName + std::to_string(blk_len) +
         "." + kGQTenFileSuffix;
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_TWO_SITE_ALGO_H */
