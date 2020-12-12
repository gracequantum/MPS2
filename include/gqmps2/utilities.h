// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-08 16:02
* 
* Description: GraceQ/MPS2 project. Utility functions used by GraceQ/MPS2.
*/
#ifndef GQMPS2_UTILITIES_H
#define GQMPS2_UTILITIES_H


#include "gqten/gqten.h"    // bfread, bfwrite

#include <iostream>
#include <fstream>          // ifstream, ofstream

#include <sys/stat.h>       // stat, mkdir, S_IRWXU, S_IRWXG, S_IROTH, S_IXOTH


namespace gqmps2 {
using namespace gqten;


template <typename TenElemT, typename QNT>
void SVD(
    const GQTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    GQTensor<TenElemT, QNT> *pu,
    GQTensor<GQTEN_Double, QNT> *ps,
    GQTensor<TenElemT, QNT> *pvt
) {
  auto t_shape = pt->GetShape();
  size_t lsize = 1;
  size_t rsize = 1;
  for (size_t i = 0; i < pt->Rank(); ++i) {
    if (i < ldims) {
      lsize *= t_shape[i];
    } else {
      rsize *= t_shape[i];
    }
  }
  auto D = ((lsize >= rsize) ? lsize : rsize);
  GQTEN_Double actual_trunc_err;
  size_t actual_bond_dim;
  SVD(
      pt,
      ldims, lqndiv, 0, D, D,
      pu, ps, pvt, &actual_trunc_err, &actual_bond_dim
  );
}


template <typename TenType>
inline void WriteGQTensorTOFile(const TenType &t, const std::string &file) {
  std::ofstream ofs(file, std::ofstream::binary);
  bfwrite(ofs, t);
  ofs.close();
}


template <typename TenType>
inline void ReadGQTensorFromFile(TenType * &rpt, const std::string &file) {
  std::ifstream ifs(file, std::ifstream::binary);
  rpt = new TenType();
  bfread(ifs, *rpt);
  ifs.close();
}


inline bool IsPathExist(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}


inline void CreatPath(const std::string &path) {
  const int dir_err = mkdir(
                          path.c_str(),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH
                      );
  if (dir_err == -1) {
    std::cout << "error creating directory!" << std::endl;
    exit(1);
  }
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_UTILITIES_H */
