// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-10 11:32
* 
* Description: GraceQ/MPS2 project. The matrix product state (MPS) class.
*/

/**
@file mps.h
@brief The matrix product state (MPS) class.
*/
#ifndef GQMPS2_ONE_DIM_TN_MPS_MPS_H
#define GQMPS2_ONE_DIM_TN_MPS_MPS_H


#include <vector>     // vector


namespace gqmps2 {


template <typename TenType>
struct MPS {
  MPS(std::vector<TenType *> &tens, const long center) :
      tens(tens), center(center), N(tens.size()) {}
  
  std::vector<TenType *> &tens; 
  long center;
  std::size_t N;
};
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_ONE_DIM_TN_MPS_MPS_H */
