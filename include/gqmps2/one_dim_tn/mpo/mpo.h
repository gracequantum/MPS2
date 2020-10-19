// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-10-19 14:19
*
* Description: GraceQ/MPS2 project. Matrix product operator (MPO).
*/

/**
@file mpo.h
@brief Matrix product operator (MPO) class.
*/
#ifndef GQMPS2_ONE_DIM_TN_MPO_MPO_H
#define GQMPS2_ONE_DIM_TN_MPO_MPO_H


#include "gqmps2/one_dim_tn/framework/ten_vec.h"    // TenVec


namespace gqmps2 {
using namespace gqten;

template <typename LocalTenT>
using MPO = TenVec<LocalTenT>;
} /* gqmps2 */
#endif /* ifndef GQMPS2_ONE_DIM_TN_MPO_MPO_H */
