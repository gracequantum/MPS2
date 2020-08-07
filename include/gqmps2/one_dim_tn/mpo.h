// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-05 21:20
* 
* Description: GraceQ/MPS2 project. Matrix state operator (MPO).
*/

/**
@file mpo.h
@brief Matrix state operator (MPO).
*/
#ifndef GQMPS2_ONE_DIM_TN_MPO_H
#define GQMPS2_ONE_DIM_TN_MPO_H


#include "gqmps2/one_dim_tn/ten_vec.h"    // TenVec
#include "gqmps2/site_vec.h"              // SiteVec

#include <assert.h>       // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqmps2 {


/**
Matrix product operator(MPO) class.

@tparam TenT Type of the MPO local tensors.
*/
template <typename TenT>
class MPO : public TenVec<TenT> {
public:
  /**
  Create a MPO using the basic information of the system.

  @param site_vec The SiteVec of the system.

  @since version 0.2.0
  */
  MPO(const SiteVec &site_vec) : TenVec<TenT>(site_vec.size) {}
};
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_ONE_DIM_TN_MPO_H */
