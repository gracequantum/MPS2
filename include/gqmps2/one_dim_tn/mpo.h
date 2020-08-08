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


#include <memory>         // shared_ptr, make_shared
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
class MPO {
public:
  /**
  Create a MPO using the basic information of the system.

  @param N The size of the MPO.

  @since version 0.2.0
  */
  MPO(const int N) :
      srdp_ten_vec_(std::make_shared<std::vector<TenT>>(N)) {}

  /**
  Get the size of the MPO.

  @since version 0.2.0
  */
  size_t size(void) const { return srdp_ten_vec_->size(); }

  /**
  MPO local tensor getter.

  @param i The site index.

  @since version 0.2.0
  */
  const TenT &operator[](const size_t i) const { return (*srdp_ten_vec_)[i]; }

  /**
  MPO local tensor setter.

  @param i The site index.

  @since version 0.2.0
  */
  TenT &operator[](const size_t i) { return (*srdp_ten_vec_)[i]; }

  /**
  Get the last local MPO tensor.

  @since version 0.2.0
  */
  TenT &back(void) { return srdp_ten_vec_->back(); }

  /// @copydoc MPO::back()
  const TenT &back(void) const { return srdp_ten_vec_->back(); }

private:
  std::shared_ptr<std::vector<TenT>> srdp_ten_vec_;
};
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_ONE_DIM_TN_MPO_H */
