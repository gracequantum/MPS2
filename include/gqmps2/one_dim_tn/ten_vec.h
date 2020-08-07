// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-06 10:04
* 
* Description: GraceQ/MPS2 project. A vector of tensors.
*/

/**
@file ten_vec.h
@brief A vector of tensors.
*/
#ifndef GQMPS2_ONE_DIM_TN_TEN_VEC_H
#define GQMPS2_ONE_DIM_TN_TEN_VEC_H


#include <vector>     // vector


namespace gqmps2 {


/**
A vector of tensors.

@tparam TenT The type of the element of a tensor vector.

@since version 0.2.0
*/
template <typename TenT>
class TenVec {
public:
  using TenPtrT = TenT *;
  using TenPtrToConstT = const TenT *;

  /**
  Create a tensor vector using a size.

  @param size The size of the tensor vector.
  
  @since version 0.2.0
  */
  TenVec(const size_t size) : pten_vec_(size, nullptr) {}

  /**
  Create a tensor vector using a standard vector of tensors.

  @param tens A standard vector of tensors.

  @since version 0.2.0
  */
  TenVec(const std::vector<TenT> &tens) {
    pten_vec_.reserve(tens.size());
    for (auto &ten : tens) {
      pten_vec_.emplace_back(new TenT(ten));
    }
  }

  /**
  Create a tensor vector using a standard vector of pointer-to-tensors.

  @param tens A standard vector of pointer-to-tensors.

  @since version 0.2.0
  */
  TenVec(const std::vector<TenPtrT> &ptens) {
    pten_vec_.reserve(ptens.size());
    for (auto pten : ptens) {
      pten_vec_.emplace_back(new TenT(*pten));
    }
  }

  /**
  Release the memory managed by this.

  @since version 0.2.0
  */
  ~TenVec(void) {
    for (auto pten : pten_vec_) {
      if (pten != nullptr) {
        delete pten;
      }
    }
  }

  /**
  Return the size of the tensor vector.
  
  @since version 0.2.0
  */
  size_t size(void) const { return pten_vec_.size(); }

  /**
  The direct access to manage the raw data. It is a very dangerous operation!

  @since version 0.2.0
  */
  std::vector<TenPtrT> &data(void) { return pten_vec_; }

  /**
  The read-only access to the raw data.

  @since version 0.2.0
  */
  std::vector<TenPtrToConstT> cdata(void) const {
    std::vector<TenPtrToConstT> cpten_vec;
    cpten_vec.reserve(pten_vec_.size());
    for (auto pten : pten_vec_) {
      cpten_vec.emplace_back(pten);
    }
    return cpten_vec; 
  }

  /**
  The read-only access to an element.

  @since version 0.2.0
  */
  TenPtrToConstT operator()(size_t n) const { return pten_vec_[n]; }

  /**
  The access to an element.

  @since version 0.2.0
  */
  TenPtrT &operator()(size_t n) { return pten_vec_[n]; }

private:
  std::vector<TenPtrT> pten_vec_;
};
} /* gqmps2 */
#endif /* ifndef GQMPS2_ONE_DIM_TN_TEN_VEC_H */
