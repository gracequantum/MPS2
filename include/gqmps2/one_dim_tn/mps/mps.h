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


#include "gqmps2/one_dim_tn/framework/ten_vec.h"    // TenVec
#include "gqmps2/consts.h"    // kMpsPath
#include "gqmps2/utilities.h"     // IsPathExist, CreatPath
#include "gqten/gqten.h"    // Svd, Contract

#include <vector>     // vector

#include <assert.h>     // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqmps2 {
using namespace gqten;


const int kUncentralizedCenterIdx = -1;


/// Canonical type of a MPS local tensor.
enum MPSTenCanoType {
  NONE,   ///< Not a canonical MPS tensor.
  LEFT,   ///< Left canonical MPS tensor.
  RIGHT   ///< Right canonical MPS tensor.
};


// Helpers
std::string GenMPSTenName(const std::string &mps_path, const size_t idx) {
  return mps_path + "/" +
         kMpsTenBaseName + std::to_string(idx) + "." + kGQTenFileSuffix;
}


/**
The matrix product state (MPS) class.

@tparam ElemT Type of the MPS local tensors.
*/
template <typename ElemT>
class MPS : public TenVec<ElemT> {
public:
  /**
  Create a empty MPS using its size.

  @param size The size of the MPS.
  */
  MPS(
      const size_t size
  ) : TenVec<ElemT>(size),
      center_(kUncentralizedCenterIdx),
      tens_cano_type_(size, MPSTenCanoType::NONE) {}

  // MPS local tensor access.
  /**
  Access to local tensor. Set canonical type to NONE and MPS to uncentralized.

  @param idx Index of the MPS local tensor.
  */
  ElemT &operator[](const size_t idx) {
    tens_cano_type_[idx] = MPSTenCanoType::NONE;
    center_ = kUncentralizedCenterIdx;
    return DuoVector<ElemT>::operator[](idx);
  }

  /**
  Read-only access to local tensor.

  @param idx Index of the MPS local tensor.
  */
  const ElemT &operator[](const size_t idx) const {
    return DuoVector<ElemT>::operator[](idx); 
  }

  /**
  Access to the pointer to local tensor. Set canonical type to NONE and MPS to
  uncentralized.

  @param idx Index of the MPS local tensor.
  */
  ElemT * &operator()(const size_t idx) {
    tens_cano_type_[idx] = MPSTenCanoType::NONE;
    center_ = kUncentralizedCenterIdx; 
    return DuoVector<ElemT>::operator()(idx);
  }

  /**
  Read-only access to the pointer to local tensor.

  @param idx Index of the MPS local tensor.
  */
  const ElemT *operator()(const size_t idx) const {
    return DuoVector<ElemT>::operator()(idx);
  }

  // MPS global operations.
  void Centralize(const int);

  // Properties getter.
  /**
  Get the center of the MPS.
  */
  int GetCenter(void) const { return center_; }

  /**
  Get the canonical type of all of the MPS local tensors.
  */
  std::vector<MPSTenCanoType> GetTensCanoType(void) const {
    return tens_cano_type_;
  }

  /**
  Get the canonical type of a MPS local tensor.

  @param idx Index of the MPS local tensor.
  */
  MPSTenCanoType GetTenCanoType(const size_t idx) const {
    return tens_cano_type_[idx];
  }

  // HDD I/O
  /**
  Dump MPS to HDD.

  @param mps_path Path to the MPS directory.
  */
  void Dump(const std::string &mps_path = kMpsPath) const {
    if (!IsPathExist(mps_path)) { CreatPath(mps_path); }
    std::string file;
    for (size_t i = 0; i < this->size(); ++i) {
      file = GenMPSTenName(mps_path, i);
      this->DumpTen(i, file);
    }
  }

  /**
  Load MPS from HDD.

  @param mps_path Path to the MPS directory.
  */
  void Load(const std::string &mps_path = kMpsPath) {
    std::string file;
    for (size_t i = 0; i < this->size(); ++i) {
      file = GenMPSTenName(mps_path, i);
      this->LoadTen(i, file);
    }
  }

private:
  int center_;
  std::vector<MPSTenCanoType> tens_cano_type_;

  void LeftCanonicalize_(const size_t);
  void LeftCanonicalizeTen_(const size_t);
  void RightCanonicalize_(const size_t);
  void RightCanonicalizeTen_(const size_t);
};


/**
Centralize the MPS.

@param target_center The new center of the MPS.
*/
template <typename ElemT>
void MPS<ElemT>::Centralize(const int target_center) {
  assert(target_center >= 0);
  auto mps_tail_idx = this->size() - 1; 
  if (target_center != 0) { LeftCanonicalize_(target_center - 1); }
  if (target_center != mps_tail_idx) {
    RightCanonicalize_(target_center + 1);
  }
  center_ = target_center;
}


template <typename ElemT>
void MPS<ElemT>::LeftCanonicalize_(const size_t stop_idx) {
  size_t start_idx;
  for (size_t i = 0; i <= stop_idx; ++i) {
    start_idx = i;
    if (tens_cano_type_[i] != MPSTenCanoType::LEFT) { break; }
    if (i == stop_idx) { return; }    // All related tensors are left canonical, do nothing.
  }
  for (size_t i = start_idx; i <= stop_idx; ++i) { LeftCanonicalizeTen_(i); }
}


template <typename ElemT>
void MPS<ElemT>::LeftCanonicalizeTen_(const size_t site_idx) {
  assert(site_idx < this->size() - 1);
  long ldims, rdims;
  if (site_idx == 0) {
    ldims = 1;
    rdims = 1;
  } else {
    ldims = 2;
    rdims = 1;
  }
  auto svd_res = Svd(
      (*this)[site_idx],
      ldims, rdims,
      Div((*this)[site_idx]), Div((*this)[site_idx+1])
  );
  delete (*this)(site_idx);
  (*this)(site_idx) = svd_res.u;
  auto temp_ten = Contract(*svd_res.s, *svd_res.v, {{1}, {0}});
  delete svd_res.s;
  delete svd_res.v;
  auto next_ten = Contract(*temp_ten, (*this)[site_idx+1], {{1}, {0}});
  delete temp_ten;
  delete (*this)(site_idx + 1);
  (*this)(site_idx + 1) = next_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::LEFT;
  tens_cano_type_[site_idx + 1] = MPSTenCanoType::NONE;
}


template <typename ElemT>
void MPS<ElemT>::RightCanonicalize_(const size_t stop_idx) {
  auto mps_tail_idx = this->size() - 1;
  size_t start_idx;
  for (size_t i = mps_tail_idx; i >= stop_idx; --i) {
    start_idx = i;
    if (tens_cano_type_[i] != MPSTenCanoType::RIGHT) { break; }
    if (i == stop_idx) { return; }    // All related tensors are right canonical, do nothing.
  }
  for (size_t i = start_idx; i >= stop_idx; --i) { RightCanonicalizeTen_(i); }
}


template <typename ElemT>
void MPS<ElemT>::RightCanonicalizeTen_(const size_t site_idx) {
  assert(site_idx > 0);
  long ldims, rdims;
  if (site_idx == this->size() - 1) {
    ldims = 1;
    rdims = 1;
  } else {
    ldims = 1;
    rdims = 2;
  }
  auto svd_res = Svd(
      (*this)[site_idx],
      ldims, rdims,
      Div((*this)[site_idx - 1]), Div((*this)[site_idx])
  );
  delete (*this)(site_idx);
  (*this)(site_idx) = svd_res.v;
  auto temp_ten = Contract(*svd_res.u, *svd_res.s, {{1}, {0}});
  delete svd_res.u;
  delete svd_res.s;
  std::vector<long> ta_ctrct_axes;
  if ((site_idx - 1) == 0) {
    ta_ctrct_axes = {1};
  } else {
    ta_ctrct_axes = {2};
  }
  auto prev_ten = Contract(
                      (*this)[site_idx - 1], *temp_ten,
                      {ta_ctrct_axes, {0}}
                  );
  delete temp_ten;
  delete (*this)(site_idx - 1);
  (*this)(site_idx - 1) = prev_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::RIGHT;
  tens_cano_type_[site_idx - 1] = MPSTenCanoType::NONE;
}
} /* gqmps2 */
#endif /* ifndef GQMPS2_ONE_DIM_TN_MPS_MPS_H */
