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
#include "gqmps2/site_vec.h"    // SiteVec
#include "gqmps2/consts.h"    // kMpsPath
#include "gqmps2/utilities.h"     // IsPathExist, CreatPath
#include "gqten/gqten.h"    // SVD, Contract

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

@tparam TenElemT Element type of the local tensors.
@tparam QNT Quantum number type of the system.
*/
template <typename TenElemT, typename QNT>
class MPS : public TenVec<GQTensor<TenElemT, QNT>> {
public:
  using LocalTenT = GQTensor<TenElemT, QNT>;
  /**
  Create a empty MPS using its size.

  @param site_vec The sites information of the system.
  */
  MPS(
      const SiteVec<TenElemT, QNT> &site_vec
  ) : TenVec<LocalTenT>(site_vec.size),
      center_(kUncentralizedCenterIdx),
      tens_cano_type_(site_vec.size, MPSTenCanoType::NONE),
      site_vec_(site_vec) {}

  // MPS local tensor access.
  /**
  Access to local tensor. Set canonical type to NONE and MPS to uncentralized.

  @param idx Index of the MPS local tensor.
  */
  LocalTenT &operator[](const size_t idx) {
    tens_cano_type_[idx] = MPSTenCanoType::NONE;
    center_ = kUncentralizedCenterIdx;
    return DuoVector<LocalTenT>::operator[](idx);
  }

  /**
  Read-only access to local tensor.

  @param idx Index of the MPS local tensor.
  */
  const LocalTenT &operator[](const size_t idx) const {
    return DuoVector<LocalTenT>::operator[](idx); 
  }

  /**
  Access to the pointer to local tensor. Set canonical type to NONE and MPS to
  uncentralized.

  @param idx Index of the MPS local tensor.
  */
  LocalTenT * &operator()(const size_t idx) {
    tens_cano_type_[idx] = MPSTenCanoType::NONE;
    center_ = kUncentralizedCenterIdx; 
    return DuoVector<LocalTenT>::operator()(idx);
  }

  /**
  Read-only access to the pointer to local tensor.

  @param idx Index of the MPS local tensor.
  */
  const LocalTenT *operator()(const size_t idx) const {
    return DuoVector<LocalTenT>::operator()(idx);
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

  /**
  Get sites information.
  */
  SiteVec<TenElemT, QNT> GetSitesInfo(void) const {
    return site_vec_;
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
  SiteVec<TenElemT, QNT> site_vec_;

  void LeftCanonicalize_(const size_t);
  void LeftCanonicalizeTen_(const size_t);
  void RightCanonicalize_(const size_t);
  void RightCanonicalizeTen_(const size_t);
};


/**
Centralize the MPS.

@param target_center The new center of the MPS.
*/
template <typename TenElemT, typename QNT>
void MPS<TenElemT, QNT>::Centralize(const int target_center) {
  assert(target_center >= 0);
  auto mps_tail_idx = this->size() - 1; 
  if (target_center != 0) { LeftCanonicalize_(target_center - 1); }
  if (target_center != mps_tail_idx) {
    RightCanonicalize_(target_center + 1);
  }
  center_ = target_center;
}


template <typename TenElemT, typename QNT>
void MPS<TenElemT, QNT>::LeftCanonicalize_(const size_t stop_idx) {
  size_t start_idx;
  for (size_t i = 0; i <= stop_idx; ++i) {
    start_idx = i;
    if (tens_cano_type_[i] != MPSTenCanoType::LEFT) { break; }
    if (i == stop_idx) { return; }    // All related tensors are left canonical, do nothing.
  }
  for (size_t i = start_idx; i <= stop_idx; ++i) { LeftCanonicalizeTen_(i); }
}


template <typename TenElemT, typename QNT>
void MPS<TenElemT, QNT>::LeftCanonicalizeTen_(const size_t site_idx) {
  assert(site_idx < this->size() - 1);
  size_t ldims;
  if (site_idx == 0) {
    ldims = 1;
  } else {
    ldims = 2;
  }
  LocalTenT s, vt;
  auto pu = new LocalTenT;
  SVD((*this)(site_idx), ldims, Div((*this)[site_idx]), pu, &s, &vt);
  delete (*this)(site_idx);
  (*this)(site_idx) = pu;

  LocalTenT temp_ten;
  Contract(&s, &vt, {{1}, {0}}, &temp_ten);
  auto pnext_ten = new LocalTenT;
  Contract(&temp_ten, (*this)(site_idx+1), {{1}, {0}}, pnext_ten);
  delete (*this)(site_idx + 1);
  (*this)(site_idx + 1) = pnext_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::LEFT;
  tens_cano_type_[site_idx + 1] = MPSTenCanoType::NONE;
}


template <typename TenElemT, typename QNT>
void MPS<TenElemT, QNT>::RightCanonicalize_(const size_t stop_idx) {
  auto mps_tail_idx = this->size() - 1;
  size_t start_idx;
  for (size_t i = mps_tail_idx; i >= stop_idx; --i) {
    start_idx = i;
    if (tens_cano_type_[i] != MPSTenCanoType::RIGHT) { break; }
    if (i == stop_idx) { return; }    // All related tensors are right canonical, do nothing.
  }
  for (size_t i = start_idx; i >= stop_idx; --i) { RightCanonicalizeTen_(i); }
}


template <typename TenElemT, typename QNT>
void MPS<TenElemT, QNT>::RightCanonicalizeTen_(const size_t site_idx) {
  assert(site_idx > 0);
  size_t ldims = 1;
  LocalTenT u, s;
  auto pvt = new LocalTenT;
  auto qndiv = Div((*this)[site_idx]);
  SVD((*this)(site_idx), ldims, qndiv - qndiv, &u, &s, pvt);
  delete (*this)(site_idx);
  (*this)(site_idx) = pvt;

  LocalTenT temp_ten;
  Contract(&u, &s, {{1}, {0}}, &temp_ten);
  std::vector<size_t> ta_ctrct_axes;
  if ((site_idx - 1) == 0) {
    ta_ctrct_axes = {1};
  } else {
    ta_ctrct_axes = {2};
  }
  std::vector<std::vector<size_t>> ctrct_axes;
  ctrct_axes.emplace_back(ta_ctrct_axes);
  ctrct_axes.push_back({0});
  auto pprev_ten = new LocalTenT;
  Contract((*this)(site_idx - 1), &temp_ten, ctrct_axes, pprev_ten);
  delete (*this)(site_idx - 1);
  (*this)(site_idx - 1) = pprev_ten;

  tens_cano_type_[site_idx] = MPSTenCanoType::RIGHT;
  tens_cano_type_[site_idx - 1] = MPSTenCanoType::NONE;
}
} /* gqmps2 */
#endif /* ifndef GQMPS2_ONE_DIM_TN_MPS_MPS_H */
