// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-01-02 10:22
*
* Description: GraceQ/MPS2 project. The finite matrix product state class.
*/

/**
@file finite_mps.h
@brief The finite matrix product state class.
*/
#ifndef GQMPS2_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_H
#define GQMPS2_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_H


#include "gqmps2/one_dim_tn/mps/mps.h"    // MPS
#include "gqten/gqten.h"                  // SVD, Contract

#include <vector>     // vector
#include <iomanip>    // fix, scientific, setw

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqmps2 {
using namespace gqten;


const int kUncentralizedCenterIdx = -1;


/// Canonical type of a MPS local tensor.
enum MPSTenCanoType {
  NONE,   ///< Not a canonical MPS tensor.
  LEFT,   ///< Left canonical MPS tensor.
  RIGHT   ///< Right canonical MPS tensor.
};


/**
The finite matrix product state class.

@tparam TenElemT Element type of the local tensors.
@tparam QNT Quantum number type of the system.
*/
template <typename TenElemT, typename QNT>
class FiniteMPS : public MPS<TenElemT, QNT> {
public:
  using LocalTenT = typename MPS<TenElemT, QNT>::LocalTenT;

  /**
  Create a empty finite MPS using system information.

  @param site_vec The sites information of the system.
  */
  FiniteMPS(const SiteVec<TenElemT, QNT> &site_vec) :
      MPS<TenElemT, QNT>(site_vec),
      center_(kUncentralizedCenterIdx),
      tens_cano_type_(site_vec.size, MPSTenCanoType::NONE) {}

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

private:
  int center_;
  std::vector<MPSTenCanoType> tens_cano_type_;

  void LeftCanonicalize_(const size_t);
  void LeftCanonicalizeTen_(const size_t);
  void RightCanonicalize_(const size_t);
  void RightCanonicalizeTen_(const size_t);
};


/**
Centralize the finite MPS.

@param target_center The new center of the finite MPS.
*/
template <typename TenElemT, typename QNT>
void FiniteMPS<TenElemT, QNT>::Centralize(const int target_center) {
  assert(target_center >= 0);
  auto mps_tail_idx = this->size() - 1;
  if (target_center != 0) { LeftCanonicalize_(target_center - 1); }
  if (target_center != mps_tail_idx) {
    RightCanonicalize_(target_center + 1);
  }
  center_ = target_center;
}


template <typename TenElemT, typename QNT>
void FiniteMPS<TenElemT, QNT>::LeftCanonicalize_(const size_t stop_idx) {
  size_t start_idx;
  for (size_t i = 0; i <= stop_idx; ++i) {
    start_idx = i;
    if (tens_cano_type_[i] != MPSTenCanoType::LEFT) { break; }
    if (i == stop_idx) { return; }    // All related tensors are left canonical, do nothing.
  }
  for (size_t i = start_idx; i <= stop_idx; ++i) { LeftCanonicalizeTen_(i); }
}


template <typename TenElemT, typename QNT>
void FiniteMPS<TenElemT, QNT>::LeftCanonicalizeTen_(const size_t site_idx) {
  assert(site_idx < this->size() - 1);
  size_t ldims;
  if (site_idx == 0) {
    ldims = 1;
  } else {
    ldims = 2;
  }
  GQTensor<GQTEN_Double, QNT> s;
  LocalTenT vt;
  auto pu = new LocalTenT;
  mock_gqten::SVD((*this)(site_idx), ldims, Div((*this)[site_idx]), pu, &s, &vt);
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
void FiniteMPS<TenElemT, QNT>::RightCanonicalize_(const size_t stop_idx) {
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
void FiniteMPS<TenElemT, QNT>::RightCanonicalizeTen_(const size_t site_idx) {
  assert(site_idx > 0);
  size_t ldims = 1;
  LocalTenT u;
  GQTensor<GQTEN_Double, QNT> s;
  auto pvt = new LocalTenT;
  auto qndiv = Div((*this)[site_idx]);
  mock_gqten::SVD((*this)(site_idx), ldims, qndiv - qndiv, &u, &s, pvt);
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


// Non-member function for finite MPS
/**
Truncate the finite MPS. First centralize the MPS to left-end and normalize the left-end
MPS local tensor, then truncate each site using SVD from left to right. The S
tensor generated from each SVD step will be normalized.

@param mps To-be truncated finite MPS.
@param trunc_err The target truncation error.
@param Dmin The 
*/
template <typename TenElemT, typename QNT>
void TruncateMPS(
    FiniteMPS<TenElemT, QNT> &mps,
    const GQTEN_Double trunc_err,
    const size_t Dmin,
    const size_t Dmax
) {
  auto mps_size = mps.size();
  assert(mps_size >= 2);

  mps.Centralize(0);
  mps[0].Normalize();

  using LocalTenT = GQTensor<TenElemT, QNT>;
  GQTEN_Double actual_trunc_err;
  size_t D;
  for (size_t i = 0; i < mps_size - 1; ++i) {
    size_t ldims;
    if (i == 0) {
      ldims = 1;
    } else {
      ldims = 2;
    }
    GQTensor<GQTEN_Double, QNT> s;
    LocalTenT vt;
    auto pu = new LocalTenT;
    SVD(
        mps(i),
        ldims, Div(mps[i]), trunc_err, Dmin, Dmax,
        pu, &s, &vt, &actual_trunc_err, &D
    );
    std::cout << "Truncate MPS bond " << std::setw(4) << i
              << " TruncErr = " << std::setprecision(2) << std::scientific << actual_trunc_err << std::fixed
              << " D = " << std::setw(5) << D;
    std::cout << std::scientific << std::endl;
    s.Normalize();
    delete mps(i);
    mps(i) = pu;

    LocalTenT temp_ten;
    Contract(&s, &vt, {{1}, {0}}, &temp_ten);
    auto pnext_ten = new LocalTenT;
    Contract(&temp_ten, mps(i + 1), {{1}, {0}}, pnext_ten);
    delete mps(i + 1);
    mps(i + 1) = pnext_ten;
  }
}
} /* gqmps2 */
#endif /* ifndef GQMPS2_ONE_DIM_TN_MPS_FINITE_MPS_FINITE_MPS_H */
