// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-10-29 15:35
* 
* Description: GraceQ/MPS2 project. Algebra of MPO's coefficient and operator.
*/
#ifndef GQMPS2_ONE_DIM_TN_MPO_MPOGEN_COEF_OP_ALG_H
#define GQMPS2_ONE_DIM_TN_MPO_MPOGEN_COEF_OP_ALG_H

#include "gqmps2/one_dim_tn/mpo/mpogen/symb_alg/sparse_mat.h"

#include <vector>
#include <algorithm>
#include <iostream>

#include <assert.h>

#ifdef Release
  #define NDEBUG
#endif


// Forward declarations.
template <typename T>
bool ElemInVec(const T &, const std::vector<T> &, long &pos);

template <typename VecT>
VecT ConcatenateTwoVec(const VecT &, const VecT &);


// Label of coefficient.
using CoefLabel = long;

const CoefLabel kIdCoefLabel = 0;     // Coefficient label for identity 1.


// Representation of coefficient.
class CoefRepr {

public:
  CoefRepr(void) : coef_label_list_() {}

  CoefRepr(const CoefLabel coef_label) {
    coef_label_list_.push_back(coef_label); 
  }

  CoefRepr(const std::vector<CoefLabel> &coef_label_list) :
      coef_label_list_(coef_label_list) {}

  CoefRepr(const CoefRepr &coef_repr) :
      coef_label_list_(coef_repr.coef_label_list_) {}

  CoefRepr &operator=(const CoefRepr &rhs) {
    coef_label_list_ = rhs.coef_label_list_;
    return *this;
  }

  std::vector<CoefLabel> GetCoefLabelList(void) const {
    return coef_label_list_; 
  }

  bool operator==(const CoefRepr &rhs) const {
    const std::vector<CoefLabel> &rc_lhs_coef_label_list = coef_label_list_;
    std::vector<CoefLabel> rhs_coef_label_list = rhs.coef_label_list_;
    if (rc_lhs_coef_label_list.size() != rhs_coef_label_list.size()) {
      return false;
    }
    for (auto &l_elem : rc_lhs_coef_label_list) {
      long pos_in_rhs = -1;
      if (!ElemInVec(l_elem, rhs_coef_label_list, pos_in_rhs)) {
        return false;
      } else {
        rhs_coef_label_list.erase(rhs_coef_label_list.begin() + pos_in_rhs);
      }
    }
    if (!rhs_coef_label_list.empty()) { return false; }
    return true;
  }

  bool operator!=(const CoefRepr &rhs) const {
    return !(*this == rhs); 
  }

  CoefRepr operator+(const CoefRepr &rhs) const {
    return CoefRepr(ConcatenateTwoVec(coef_label_list_, rhs.coef_label_list_));
  }

  template <typename CoefT>
  CoefT Realize(const std::vector<CoefT> &label_coef_mapping) {
    CoefT coef = 0;
    for (auto coef_label : coef_label_list_) {
      coef += label_coef_mapping[coef_label];
    }
    return coef;
  }

private:
  std::vector<CoefLabel> coef_label_list_;
};


const CoefRepr kNullCoefRepr = CoefRepr();            // Coefficient representation for null coefficient.
const CoefRepr kIdCoefRepr = CoefRepr(kIdCoefLabel);  // Coefficient representation for identity coefficient 1.

using CoefReprVec = std::vector<CoefRepr>;


// Label of operator.
using OpLabel = long;

const OpLabel kIdOpLabel = 0;         // Coefficient label for identity id.


// Representation of operator.
class SparOpReprMat;    // Forward declaration.

class OpRepr {
friend std::pair<CoefRepr, OpRepr> SeparateCoefAndBase(const OpRepr &);
friend OpRepr CoefReprOpReprIncompleteMulti(const CoefRepr &, const OpRepr &);
friend std::vector<OpRepr> CalcSparOpReprMatRowLinCmb(
    const SparOpReprMat &, const CoefReprVec &);
friend std::vector<OpRepr> CalcSparOpReprMatColLinCmb(
    const SparOpReprMat &, const CoefReprVec &);

public:
  OpRepr(void) : coef_repr_list_(), op_label_list_() {}

  OpRepr(const OpLabel op_label) {
    coef_repr_list_.push_back(kIdCoefRepr);
    op_label_list_.push_back(op_label);
  }

  OpRepr(const CoefRepr &coef_repr, const OpLabel op_label) {
    coef_repr_list_.push_back(coef_repr);
    op_label_list_.push_back(op_label);
  }

  OpRepr(
      const std::vector<CoefRepr> &coef_reprs,
      const std::vector<OpLabel> &op_labels) {
    for (size_t i = 0; i < op_labels.size(); ++i) {
      auto poss_it = std::find(op_label_list_.cbegin(), op_label_list_.cend(),
                               op_labels[i]);
      if (poss_it == op_label_list_.cend()) {
        coef_repr_list_.push_back(coef_reprs[i]);
        op_label_list_.push_back(op_labels[i]);
      } else {
        size_t idx = poss_it - op_label_list_.cbegin();
        coef_repr_list_[idx] = coef_repr_list_[idx] + coef_reprs[i];
      }
    }
    assert(coef_repr_list_.size() == op_label_list_.size());
  }

  OpRepr(const std::vector<OpLabel> &op_labels) :
      OpRepr(CoefReprVec(op_labels.size(), kIdCoefRepr), op_labels) {}

  std::vector<CoefRepr> GetCoefReprList(void) const {
    return coef_repr_list_; 
  }

  std::vector<OpLabel> GetOpLabelList(void) const {
    return op_label_list_; 
  }

  bool operator==(const OpRepr &rhs) const {
    const std::vector<CoefRepr> &rc_lhs_coef_repr_list = coef_repr_list_;
    const std::vector<OpLabel> &rc_lhs_op_label_list = op_label_list_;
    if (rc_lhs_op_label_list.size() != rhs.op_label_list_.size()) {
      return false;
    }
    std::vector<CoefRepr> rhs_coef_repr_list = rhs.coef_repr_list_;
    std::vector<OpLabel> rhs_op_label_list = rhs.op_label_list_;
    for (size_t i = 0; i < rc_lhs_op_label_list.size(); ++i) {
      long pos_in_rhs = -1;
      if (!ElemInVec(
               rc_lhs_op_label_list[i], rhs_op_label_list, pos_in_rhs)) {
        return false;
      } else if (rc_lhs_coef_repr_list[i] != rhs_coef_repr_list[pos_in_rhs]) {
        return false;
      } else {
        rhs_coef_repr_list.erase(rhs_coef_repr_list.begin() + pos_in_rhs);
        rhs_op_label_list.erase(rhs_op_label_list.begin() + pos_in_rhs);
      }
    }
    if (!rhs_op_label_list.empty()) { return false; }
    return true;
  }

  bool operator!=(const OpRepr &rhs) const {
    return !(*this == rhs);
  }

  OpRepr operator+(const OpRepr &rhs) const {
    auto coef_repr_list = coef_repr_list_;
    auto op_label_list = op_label_list_;
    auto rhs_size = rhs.coef_repr_list_.size();
    for (size_t i = 0; i < rhs_size; ++i) {
      auto poss_it = std::find(op_label_list.cbegin(), op_label_list.cend(),
                               rhs.op_label_list_[i]);
      if (poss_it == op_label_list.cend()) {
        coef_repr_list.push_back(rhs.coef_repr_list_[i]);
        op_label_list.push_back(rhs.op_label_list_[i]);
      } else {
        size_t idx = poss_it - op_label_list.cbegin();
        coef_repr_list[idx] = coef_repr_list[idx] + rhs.coef_repr_list_[i];
      }
    }
    return OpRepr(coef_repr_list, op_label_list);
  }

  template<typename CoefT, typename OpT>
  OpT Realize(
      const std::vector<CoefT> &label_coef_mapping,
      const std::vector<OpT> &label_op_mapping) {
    auto base_op_num = op_label_list_.size();
    assert(base_op_num == coef_repr_list_.size());
    OpT op;
    if (base_op_num == 0) {
      return OpT();
    } else if (base_op_num == 1) {
      return coef_repr_list_[0].Realize(label_coef_mapping) *
             label_op_mapping[op_label_list_[0]];
    } else {
      op = coef_repr_list_[0].Realize(label_coef_mapping) *
           label_op_mapping[op_label_list_[0]];
      for (size_t i = 1; i < base_op_num; ++i) {
        op += coef_repr_list_[i].Realize(label_coef_mapping) *
              label_op_mapping[op_label_list_[i]];
      }
    }
    return op;
  }

private:
  std::vector<CoefRepr> coef_repr_list_;
  std::vector<OpLabel> op_label_list_;
};

const OpRepr kNullOpRepr = OpRepr();          // Operator representation for null operator.
const OpRepr kIdOpRepr = OpRepr(kIdOpLabel);  // Operator representation for identity operator.

using OpReprVec = std::vector<OpRepr>;


inline std::pair<CoefRepr, OpRepr> SeparateCoefAndBase(const OpRepr &op_repr) {
  auto term_num = op_repr.coef_repr_list_.size();
  if (term_num == 0) {
    return std::make_pair(kNullCoefRepr, kNullOpRepr);
  } else if (term_num == 1) {
    return std::make_pair(op_repr.coef_repr_list_[0],
                          OpRepr(op_repr.op_label_list_[0]));
  } else {
    auto coef =  op_repr.coef_repr_list_[0];
    for (size_t i = 1; i < term_num; ++i) {
      auto coef1 = op_repr.coef_repr_list_[i];
      if (coef1 != coef) {
        return std::make_pair(kIdCoefRepr, OpRepr(op_repr));
      }
    }
    return std::make_pair(coef, OpRepr(op_repr.op_label_list_));
  }
}


inline CoefRepr GetOpReprCoef(const OpRepr &op_repr) {
  return SeparateCoefAndBase(op_repr).first;
}


// Sparse coefficient representation matrix.
using SparCoefReprMat = SparMat<CoefRepr>;


// Sparse operator representation matrix.
using SparOpReprMatBase = SparMat<OpRepr>;

class SparOpReprMat : public SparOpReprMatBase {
public:
  SparOpReprMat(void) : SparOpReprMatBase() {}

  SparOpReprMat(const size_t row_num, const size_t col_num) :
      SparOpReprMatBase(row_num, col_num) {}

  SparOpReprMat(const SparOpReprMat &spar_mat) :
      SparOpReprMatBase(spar_mat) {}

  SparOpReprMat &operator=(const SparOpReprMat &spar_mat) {
    rows = spar_mat.rows;
    cols = spar_mat.cols;
    data = spar_mat.data;
    indexes = spar_mat.indexes;
    return *this;
  }

  std::vector<size_t> SortRows(void) {
    auto mapping = GenSortRowsMapping_();
    std::sort(mapping.begin(), mapping.end());
    std::vector<size_t> sorted_row_idxs(rows);
    for (size_t i = 0; i < rows; ++i) {
      sorted_row_idxs[i] = mapping[i].second;
    }
    TransposeRows(sorted_row_idxs);
    return sorted_row_idxs;
  }

  std::vector<size_t> SortCols(void) {
    auto mapping = GenSortColsMapping_();
    std::sort(mapping.begin(), mapping.end());
    std::vector<size_t> sorted_col_idxs(cols);
    for (size_t i = 0; i < cols; ++i) {
      sorted_col_idxs[i] = mapping[i].second;
    }
    TransposeCols(sorted_col_idxs);
    return sorted_col_idxs;
  }

  CoefRepr CalcRowCoef(const size_t row_idx) {
    std::vector<CoefRepr> nonull_op_repr_coefs;
    for (size_t y = 0; y < cols; ++y) {
      if (indexes[CalcOffset(row_idx, y)] != -1) {
        nonull_op_repr_coefs.push_back(GetOpReprCoef((*this)(row_idx, y)));
      }
    }
    if (nonull_op_repr_coefs.size() == 0) {
      return kNullCoefRepr;
    } else {
      auto coef = nonull_op_repr_coefs[0];
      for (auto &coef1 : nonull_op_repr_coefs) {
        if (coef1 != coef) {
          return kIdCoefRepr;
        }
      }
      return coef;
    }
  }

  CoefRepr CalcColCoef(const size_t col_idx) {
    std::vector<CoefRepr> nonull_op_repr_coefs;
    for (size_t x = 0; x < rows; ++x) {
      if (indexes[CalcOffset(x, col_idx)] != -1) {
        nonull_op_repr_coefs.push_back(GetOpReprCoef((*this)(x, col_idx)));
      }
    }
    if (nonull_op_repr_coefs.size() == 0) {
      return kNullCoefRepr;
    } else {
      auto coef = nonull_op_repr_coefs[0];
      for (auto &coef1 : nonull_op_repr_coefs) {
        if (coef1 != coef) {
          return kIdCoefRepr;
        }
      }
      return coef;
    }
  }

  void RemoveRowCoef(const size_t row_idx) {
    for (size_t y = 0; y < cols; ++y) {
      if (indexes[CalcOffset(row_idx, y)] != -1) {
        auto elem = (*this)(row_idx, y);
        this->SetElem(row_idx, y, SeparateCoefAndBase(elem).second);
      }
    }
  }

  void RemoveColCoef(const size_t col_idx) {
    for (size_t x = 0; x < rows; ++x) {
      if (indexes[CalcOffset(x, col_idx)] != -1) {
        auto elem = (*this)(x, col_idx);
        this->SetElem(x, col_idx, SeparateCoefAndBase(elem).second);
      }
    }
  }

  CoefReprVec CalcRowLinCmb(const size_t row_idx) const {
    auto row = GetRow(row_idx);
    CoefReprVec cmb_coefs;
    for (size_t x = 0; x < row_idx; ++x) {
      cmb_coefs.push_back(CalcRowOverlap_(row, x));
    }
    return cmb_coefs;
  }

  CoefReprVec CalcColLinCmb(const size_t col_idx) const {
    auto col = GetCol(col_idx);
    CoefReprVec cmb_coefs;
    for (size_t y = 0; y < col_idx; ++y) {
      cmb_coefs.push_back(CalcColOverlap_(col, y));
    }
    return cmb_coefs;
  }

private:
  using SortMapping = std::vector<std::pair<size_t, size_t>>;   // # of no null : row_idx

  SortMapping GenSortRowsMapping_(void) const {
    SortMapping mapping;
    for (size_t x = 0; x < rows; ++x) {
      size_t nonull_elem_num = 0;
      for (size_t y = 0; y < cols; ++y) {
        if (indexes[CalcOffset(x, y)] != -1) { nonull_elem_num++; }
      }
      mapping.push_back(std::make_pair(nonull_elem_num, x));
    }
    return mapping;
  }

  SortMapping GenSortColsMapping_(void) const {
    SortMapping mapping;
    for (size_t y = 0; y < cols; ++y) {
      size_t nonull_elem_num = 0;
      for (size_t x = 0; x < rows; ++x) {
        if (indexes[CalcOffset(x, y)] != -1) { nonull_elem_num++; }
      }
      mapping.push_back(std::make_pair(nonull_elem_num, y));
    }
    return mapping;
  }

  CoefRepr CalcRowOverlap_(
      const std::vector<OpRepr> &row, const size_t tgt_row_idx) const {
    CoefReprVec poss_overlaps;
    for (size_t y = 0; y < cols; ++y) {
      if (indexes[CalcOffset(tgt_row_idx, y)] != -1) {
        auto tgt_op = row[y];
        auto base_op = (*this)(tgt_row_idx, y);
        if (tgt_op == base_op) {
          poss_overlaps.push_back(kIdCoefRepr);
        } else {
          auto tgt_coef_and_base_op = SeparateCoefAndBase(tgt_op);
          if (tgt_coef_and_base_op.second == base_op) {
            poss_overlaps.push_back(tgt_coef_and_base_op.first);
          } else {
            return kNullCoefRepr;
          }
        }
      }
    }
    if (poss_overlaps.empty()) { return kNullCoefRepr; }
    for (auto &poss_overlap : poss_overlaps) {
      if (poss_overlap != poss_overlaps[0]) {
        return kNullCoefRepr;
      }
    }
    return poss_overlaps[0];
  }

  CoefRepr CalcColOverlap_(
      const std::vector<OpRepr> &col, const size_t tgt_col_idx) const {
    CoefReprVec poss_overlaps;
    for (size_t x = 0; x < rows; ++x) {
      if (indexes[CalcOffset(x, tgt_col_idx)] != -1) {
        auto tgt_op = col[x];
        auto base_op = (*this)(x, tgt_col_idx);
        if (tgt_op == base_op) {
          poss_overlaps.push_back(kIdCoefRepr);
        } else {
          auto tgt_coef_and_base_op = SeparateCoefAndBase(tgt_op);
          if (tgt_coef_and_base_op.second == base_op) {
            poss_overlaps.push_back(tgt_coef_and_base_op.first);
          } else {
            return kNullCoefRepr;
          }
        }
      }
    }
    if (poss_overlaps.empty()) { return kNullCoefRepr; }
    for (auto &poss_overlap : poss_overlaps) {
      if (poss_overlap != poss_overlaps[0]) {
        return kNullCoefRepr;
      }
    }
    return poss_overlaps[0];
  }
};

using SparOpReprMatVec = std::vector<SparOpReprMat>;


// Incomplete multiplication for SparMat.
inline OpRepr CoefReprOpReprIncompleteMulti(const CoefRepr &coef, const OpRepr &op) {
  if (op == kNullOpRepr) { return kNullOpRepr; }
  if (coef == kIdCoefRepr) { return op; }
  for (auto &c : op.coef_repr_list_) {
    if (c != kIdCoefRepr) {
      std::cout << "CoefReprOpReprIncompleteMulti fail!" << std::endl;
      exit(1);
    }
  }
  CoefReprVec new_coefs(op.coef_repr_list_.size(), coef);
  return OpRepr(new_coefs, op.op_label_list_);
}


inline void SparCoefReprMatSparOpReprMatIncompleteMultiKernel(
    const SparCoefReprMat &coef_mat, const SparOpReprMat &op_mat,
    const size_t coef_mat_row_idx, const size_t op_mat_col_idx,
    SparOpReprMat &res) {
  OpRepr res_elem;
  for (size_t i = 0; i < coef_mat.cols; ++i) {
    if (coef_mat.indexes[coef_mat.CalcOffset(coef_mat_row_idx, i)] != -1 &&
        op_mat.indexes[op_mat.CalcOffset(i, op_mat_col_idx)] != -1) {
      res_elem = res_elem + CoefReprOpReprIncompleteMulti(
                                coef_mat(coef_mat_row_idx, i),
                                op_mat(i, op_mat_col_idx));
    }
  }
  if (res_elem != kNullOpRepr) {
    res.SetElem(coef_mat_row_idx, op_mat_col_idx, res_elem);
  }
}


inline void SparOpReprMatSparCoefReprMatIncompleteMultiKernel(
    const SparOpReprMat &op_mat, const SparCoefReprMat &coef_mat,
    const size_t op_mat_row_idx, const size_t coef_mat_col_idx,
    SparOpReprMat &res) {
  OpRepr res_elem;
  for (size_t i = 0; i < op_mat.cols; ++i) {
    if (op_mat.indexes[op_mat.CalcOffset(op_mat_row_idx, i)] != -1 &&
        coef_mat.indexes[coef_mat.CalcOffset(i, coef_mat_col_idx)] != -1) {
      res_elem = res_elem + CoefReprOpReprIncompleteMulti(
                                coef_mat(i, coef_mat_col_idx),
                                op_mat(op_mat_row_idx, i));
    }
  }
  if (res_elem != kNullOpRepr) {
    res.SetElem(op_mat_row_idx, coef_mat_col_idx, res_elem);
  }
}


inline SparOpReprMat SparCoefReprMatSparOpReprMatIncompleteMulti(
    const SparCoefReprMat &coef_mat, const SparOpReprMat &op_mat) {
  assert(coef_mat.cols == op_mat.rows);
  SparOpReprMat res(coef_mat.rows, op_mat.cols);
  for (size_t x = 0; x < coef_mat.rows; ++x) {
    for (size_t y = 0; y < op_mat.cols; ++y) {
      SparCoefReprMatSparOpReprMatIncompleteMultiKernel(
          coef_mat, op_mat, x, y, res);
    }
  }
  return res;
}


inline SparOpReprMat SparOpReprMatSparCoefReprMatIncompleteMulti(
    const SparOpReprMat &op_mat, const SparCoefReprMat &coef_mat) {
  assert(op_mat.cols == coef_mat.rows);
  SparOpReprMat res(op_mat.rows, coef_mat.cols);
  for (size_t x = 0; x < op_mat.rows; ++x) {
    for (size_t y = 0; y < coef_mat.cols; ++y) {
      SparOpReprMatSparCoefReprMatIncompleteMultiKernel(
          op_mat, coef_mat, x, y, res);
    }
  }
  return res;
}


// Row and column delinearization.
/* TODO: So bad implementation, need refactor. */
inline OpReprVec CalcSparOpReprMatRowLinCmb(
    const SparOpReprMat &m, const CoefReprVec &cmb) {
  auto work_row_num = cmb.size();
  assert(work_row_num > 0);
  auto res = OpReprVec(m.cols, kNullOpRepr);
  for (size_t i = 0; i < work_row_num; ++i) {
    auto cmb_coef = cmb[i];
    auto row = m.GetRow(i);
    if (cmb_coef == kIdCoefRepr) {
      for (size_t j = 0; j < m.cols; ++j) {
        res[j] = res[j] + row[j];
      }
    } else if (cmb_coef == kNullCoefRepr) {
      // Do nothing.
    } else {
      for (size_t j = 0; j < m.cols; ++j) {
        auto elem = row[j];
        for (auto &coef_repr : elem.coef_repr_list_) {
          if (coef_repr == kIdCoefRepr) {
            coef_repr = cmb_coef;
          } else {
            std::cout << "Unsupported operation!" << std::endl;
            exit(1);
          }
        }
        res[j] = res[j] + elem;
      }
    }
  }
  return res;
}


inline OpReprVec CalcSparOpReprMatColLinCmb(
    const SparOpReprMat &m, const CoefReprVec &cmb) {
  auto work_col_num = cmb.size();
  assert(work_col_num > 0);
  auto res = OpReprVec(m.rows, kNullOpRepr);
  for (size_t i = 0; i < work_col_num; ++i) {
    auto cmb_coef = cmb[i];
    auto col = m.GetCol(i);
    if (cmb_coef == kIdCoefRepr) {
      for (size_t j = 0; j < m.rows; ++j) {
        res[j] = res[j] + col[j];
      }
    } else if (cmb_coef == kNullCoefRepr) {
      // Do nothing.
    } else {
      for (size_t j = 0; j < m.rows; ++j) {
        auto elem = col[j];
        for (auto &coef_repr : elem.coef_repr_list_) {
          if (coef_repr == kIdCoefRepr) {
            coef_repr = cmb_coef;
          } else {
            std::cout << "Unsupported operation!" << std::endl;
            exit(1);
          }
        }
        res[j] = res[j] + elem;
      }
    }
  }
  return res;
}


inline void SparOpReprMatRowDelinearize(
    SparOpReprMat &target, SparOpReprMat &follower) {
  auto row_num = target.rows;
  size_t i;
  for (i = 1; i < row_num; ++i) {
    auto cmb = target.CalcRowLinCmb(i);
    if (CalcSparOpReprMatRowLinCmb(target, cmb) == target.GetRow(i)) {
      // Remove the row.
      target.RemoveRow(i);
      // Construct transform matrix.
      SparCoefReprMat trans_mat(row_num, row_num-1);
      for (size_t j = 0; j < i; ++j) {
        trans_mat.SetElem(j, j, kIdCoefRepr);
      }
      for (size_t j = 0; j < i; ++j) {
        trans_mat.SetElem(i, j, cmb[j]);
      }
      for (size_t j = i+1; j < row_num; ++j) {
        trans_mat.SetElem(j, j-1, kIdCoefRepr);
      }
      // Calculate new follower.
      follower = SparOpReprMatSparCoefReprMatIncompleteMulti(
                     follower, trans_mat);
      break;
    }
  }
  if (i < row_num) {
    SparOpReprMatRowDelinearize(target, follower);
  }
}


inline void SparOpReprMatColDelinearize(
    SparOpReprMat &target, SparOpReprMat &follower) {
  auto col_num = target.cols;
  size_t i;
  for (i = 1; i < col_num; ++i) {
    auto cmb = target.CalcColLinCmb(i);
    if (CalcSparOpReprMatColLinCmb(target, cmb) == target.GetCol(i)) {
      // Remove the col.
      target.RemoveCol(i);
      // Construct transform matrix.
      SparCoefReprMat trans_mat(col_num-1, col_num);
      for (size_t j = 0; j < i; ++j) {
        trans_mat.SetElem(j, j, kIdCoefRepr);
      }
      for (size_t j = 0; j < i; ++j) {
        trans_mat.SetElem(j, i, cmb[j]);
      }
      for (size_t j = i+1; j < col_num; ++j) {
        trans_mat.SetElem(j-1, j, kIdCoefRepr);
      }
      // Calculate new follower.
      follower = SparCoefReprMatSparOpReprMatIncompleteMulti(
                     trans_mat, follower);
      break;
    }
  }
  if (i < col_num) {
    SparOpReprMatColDelinearize(target, follower);
  }
}


// Row and column compresser.
inline void SparOpReprMatRowCompresser(
    SparOpReprMat &target, SparOpReprMat &follower) {
  assert(target.rows == follower.cols);
  auto row_num = target.rows;
  if (row_num == 1) { return; }
  // Sort rows of target and transpose cols of follower.
  auto sorted_row_idxs = target.SortRows();
  follower.TransposeCols(sorted_row_idxs);
  // Separate row coefficients of target.
  bool need_separate_row_coef = false;
  SparCoefReprMat row_coef_trans_mat(row_num, row_num);
  for (size_t row_idx = 0; row_idx < row_num; ++row_idx) {
    auto row_coef = target.CalcRowCoef(row_idx);
    if (row_coef != kNullCoefRepr) {
      row_coef_trans_mat.SetElem(row_idx, row_idx, row_coef);
    } else {
      row_coef_trans_mat.SetElem(row_idx, row_idx, kIdCoefRepr);
    }
    if ((row_coef != kNullCoefRepr) && (row_coef != kIdCoefRepr)) {
      need_separate_row_coef = true;
      target.RemoveRowCoef(row_idx);
    }
  }
  if (need_separate_row_coef) {
    follower = SparOpReprMatSparCoefReprMatIncompleteMulti(
                   follower, row_coef_trans_mat);
  }
  // Delinearize rows of target.
  SparOpReprMatRowDelinearize(target, follower);
}


inline void SparOpReprMatColCompresser(
    SparOpReprMat &target, SparOpReprMat &follower) {
  assert(target.cols == follower.rows);
  auto col_num = target.cols;
  if (col_num == 1) { return; }
  // Sort cols of target and transpose rows of follower.
  auto sorted_col_idxs = target.SortCols();
  follower.TransposeRows(sorted_col_idxs);
  // Separate col coefficients of target.
  bool need_separate_col_coef = false;
  SparCoefReprMat col_coef_trans_mat(col_num, col_num);
  for (size_t col_idx = 0; col_idx < col_num; ++col_idx) {
    auto col_coef = target.CalcColCoef(col_idx);
    if (col_coef != kNullCoefRepr) {
      col_coef_trans_mat.SetElem(col_idx, col_idx, col_coef);
    } else {
      col_coef_trans_mat.SetElem(col_idx, col_idx, kIdCoefRepr);
    }
    if ((col_coef != kNullCoefRepr) && (col_coef != kIdCoefRepr)) {
      need_separate_col_coef = true;
      target.RemoveColCoef(col_idx);
    }
  }
  if (need_separate_col_coef) {
    follower = SparCoefReprMatSparOpReprMatIncompleteMulti(
                   col_coef_trans_mat, follower);
  }
  // Delinearize cols of target.
  SparOpReprMatColDelinearize(target, follower);
}


// Helpers.
template <typename T>
bool ElemInVec(const T &e, const std::vector<T> &v, long &pos) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (e == v[i]) {
      pos = i;
      return true;
    }
  }
  pos = -1;
  return false;
}


template <typename VecT>
VecT ConcatenateTwoVec(const VecT &va, const VecT &vb) {
  VecT res;
  res.reserve(va.size() + vb.size());
  res.insert(res.end(), va.begin(), va.end());
  res.insert(res.end(), vb.begin(), vb.end());
  return res;
}
#endif /* ifndef GQMPS2_ONE_DIM_TN_MPO_MPOGEN_COEF_OP_ALG_H */
