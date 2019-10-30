// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-10-29 15:35
* 
* Description: GraceQ/MPS2 project. Algebra of MPO's coefficient and operator.
*/
#ifndef GQMPS2_DETAIL_MPOGEN_COEF_OP_ALG_H
#define GQMPS2_DETAIL_MPOGEN_COEF_OP_ALG_H


#include <vector>


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

private:
  std::vector<CoefLabel> coef_label_list_;
};


const CoefRepr kIdCoefRepr = CoefRepr(kIdCoefLabel);  // coefficient representation for identity coefficient 1.


// Label of operator.
using OpLabel = long;

const OpLabel kIdOpLabel = 0;         // Coefficient label for identity id.


// Representation of operator.
class OpRepr {
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
      const std::vector<OpLabel> &op_labels) :
          coef_repr_list_(coef_reprs), op_label_list_(op_labels) {
    assert(coef_repr_list_.size() == op_label_list_.size());
  }

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
    return OpRepr(
        ConcatenateTwoVec(coef_repr_list_, rhs.coef_repr_list_),
        ConcatenateTwoVec(op_label_list_, rhs.op_label_list_)); 
  }

private:
  std::vector<CoefRepr> coef_repr_list_;
  std::vector<OpLabel> op_label_list_;
};


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
#endif /* ifndef GQMPS2_DETAIL_MPOGEN_COEF_OP_ALG_H */
