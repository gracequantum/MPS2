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
bool ElemInVector(const T &, const std::vector<T> &, long &pos);


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
      if (!ElemInVector(l_elem, rhs_coef_label_list, pos_in_rhs)) {
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
    const std::vector<CoefLabel> &rc_rhs_coef_label_list = rhs.coef_label_list_;
    std::vector<CoefLabel> added_coef_label_list;
    added_coef_label_list.reserve(
        coef_label_list_.size() + rc_rhs_coef_label_list.size());
    added_coef_label_list.insert(
        added_coef_label_list.end(),
        coef_label_list_.begin(), coef_label_list_.end());
    added_coef_label_list.insert(
        added_coef_label_list.end(),
        rc_rhs_coef_label_list.begin(), rc_rhs_coef_label_list.end());
    return CoefRepr(added_coef_label_list);
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

  std::vector<CoefRepr> GetCoefReprList(void) const {
    return coef_repr_list_; 
  }

  std::vector<OpLabel> GetOpLabelList(void) const {
    return op_label_list_; 
  }

private:
  std::vector<CoefRepr> coef_repr_list_;
  std::vector<OpLabel> op_label_list_;
};


// Helpers.
template <typename T>
bool ElemInVector(const T &e, const std::vector<T> &v, long &pos) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (e == v[i]) {
      pos = i;
      return true;
    }
  }
  pos = -1;
  return false;
}
#endif /* ifndef GQMPS2_DETAIL_MPOGEN_COEF_OP_ALG_H */
