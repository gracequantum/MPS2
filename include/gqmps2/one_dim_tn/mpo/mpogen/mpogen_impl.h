// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-27 17:38
* 
* Description: GraceQ/MPS2 project. Implantation details for MPO generator.
*/
#include "gqmps2/consts.h"     // kNullIntVec
#include "gqmps2/one_dim_tn/mpo/mpogen/mpogen.h"
#include "gqmps2/one_dim_tn/mpo/mpogen/symb_alg/coef_op_alg.h"
#include "gqten/gqten.h"

#include <iostream>
#include <iomanip>
#include <algorithm>    // is_sorted
#include <map>

#include <assert.h>     // assert

#ifdef Release
  #define NDEBUG
#endif


namespace gqmps2 {
using namespace gqten;


// Forward declarations.
template <typename TenT>
void AddOpToHeadMpoTen(TenT *, const TenT &, const size_t);

template <typename TenT>
void AddOpToTailMpoTen(TenT *, const TenT &, const size_t);

template <typename TenT>
void AddOpToCentMpoTen(TenT *, const TenT &, const size_t, const size_t);


/**
Create a MPO generator. Create a MPO generator using the sites of the system
which is described by a SiteVec.

@param site_vec The local Hilbert spaces of each sites of the system.
@param zero_div The zero value of the given quantum number type which is used
       to set the divergence of the MPO.

@since version 0.2.0
*/
template <typename TenElemT, typename QNT>
MPOGenerator<TenElemT, QNT>::MPOGenerator(
    const SiteVec<TenElemT, QNT> & site_vec,
    const QNT & zero_div
) : N_(site_vec.size),
    site_vec_(site_vec),
    zero_div_(zero_div),
    fsm_(site_vec.size) {
  pb_out_vector_.reserve(N_);
  pb_in_vector_.reserve(N_);
  for (size_t i = 0; i < N_; ++i) {
    pb_out_vector_.emplace_back(site_vec.sites[i]);
    pb_in_vector_.emplace_back(InverseIndex(site_vec.sites[i]));
  }
  id_op_vector_ = site_vec.id_ops;
  op_label_convertor_ = LabelConvertor<GQTensorT>(id_op_vector_[0]);
  std::vector<OpLabel> id_op_label_vector;
  for(auto iter = id_op_vector_.begin(); iter< id_op_vector_.end();iter++){
    id_op_label_vector.push_back(op_label_convertor_.Convert(*iter));
  }
  fsm_.ReplaceIdOpLabels(id_op_label_vector);

  coef_label_convertor_ = LabelConvertor<TenElemT>(TenElemT(1));
}


/**
The most generic API for adding a many-body term to the MPO generator. Notice
that the indexes of the operators have to be ascending sorted.

@param coef The coefficient of the term.
@param local_ops All the local (on-site) operators in the term.
@param local_ops_idxs The site indexes of these local operators.

@since version 0.2.0
*/
template <typename TenElemT, typename QNT>
void MPOGenerator<TenElemT, QNT>::AddTerm(
    const TenElemT coef,
    const GQTensorVec &local_ops,
    const std::vector<int> &local_ops_idxs
) {
  assert(local_ops.size() == local_ops_idxs.size());
  assert(std::is_sorted(local_ops_idxs.cbegin(), local_ops_idxs.cend()));
  assert(local_ops_idxs.back() < N_);
  if (coef == TenElemT(0)) { return; }   // If coef is zero, do nothing.

  auto coef_label = coef_label_convertor_.Convert(coef);
  int ntrvl_ops_idxs_head = local_ops_idxs.front();
  int ntrvl_ops_idxs_tail = local_ops_idxs.back();
  OpReprVec ntrvl_ops_reprs;
  for (int i = ntrvl_ops_idxs_head; i <= ntrvl_ops_idxs_tail; ++i) {
    auto poss_it = std::find(local_ops_idxs.cbegin(), local_ops_idxs.cend(), i);
    if (poss_it != local_ops_idxs.cend()) {     // Nontrivial operator
      auto local_op_loc = poss_it - local_ops_idxs.cbegin();    // Location of the local operator in the local operators list.
      auto op_label = op_label_convertor_.Convert(local_ops[local_op_loc]);
      if (local_op_loc == 0) {
        ntrvl_ops_reprs.push_back(OpRepr(coef_label, op_label));
      } else {
        ntrvl_ops_reprs.push_back(OpRepr(op_label));
      }
    } else {
      auto op_label = op_label_convertor_.Convert(id_op_vector_[i]);
      ntrvl_ops_reprs.push_back(OpRepr(op_label));
    }
  }
  assert(
      ntrvl_ops_reprs.size() == (ntrvl_ops_idxs_tail - ntrvl_ops_idxs_head + 1)
  );

  fsm_.AddPath(ntrvl_ops_idxs_head, ntrvl_ops_idxs_tail, ntrvl_ops_reprs);
}


/**
Add a many-body term defined by physical operators and insertion operators to
the MPO generator. The indexes of the operators have to be ascending sorted.

@param coef The coefficient of the term.
@param phys_ops Operators with physical meaning in this term. Like
       \f$c^{\dagger}\f$ operator in the \f$-t c^{\dagger}_{i} c_{j}\f$
       hopping term. Its size must be larger than 1.
@param phys_ops_idxs The corresponding site indexes of the physical operators.
@param inst_ops Operators which will be inserted between physical operators and
       also behind the last physical operator as a tail string. For example the
       Jordan-Wigner string operator.
@param inst_ops_idxs_set Each element defines the explicit site indexes of the
       corresponding inserting operator. If it is set to empty (default value),
       every site between the corresponding physical operators will be inserted
       a same insertion operator.

@since version 0.2.0
*/
template <typename TenElemT, typename QNT>
void MPOGenerator<TenElemT, QNT>::AddTerm(
    const TenElemT coef,
    const GQTensorVec &phys_ops,
    const std::vector<int> &phys_ops_idxs,
    const GQTensorVec &inst_ops,
    const std::vector<std::vector<int>> &inst_ops_idxs_set
) {
  assert(phys_ops.size() >= 2);
  assert(phys_ops.size() == phys_ops_idxs.size());
  assert(
      (inst_ops.size() == phys_ops.size() - 1) ||
      (inst_ops.size() == phys_ops.size())
  );
  if (inst_ops_idxs_set != kNullIntVecVec) {
    assert(inst_ops_idxs_set.size() == inst_ops.size());
  }

  GQTensorVec local_ops;
  std::vector<int> local_ops_idxs;
  for (int i = 0; i < phys_ops.size()-1; ++i) {
    local_ops.push_back(phys_ops[i]);
    local_ops_idxs.push_back(phys_ops_idxs[i]);
    if (inst_ops_idxs_set == kNullIntVecVec) {
      for (int j = phys_ops_idxs[i]+1; j < phys_ops_idxs[i+1]; ++j) {
        local_ops.push_back(inst_ops[i]);
        local_ops_idxs.push_back(j);
      }
    } else {
      for (auto inst_op_idx : inst_ops_idxs_set[i]) {
        local_ops.push_back(inst_ops[i]);
        local_ops_idxs.push_back(inst_op_idx);
      }
    }
  }
  // Deal with the last physical operator and possible insertion operator tail
  // string.
  local_ops.push_back(phys_ops.back());
  local_ops_idxs.push_back(phys_ops_idxs.back());
  if (inst_ops.size() == phys_ops.size()) {
    if (inst_ops_idxs_set == kNullIntVecVec) {
      for (int j = phys_ops_idxs.back(); j < N_; ++j) {
        local_ops.push_back(inst_ops.back());
        local_ops_idxs.push_back(j);
      }
    } else {
      for (auto inst_op_idx : inst_ops_idxs_set.back()) {
        local_ops.push_back(inst_ops.back());
        local_ops_idxs.push_back(inst_op_idx);
      }
    }
  }

  AddTerm(coef, local_ops, local_ops_idxs);
}


/**
Add one-body or two-body interaction term.

@param coef The coefficient of the term.
@param op1 The first physical operator for the term.
@param op1_idx The site index of the first physical operator.
@param op2 The second physical operator for the term.
@param op2_idx The site index of the second physical operator.
@param inst_op The insertion operator for the two-body interaction term.
@param inst_op_idxs The explicit site indexes of the insertion operator.

@since version 0.2.0
*/
template <typename TenElemT, typename QNT>
void MPOGenerator<TenElemT, QNT>::AddTerm(
    const TenElemT coef,
    const GQTensorT &op1,
    const int op1_idx,
    const GQTensorT &op2,
    const int op2_idx,
    const GQTensorT &inst_op,
    const std::vector<int> &inst_op_idxs
) {
  if (op2 == GQTensorT()) {     // One-body interaction term
    GQTensorVec local_ops = {op1};
    std::vector<int> local_ops_idxs = {op1_idx};
    AddTerm(coef, local_ops, local_ops_idxs);     // Use the most generic API
  } else {                      // Two-body interaction term
    assert(op2_idx != 0);
    if (inst_op == GQTensorT()) {     // Trivial insertion operator
      AddTerm(coef, {op1, op2}, {op1_idx, op2_idx});
    } else {                          // Non-trivial insertion operator
      if (inst_op_idxs == kNullIntVec) {    // Uniform insertion
        AddTerm(coef, {op1, op2}, {op1_idx, op2_idx}, {inst_op});
      } else {                              // Non-uniform insertion
        AddTerm(
            coef,
            {op1, op2}, {op1_idx, op2_idx},
            {inst_op}, {inst_op_idxs}
        );
      }
    }
  }
}


template <typename TenElemT, typename QNT>
MPO<typename MPOGenerator<TenElemT, QNT>::GQTensorT>
MPOGenerator<TenElemT, QNT>::Gen(void) {
  auto fsm_comp_mat_repr = fsm_.GenCompressedMatRepr();
  auto label_coef_mapping = coef_label_convertor_.GetLabelObjMapping();
  auto label_op_mapping = op_label_convertor_.GetLabelObjMapping();
  
  // Print MPO tensors virtual bond dimension.
  for (auto &mpo_ten_repr : fsm_comp_mat_repr) {
    std::cout << std::setw(3) << mpo_ten_repr.cols << std::endl;
  }

  MPO<GQTensorT> mpo(N_);
  IndexT trans_vb({QNSctT(zero_div_, 1)}, OUT);
  std::vector<size_t> transposed_idxs;
  for (size_t i = 0; i < N_; ++i) {
    if (i == 0) {
      transposed_idxs = SortSparOpReprMatColsByQN_(
                            fsm_comp_mat_repr[i], trans_vb, label_op_mapping);
      mpo[i] = HeadMpoTenRepr2MpoTen_(
                   fsm_comp_mat_repr[i], trans_vb,
                   label_coef_mapping, label_op_mapping);
    } else if (i == N_-1) {
      fsm_comp_mat_repr[i].TransposeRows(transposed_idxs);
      auto lvb = InverseIndex(trans_vb);
      mpo[i] = TailMpoTenRepr2MpoTen_(
                   fsm_comp_mat_repr[i], lvb,
                   label_coef_mapping, label_op_mapping);
    } else {
      fsm_comp_mat_repr[i].TransposeRows(transposed_idxs);
      auto lvb = InverseIndex(trans_vb);
      transposed_idxs = SortSparOpReprMatColsByQN_(
                            fsm_comp_mat_repr[i], trans_vb, label_op_mapping);
      mpo[i] = CentMpoTenRepr2MpoTen_(
                   fsm_comp_mat_repr[i], lvb, trans_vb,
                   label_coef_mapping, label_op_mapping, i);
    }
  }
  return mpo;
}


template< typename TenElemT, typename QNT>
QNT MPOGenerator<TenElemT, QNT>::CalcTgtRvbQN_(
    const size_t x, const size_t y, const OpRepr &op_repr,
    const GQTensorVec &label_op_mapping, const IndexT &trans_vb
) {
  auto lvb = InverseIndex(trans_vb);
  auto lvb_qn = lvb.GetQNSctFromActualCoor(x).GetQn();
  auto op0_in_op_repr = label_op_mapping[op_repr.GetOpLabelList()[0]];
  return zero_div_ - Div(op0_in_op_repr) + lvb_qn;
}


template <typename TenElemT, typename QNT>
std::vector<size_t> MPOGenerator<TenElemT, QNT>::SortSparOpReprMatColsByQN_(
    SparOpReprMat &op_repr_mat, IndexT &trans_vb,
    const GQTensorVec &label_op_mapping) {
  std::vector<std::pair<QNT, size_t>> rvb_qn_dim_pairs;
  std::vector<size_t> transposed_idxs;
  for (size_t y = 0; y < op_repr_mat.cols; ++y) {
    bool has_ntrvl_op = false;
    QNT col_rvb_qn;
    for (size_t x = 0; x < op_repr_mat.rows; ++x) {
      auto elem = op_repr_mat(x, y);
      if (elem != kNullOpRepr) {
        auto rvb_qn = CalcTgtRvbQN_(
                          x, y, elem, label_op_mapping, trans_vb
                      );
        if (!has_ntrvl_op) {
          col_rvb_qn = rvb_qn;
          has_ntrvl_op = true;
          bool has_qn = false;
          size_t offset = 0;
          for (auto &qn_dim : rvb_qn_dim_pairs) {
            if (qn_dim.first == rvb_qn) {
              qn_dim.second += 1;
              auto beg_it = transposed_idxs.begin();
              transposed_idxs.insert(beg_it+offset, y);
              has_qn = true;
              break;
            } else {
              offset += qn_dim.second;
            }
          }
          if (!has_qn) {
            rvb_qn_dim_pairs.push_back(std::make_pair(rvb_qn, 1));
            auto beg_it = transposed_idxs.begin();
            transposed_idxs.insert(beg_it+offset, y);
          }
        } else {
          assert(rvb_qn == col_rvb_qn);
        }
      }
    }
  }
  op_repr_mat.TransposeCols(transposed_idxs);
  std::vector<QNSctT> rvb_qnscts;
  for (auto &qn_dim : rvb_qn_dim_pairs) {
    rvb_qnscts.push_back(QNSctT(qn_dim.first, qn_dim.second));
  }
  trans_vb = IndexT(rvb_qnscts, OUT);
  return transposed_idxs;
}


template <typename TenElemT, typename QNT>
typename MPOGenerator<TenElemT, QNT>::GQTensorT
MPOGenerator<TenElemT, QNT>::HeadMpoTenRepr2MpoTen_(
    const SparOpReprMat &op_repr_mat,
    const IndexT &rvb,
    const TenElemVec &label_coef_mapping, const GQTensorVec &label_op_mapping
) {
  auto mpo_ten = GQTensorT({pb_in_vector_.front(), rvb, pb_out_vector_.front()});
  for (size_t y = 0; y < op_repr_mat.cols; ++y) {
    auto elem = op_repr_mat(0, y);
    if (elem != kNullOpRepr) {
      auto op = elem.Realize(label_coef_mapping, label_op_mapping);
      AddOpToHeadMpoTen(&mpo_ten, op, y);
    }
  }
  return mpo_ten;
}


template <typename TenElemT, typename QNT>
typename MPOGenerator<TenElemT, QNT>::GQTensorT
MPOGenerator<TenElemT, QNT>::TailMpoTenRepr2MpoTen_(
    const SparOpReprMat &op_repr_mat,
    const IndexT &lvb,
    const TenElemVec &label_coef_mapping, const GQTensorVec &label_op_mapping) {
  auto mpo_ten = GQTensorT({pb_in_vector_.back(), lvb, pb_out_vector_.back()});
  for (size_t x = 0; x < op_repr_mat.rows; ++x) {
    auto elem = op_repr_mat(x, 0);
    if (elem != kNullOpRepr) {
      auto op = elem.Realize(label_coef_mapping, label_op_mapping);
      AddOpToTailMpoTen(&mpo_ten, op, x);
    }
  }
  return mpo_ten;
}


template <typename TenElemT, typename QNT>
typename MPOGenerator<TenElemT, QNT>::GQTensorT
MPOGenerator<TenElemT, QNT>::CentMpoTenRepr2MpoTen_(
    const SparOpReprMat &op_repr_mat,
    const IndexT &lvb,
    const IndexT &rvb,
    const TenElemVec &label_coef_mapping, const GQTensorVec &label_op_mapping,
    const size_t site
) {
  auto mpo_ten = GQTensorT({lvb, pb_in_vector_[site], pb_out_vector_[site], rvb});
  for (size_t x = 0; x < op_repr_mat.rows; ++x) {
    for (size_t y = 0; y < op_repr_mat.cols; ++y) {
      auto elem = op_repr_mat(x, y);
      if (elem != kNullOpRepr) {
        auto op = elem.Realize(label_coef_mapping, label_op_mapping);
        AddOpToCentMpoTen(&mpo_ten, op, x, y);
      }
    }
  }
  return mpo_ten;
}


template <typename TenT>
void AddOpToHeadMpoTen(TenT *pmpo_ten, const TenT &rop, const size_t rvb_coor) {
  for (size_t bpb_coor = 0; bpb_coor < rop.GetIndexes()[0].dim(); ++bpb_coor) {
    for (size_t tpb_coor = 0; tpb_coor < rop.GetIndexes()[1].dim(); ++tpb_coor) {
      auto elem = rop.GetElem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)(bpb_coor, rvb_coor, tpb_coor) = elem;
      }
    }
  }
}


template <typename TenT>
void AddOpToTailMpoTen(TenT *pmpo_ten, const TenT &rop, const size_t lvb_coor) {
  for (size_t bpb_coor = 0; bpb_coor < rop.GetIndexes()[0].dim(); ++bpb_coor) {
    for (size_t tpb_coor = 0; tpb_coor < rop.GetIndexes()[1].dim(); ++tpb_coor) {
      auto elem = rop.GetElem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)(bpb_coor, lvb_coor, tpb_coor) = elem;
      }
    }
  }
}


template <typename TenT>
void AddOpToCentMpoTen(
    TenT *pmpo_ten, const TenT &rop,
    const size_t lvb_coor, const size_t rvb_coor
) {
  for (size_t bpb_coor = 0; bpb_coor < rop.GetIndexes()[0].dim(); ++bpb_coor) {
    for (size_t tpb_coor = 0; tpb_coor < rop.GetIndexes()[1].dim(); ++tpb_coor) {
      auto elem = rop.GetElem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)(lvb_coor, bpb_coor, tpb_coor, rvb_coor) = elem;
      }
    }
  }
}
} /* gqmps2 */
