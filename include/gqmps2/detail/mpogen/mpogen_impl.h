// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-27 17:38
* 
* Description: GraceQ/MPS2 project. Implantation details for MPO generator.
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"
#include "gqmps2/detail/mpogen/coef_op_alg.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

#include <assert.h>

#ifdef Release
  #define NDEBUG
#endif


namespace gqmps2 {
using namespace gqten;


// Forward declarations.
template <typename TenElemType>
void AddOpToHeadMpoTen(
    GQTensor<TenElemType> *, const GQTensor<TenElemType> &, const long);

template <typename TenElemType>
void AddOpToTailMpoTen(
    GQTensor<TenElemType> *, const GQTensor<TenElemType> &, const long);

template <typename TenElemType>
void AddOpToCentMpoTen(
    GQTensor<TenElemType> *, const GQTensor<TenElemType> &,
    const long, const long);


// MPO generator
template <typename TenElemType>
MPOGenerator<TenElemType>::MPOGenerator(
    const long N, const Index &pb, const QN &zero_div) :
    N_(N),
    pb_out_(pb),
    zero_div_(zero_div),
    fsm_(N) {
  pb_in_ = InverseIndex(pb_out_);
  id_op_ = GenIdOpTen_(pb_out_);
  coef_label_convertor_ = LabelConvertor<TenElemType>(TenElemType(1));
  op_label_convertor_ = LabelConvertor<GQTensorT>(id_op_);
}


template <typename TenElemType>
GQTensor<TenElemType>
MPOGenerator<TenElemType>::GenIdOpTen_(const Index &pb_out) {
  auto pb_in = InverseIndex(pb_out);
  auto id_op = GQTensorT({pb_in, pb_out});
  for (long i = 0; i < pb_out.dim; ++i) { id_op({i, i}) = 1; }
  return id_op;
}


template <typename TenElemType>
void MPOGenerator<TenElemType>::AddTerm(
    const TenElemType coef,
    const GQTensorVec &phys_ops,
    const std::vector<long> &idxs,
    const GQTensorVec &inst_ops) {
  assert(phys_ops.size() == idxs.size());
  for (auto idx : idxs) { assert(idx < N_); }
  assert((inst_ops.size() == phys_ops.size()-1) ||
         (inst_ops.size() == phys_ops.size()));
  if (coef == TenElemType(0)) { return; }   // If coef is zero, do nothing.
  CoefLabel coef_label = coef_label_convertor_.Convert(coef);
  long ntrvl_op_idx_head, ntrvl_op_idx_tail;
  ntrvl_op_idx_head = idxs.front();
  if (inst_ops.size() == phys_ops.size()-1) {
    ntrvl_op_idx_tail = idxs.back();
  } else if (inst_ops.size() == phys_ops.size()) {
    ntrvl_op_idx_tail = N_ - 1;
  }
  OpReprVec ntrvl_op_reprs;
  size_t last_phys_op_idx;
  for (long i = ntrvl_op_idx_head; i <= ntrvl_op_idx_tail; ++i) {
    auto poss_it = std::find(idxs.cbegin(), idxs.cend(), i);
    if (poss_it != idxs.cend()) {
      auto phys_ops_idx = poss_it - idxs.cbegin();
      OpLabel op_label = op_label_convertor_.Convert(phys_ops[phys_ops_idx]);
      if (phys_ops_idx == 0) {
        ntrvl_op_reprs.push_back(OpRepr(coef_label, op_label));
      } else {
        ntrvl_op_reprs.push_back(OpRepr(op_label));
      }
      last_phys_op_idx = phys_ops_idx;
    } else {
      OpLabel op_label = op_label_convertor_.Convert(
                             inst_ops[last_phys_op_idx]);
      ntrvl_op_reprs.push_back(OpRepr(op_label));
    }
  }
  assert(ntrvl_op_reprs.size() == (ntrvl_op_idx_tail - ntrvl_op_idx_head + 1));

  fsm_.AddPath(ntrvl_op_idx_head, ntrvl_op_idx_tail, ntrvl_op_reprs);
}

template <typename TenElemType>
void MPOGenerator<TenElemType>::AddTerm(
    const TenElemType coef,
    const GQTensorVec &phys_ops,
    const std::vector<long> &idxs,
    const GQTensorT &inst_op) {
  GQTensorT instop = inst_op;
  if (instop == kNullOperator<TenElemType>) { instop = id_op_; }
  auto phys_ops_num = phys_ops.size();
  assert(phys_ops_num > 0);
  GQTensorVec inst_ops(phys_ops_num-1, instop);

  AddTerm(coef, phys_ops, idxs, inst_ops);
}


template <typename TenElemType>
void MPOGenerator<TenElemType>::AddTerm(
    const TenElemType coef,
    const GQTensorT &phys_op,
    const long idx) {
  AddTerm(
      coef,
      GQTensorVec({phys_op}),
      std::vector<long>({idx}),
      GQTensorVec({}));
}


template <typename TenElemType>
typename MPOGenerator<TenElemType>::PGQTensorVec
MPOGenerator<TenElemType>::Gen(void) {
  auto fsm_comp_mat_repr = fsm_.GenCompressedMatRepr();
  auto label_coef_mapping = coef_label_convertor_.GetLabelObjMapping();
  auto label_op_mapping = op_label_convertor_.GetLabelObjMapping();
  
  // Print MPO tensors virtual bond dimension.
  for (auto &mpo_ten_repr : fsm_comp_mat_repr) {
    std::cout << std::setw(3) << mpo_ten_repr.cols << std::endl;
  }

  PGQTensorVec mpo(N_);
  Index trans_vb({QNSector(zero_div_, 1)}, OUT);
  std::vector<size_t> transposed_idxs;
  for (long i = 0; i < N_; ++i) {
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
                   label_coef_mapping, label_op_mapping);
    }
  }
  return mpo;
}


template< typename TenElemType>
QN MPOGenerator<TenElemType>::CalcTgtRvbQN_(
    const size_t x, const size_t y, const OpRepr &op_repr,
    const GQTensorVec &label_op_mapping, const Index &trans_vb) {
  auto lvb = InverseIndex(trans_vb);
  auto coor_off_set_and_qnsct = lvb.CoorInterOffsetAndQnsct(x);
  auto lvb_qn = coor_off_set_and_qnsct.qnsct.qn;
  auto op0_in_op_repr = label_op_mapping[op_repr.GetOpLabelList()[0]];
  return zero_div_ - Div(op0_in_op_repr) + lvb_qn;
}


template <typename TenElemType>
std::vector<size_t> MPOGenerator<TenElemType>::SortSparOpReprMatColsByQN_(
    SparOpReprMat &op_repr_mat, Index &trans_vb,
    const GQTensorVec &label_op_mapping) {
  std::vector<QNSector> rvb_qnscts;
  std::vector<size_t> transposed_idxs;
  for (size_t y = 0; y < op_repr_mat.cols; ++y) {
    bool has_ntrvl_op = false;
    QN col_rvb_qn;
    for (size_t x = 0; x < op_repr_mat.rows; ++x) {
      auto elem = op_repr_mat(x, y);
      if (elem != kNullOpRepr) {
        auto rvb_qn = CalcTgtRvbQN_(
                          x, y, elem, label_op_mapping, trans_vb);  
        if (!has_ntrvl_op) {
          col_rvb_qn = rvb_qn;
          has_ntrvl_op = true; 
          bool has_qn = false;
          size_t offset = 0;
          for (auto &qnsct : rvb_qnscts) {
            if (qnsct.qn == rvb_qn) {
              qnsct.dim += 1;
              auto beg_it = transposed_idxs.begin();
              transposed_idxs.insert(beg_it+offset, y);
              has_qn = true;
              break;
            } else {
              offset += qnsct.dim;
            }
          }
          if (!has_qn) {
            rvb_qnscts.push_back(QNSector(rvb_qn, 1));
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
  trans_vb = Index(rvb_qnscts, OUT);
  return transposed_idxs;
}


template <typename TenElemType>
typename MPOGenerator<TenElemType>::GQTensorT *
MPOGenerator<TenElemType>::HeadMpoTenRepr2MpoTen_(
    const SparOpReprMat &op_repr_mat,
    const Index &rvb,
    const TenElemVec &label_coef_mapping, const GQTensorVec &label_op_mapping) {
  auto pmpo_ten = new GQTensorT({pb_in_, rvb, pb_out_});
  for (size_t y = 0; y < op_repr_mat.cols; ++y) {
    auto elem = op_repr_mat(0, y);
    if (elem != kNullOpRepr) {
      auto op = elem.Realize(label_coef_mapping, label_op_mapping);
      AddOpToHeadMpoTen(pmpo_ten, op, y);
    }
  }
  return pmpo_ten;
}


template <typename TenElemType>
typename MPOGenerator<TenElemType>::GQTensorT *
MPOGenerator<TenElemType>::TailMpoTenRepr2MpoTen_(
    const SparOpReprMat &op_repr_mat,
    const Index &lvb,
    const TenElemVec &label_coef_mapping, const GQTensorVec &label_op_mapping) {
  auto pmpo_ten = new GQTensor<TenElemType>({pb_in_, lvb, pb_out_});
  for (size_t x = 0; x < op_repr_mat.rows; ++x) {
    auto elem = op_repr_mat(x, 0);
    if (elem != kNullOpRepr) {
      auto op = elem.Realize(label_coef_mapping, label_op_mapping);
      AddOpToTailMpoTen(pmpo_ten, op, x);
    }
  }
  return pmpo_ten;
}


template <typename TenElemType>
typename MPOGenerator<TenElemType>::GQTensorT *
MPOGenerator<TenElemType>::CentMpoTenRepr2MpoTen_(
    const SparOpReprMat &op_repr_mat,
    const Index &lvb,
    const Index &rvb,
    const TenElemVec &label_coef_mapping, const GQTensorVec &label_op_mapping) {
  auto pmpo_ten = new GQTensor<TenElemType>({lvb, pb_in_, pb_out_, rvb});
  for (size_t x = 0; x < op_repr_mat.rows; ++x) {
    for (size_t y = 0; y < op_repr_mat.cols; ++y) {
      auto elem = op_repr_mat(x, y);
      if (elem != kNullOpRepr) {
        auto op = elem.Realize(label_coef_mapping, label_op_mapping);
        AddOpToCentMpoTen(pmpo_ten, op, x, y);
      }
    }
  }
  return pmpo_ten;
}


template <typename TenElemType>
void AddOpToHeadMpoTen(
    GQTensor<TenElemType> *pmpo_ten, const GQTensor<TenElemType> &rop, const long rvb_coor) {
  for (long bpb_coor = 0; bpb_coor < rop.indexes[0].dim; ++bpb_coor) {
    for (long tpb_coor = 0; tpb_coor < rop.indexes[1].dim; ++tpb_coor) {
      auto elem = rop.Elem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)({bpb_coor, rvb_coor, tpb_coor}) = elem;
      }
    }
  }
}


template <typename TenElemType>
void AddOpToTailMpoTen(
    GQTensor<TenElemType> *pmpo_ten, const GQTensor<TenElemType> &rop, const long lvb_coor) {
  for (long bpb_coor = 0; bpb_coor < rop.indexes[0].dim; ++bpb_coor) {
    for (long tpb_coor = 0; tpb_coor < rop.indexes[1].dim; ++tpb_coor) {
      auto elem = rop.Elem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)({bpb_coor, lvb_coor, tpb_coor}) = elem;
      }
    }
  }
}


template <typename TenElemType>
void AddOpToCentMpoTen(
    GQTensor<TenElemType> *pmpo_ten, const GQTensor<TenElemType> &rop,
    const long lvb_coor, const long rvb_coor) {
  for (long bpb_coor = 0; bpb_coor < rop.indexes[0].dim; ++bpb_coor) {
    for (long tpb_coor = 0; tpb_coor < rop.indexes[1].dim; ++tpb_coor) {
      auto elem = rop.Elem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)({lvb_coor, bpb_coor, tpb_coor, rvb_coor}) = elem;
      }
    }
  }
}
} /* gqmps2 */ 
