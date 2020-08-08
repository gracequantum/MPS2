// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-29 20:43
* 
* Description: GraceQ/MPS2 project. MPO generator.
*/

/**
@file mpogen.h
@brief MPO generator for generic quantum many-body systems.
*/
#ifndef GQMPS2_MPOGEN_MPOGEN_H
#define GQMPS2_MPOGEN_MPOGEN_H


#include "gqmps2/consts.h"       // kNullIntVec
#include "gqmps2/site_vec.h"     // SiteVec
#include "gqmps2/one_dim_tn/mpo.h"    // MPO
#include "gqmps2/mpogen/fsm.h"
#include "gqmps2/mpogen/symb_alg/coef_op_alg.h"
#include "gqten/gqten.h"


namespace gqmps2 {
using namespace gqten;


/**
A generic MPO generator. A matrix-product operator (MPO) generator which can
generate an efficient MPO for a quantum many-body system with any type of n-body
interaction term.

@tparam TenElemType Element type of the MPO tensors, can be GQTEN_Double or
        GQTEN_Complex.

@since version 0.0.0
*/
template <typename TenElemType>
class MPOGenerator {
public:
  using TenElemVec = std::vector<TenElemType>;
  using GQTensorT = GQTensor<TenElemType>;
  using GQTensorVec = std::vector<GQTensorT>;
  using PGQTensorVec = std::vector<GQTensorT *>;

  MPOGenerator(const SiteVec &, const QN &);

  void AddTerm(
      const TenElemType,
      const GQTensorVec &,
      const std::vector<int> &
  );

  void AddTerm(
      const TenElemType,
      const GQTensorVec &,
      const std::vector<int> &,
      const GQTensorVec &,
      const std::vector<std::vector<int>> &inst_ops_idxs_set = kNullIntVecVec
  );

  void AddTerm(
    const TenElemType,
    const GQTensorT &,
    const int,
    const GQTensorT &op2 = GQTensorT(),
    const int op2_idx = 0,
    const GQTensorT &inst_op = GQTensorT(),
    const std::vector<int> &inst_op_idxs = kNullIntVec
  );

  FSM GetFSM(void) { return fsm_; }

  MPO<GQTensorT> Gen(void);

private:
  long N_;
  SiteVec site_vec_;
  std::vector<Index> pb_in_vector_;
  std::vector<Index> pb_out_vector_;
  QN zero_div_;
  std::vector<GQTensorT> id_op_vector_;
  FSM fsm_;
  LabelConvertor<TenElemType> coef_label_convertor_;
  LabelConvertor<GQTensorT> op_label_convertor_;

  GQTensorT GenIdOpTen_(const Index &);

  std::vector<size_t> SortSparOpReprMatColsByQN_(
      SparOpReprMat &, Index &, const GQTensorVec &);

  QN CalcTgtRvbQN_(
    const size_t, const size_t, const OpRepr &,
    const GQTensorVec &, const Index &);

  GQTensorT HeadMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const Index &,
      const TenElemVec &, const GQTensorVec &);

  GQTensorT TailMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const Index &,
      const TenElemVec &, const GQTensorVec &);

  GQTensorT CentMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const Index &,
      const Index &,
      const TenElemVec &,
      const GQTensorVec &, const long);
};
} /* gqmps2 */ 


// Implementation details
#include "gqmps2/mpogen/mpogen_impl.h"


#endif /* ifndef GQMPS2_MPOGEN_MPOGEN_H */
