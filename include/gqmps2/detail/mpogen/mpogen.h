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
#ifndef GQMPS2_DETAIL_MPOGEN_MPOGEN_H
#define GQMPS2_DETAIL_MPOGEN_MPOGEN_H


#include "gqmps2/detail/consts.h"       // kNullIntVec
#include "gqmps2/detail/site_vec.h"     // SiteVec
#include "gqmps2/detail/mpogen/fsm.h"
#include "gqmps2/detail/mpogen/symb_alg/coef_op_alg.h"
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
      const std::vector<size_t> &
  );

  void AddTerm(
      const TenElemType,
      const GQTensorVec &,
      const std::vector<size_t> &,
      const GQTensorVec &,
      const std::vector<std::vector<size_t>> &inst_ops_idxs_set = kNullIntVecVec
  );

  void AddTerm(
    const TenElemType,
    const GQTensorT &,
    const size_t,
    const GQTensorT &op2 = GQTensorT(),
    const size_t op2_idx = 0,
    const GQTensorT &inst_op = GQTensorT(),
    const std::vector<size_t> &inst_op_idxs = kNullIntVec
  );


  //void AddTerm(
      //const TenElemType,
      //const GQTensorVec &,
      //const std::vector<long> &,
      //const GQTensorVec &);

  //void AddTerm(
    //const TenElemType coef,
    //GQTensorVec phys_ops,
    //std::vector<long> idxs,
    //const GQTensorVec &inst_ops,
    //const std::vector<long> &inst_idxs);

  //void AddTerm(
      //const TenElemType,
      //const GQTensorVec &,
      //const std::vector<long> &,
      //const GQTensorT &inst_op=GQTensor<TenElemType>()
  //);

  //void AddTerm(
      //const TenElemType,
      //const GQTensorT &,
      //const long);

  FSM GetFSM(void) { return fsm_; }

  PGQTensorVec Gen(void);

private:
  long N_;
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

  GQTensorT *HeadMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const Index &,
      const TenElemVec &, const GQTensorVec &);

  GQTensorT *TailMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const Index &,
      const TenElemVec &, const GQTensorVec &);

  GQTensorT *CentMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const Index &,
      const Index &,
      const TenElemVec &,
      const GQTensorVec &, const long);
};
} /* gqmps2 */ 


// Implementation details
#include "gqmps2/detail/mpogen/mpogen_impl.h"


#endif /* ifndef GQMPS2_DETAIL_MPOGEN_MPOGEN_H */
