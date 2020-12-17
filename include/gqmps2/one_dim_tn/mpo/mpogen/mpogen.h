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
#ifndef GQMPS2_ONE_DIM_TN_MPO_MPOGEN_MPOGEN_H
#define GQMPS2_ONE_DIM_TN_MPO_MPOGEN_MPOGEN_H


#include "gqmps2/consts.h"       // kNullIntVec
#include "gqmps2/site_vec.h"     // SiteVec
#include "gqmps2/one_dim_tn/mpo/mpo.h"    // MPO
#include "gqmps2/one_dim_tn/mpo/mpogen/fsm.h"
#include "gqmps2/one_dim_tn/mpo/mpogen/symb_alg/coef_op_alg.h"
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
template <typename TenElemT, typename QNT>
class MPOGenerator {
public:
  using TenElemVec = std::vector<TenElemT>;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using GQTensorT = GQTensor<TenElemT, QNT>;
  using GQTensorVec = std::vector<GQTensorT>;
  using PGQTensorVec = std::vector<GQTensorT *>;

  MPOGenerator(const SiteVec<TenElemT, QNT> &, const QNT &);

  void AddTerm(
      const TenElemT,
      const GQTensorVec &,
      const std::vector<int> &
  );

  void AddTerm(
      const TenElemT,
      const GQTensorVec &,
      const std::vector<int> &,
      const GQTensorVec &,
      const std::vector<std::vector<int>> &inst_ops_idxs_set = kNullIntVecVec
  );

  void AddTerm(
    const TenElemT,
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
  size_t N_;
  SiteVec<TenElemT, QNT> site_vec_;
  std::vector<IndexT> pb_in_vector_;
  std::vector<IndexT> pb_out_vector_;
  QNT zero_div_;
  std::vector<GQTensorT> id_op_vector_;
  FSM fsm_;
  LabelConvertor<TenElemT> coef_label_convertor_;
  LabelConvertor<GQTensorT> op_label_convertor_;

  std::vector<size_t> SortSparOpReprMatColsByQN_(
      SparOpReprMat &, IndexT &, const GQTensorVec &
  );

  QNT CalcTgtRvbQN_(
    const size_t, const size_t, const OpRepr &,
    const GQTensorVec &, const IndexT &
  );

  GQTensorT HeadMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const IndexT &,
      const TenElemVec &, const GQTensorVec &
  );

  GQTensorT TailMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const IndexT &,
      const TenElemVec &, const GQTensorVec &
  );

  GQTensorT CentMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const IndexT &,
      const IndexT &,
      const TenElemVec &,
      const GQTensorVec &, const size_t
  );
};
} /* gqmps2 */


// Implementation details
#include "gqmps2/one_dim_tn/mpo/mpogen/mpogen_impl.h"


#endif /* ifndef GQMPS2_ONE_DIM_TN_MPO_MPOGEN_MPOGEN_H */
