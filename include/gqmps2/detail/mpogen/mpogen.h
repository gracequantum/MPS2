// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-29 20:43
* 
* Description: GraceQ/MPS2 project. MPO generator.
*/

/** @file mpogen.h
 *  @brief MPO generator for generic quantum many-body systems.
 */
#ifndef GQMPS2_DETAIL_MPOGEN_MPOGEN_H
#define GQMPS2_DETAIL_MPOGEN_MPOGEN_H


#include "gqmps2/detail/mpogen/fsm.h"
#include "gqmps2/detail/mpogen/symb_alg/coef_op_alg.h"
#include "gqten/gqten.h"


namespace gqmps2 {
using namespace gqten;


template <typename TenElemType>
class MPOGenerator {
public:
  MPOGenerator(const long, const Index &, const QN &);
  /** MPOGenerator Generator for non-uniform local hilbert space
    Input: - vector<Index>& pb_out_vector: the sets collecting the indices of all sites
           - const QN& zero_div: The leftmost index of MPO
   */
  MPOGenerator(const std::vector<Index> &, const QN& );

  using TenElemVec = std::vector<TenElemType>;
  using GQTensorT = GQTensor<TenElemType>;
  using GQTensorVec = std::vector<GQTensorT>;
  using PGQTensorVec = std::vector<GQTensorT *>;

  void AddTerm(
      const TenElemType,
      const GQTensorVec &,
      const std::vector<long> &,
      const GQTensorVec &);

  void AddTerm(
    const TenElemType coef,
    GQTensorVec phys_ops,
    std::vector<long> idxs,
    const GQTensorVec &inst_ops,
    const std::vector<long> &inst_idxs);

  void AddTerm(
      const TenElemType,
      const GQTensorVec &,
      const std::vector<long> &,
      const GQTensorT &inst_op=GQTensor<TenElemType>()
  );

  void AddTerm(
      const TenElemType,
      const GQTensorT &,
      const long);

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
