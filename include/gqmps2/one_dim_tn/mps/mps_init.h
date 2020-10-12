// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-29 22:11
*
* Description: GraceQ/MPS2 project. MPS initializations.
*/
#ifndef GQMPS2_ONE_DIM_TN_MPS_MPS_INIT_H
#define GQMPS2_ONE_DIM_TN_MPS_MPS_INIT_H


#include "gqmps2/one_dim_tn/mps/mps.h"    // MPS
#include "gqten/gqten.h"

#include <assert.h>

#ifdef Release
  #define NDEBUG
#endif


namespace  gqmps2 {
using  namespace gqten;


// Forward declarations
// For random initialize MPS operation.
Index GenHeadRightVirtBond(const Index &, const QN &, const long);

Index GenBodyRightVirtBond(
    const Index &, const Index &, const QN &, const long);

Index GenTailLeftVirtBond(const Index &, const QN &, const long);

Index GenBodyLeftVirtBond(const Index &, const Index &, const QN &, const long);

void DimCut(std::vector<QNSector> &, const long, const long);


// Helpers
inline bool GreaterQNSectorDim(const QNSector &qnsct1, const QNSector &qnsct2) {
  return qnsct1.dim > qnsct2.dim;
}


template <typename MPST>
inline void MpsFree(MPST &mps) {
  for (int i = 0; i < mps.size(); ++i) {
    mps.dealloc(i);
  }
}


// MPS initializations.
template <typename TenType>
void RandomInitMps(
    MPS<TenType> &mps,
    const Index &pb,
    const QN &tot_div,
    const QN &zero_div,
    const long dmax) {
  MpsFree(mps);
  Index lvb, rvb;

  // Left to center.
  rvb = GenHeadRightVirtBond(pb, tot_div, dmax);
  mps(0) = new TenType({pb, rvb});
  mps(0)->Random(tot_div);
  assert(Div(mps[0]) == tot_div);
  auto N = mps.size();
  for (std::size_t i = 1; i < N/2; ++i) {
    lvb = InverseIndex(rvb);
    rvb = GenBodyRightVirtBond(lvb, pb, zero_div, dmax);
    mps(i) = new TenType({lvb, pb, rvb});
    mps(i)->Random(zero_div);
    assert(Div(mps[i]) == zero_div);
  }
  auto cent_bond = rvb;

  // Right to center.
  lvb = GenTailLeftVirtBond(pb, zero_div, dmax);
  mps(N-1) = new TenType({lvb, pb});
  mps(N-1)->Random(zero_div);
  assert(Div(mps[N-1]) == zero_div);
  for (std::size_t i = N-2; i > N/2; --i) {
    rvb = InverseIndex(lvb);
    lvb = GenBodyLeftVirtBond(rvb, pb, zero_div, dmax);
    mps(i) = new TenType({lvb, pb, rvb});
    mps(i)->Random(zero_div);
    assert(Div(mps[i]) == zero_div);
  }

  rvb = InverseIndex(lvb);
  lvb = InverseIndex(cent_bond);
  mps(N/2) = new TenType({lvb, pb, rvb});
  mps(N/2)->Random(zero_div);
  assert(Div(mps[N/2]) == zero_div);

  // Centralize MPS.
  mps.Centralize(0);
}


/** Random initialization of MPS, usually for non-uniform cases
 *
 * @tparam TenType a realization of template class GQTensor
 * @param mps a vector storing a series of pointers which point to mps tensors
 * @param pb_vector physical bond Indices with good quantum numbers, size == system size
 * @param tot_div total quantum number, the right size quantum number
 * @param zero_div left side quantum number of mps
 * @param dmax bond dimension
 */
//template <typename TenType>
//void RandomInitMps(
  //std::vector<TenType *> &mps,
  //const std::vector<Index> &pb_vector,
  //const QN &tot_div,
  //const QN &zero_div,
  //const long dmax) {
  //MpsFree(mps);
  //Index lvb, rvb;

  //assert(pb_vector.size()==mps.size());
  //// Left to center.
  //rvb = GenHeadRightVirtBond(pb_vector.front(), tot_div, dmax);
  //mps[0] = new TenType({pb_vector.front(), rvb});
  //mps[0]->Random(tot_div);
  //assert(Div(*mps[0]) == tot_div);
  //auto N = mps.size();
  //for (std::size_t i = 1; i < N/2; ++i) {
    //lvb = InverseIndex(rvb);
    //rvb = GenBodyRightVirtBond(lvb, pb_vector[i], zero_div, dmax);
    //mps[i] = new TenType({lvb, pb_vector[i], rvb});
    //mps[i]->Random(zero_div);
    //assert(Div(*mps[i]) == zero_div);
  //}
  //auto cent_bond = rvb;

  //// Right to center.
  //lvb = GenTailLeftVirtBond(pb_vector.back(), zero_div, dmax);
  //mps[N-1] = new TenType({lvb, pb_vector.back()});
  //mps[N-1]->Random(zero_div);
  //assert(Div(*mps[N-1]) == zero_div);
  //for (std::size_t i = N-2; i > N/2; --i) {
    //rvb = InverseIndex(lvb);
    //lvb = GenBodyLeftVirtBond(rvb, pb_vector[i], zero_div, dmax);
    //mps[i] = new TenType({lvb, pb_vector[i], rvb});
    //mps[i]->Random(zero_div);
    //assert(Div(*mps[i]) == zero_div);
  //}

  //rvb = InverseIndex(lvb);
  //lvb = InverseIndex(cent_bond);
  //mps[N/2] = new TenType({lvb, pb_vector[N/2], rvb});
  //mps[N/2]->Random(zero_div);
  //assert(Div(*mps[N/2]) == zero_div);
//}



inline Index GenHeadRightVirtBond(
    const Index &pb, const QN &tot_div, const long dmax) {
  std::vector<QNSector> new_qnscts;
  for (auto &qnsct : pb.qnscts) {
    auto new_qn = tot_div - qnsct.qn;
    auto has_qn = false;
    for (auto &new_qnsct : new_qnscts) {
      if (new_qnsct.qn == new_qn) {
        new_qnsct.dim += qnsct.dim;
        has_qn = true;
        break;
      }
    }
    if (!has_qn) {
      new_qnscts.push_back(QNSector(new_qn, qnsct.dim));
    }
  }
  DimCut(new_qnscts, dmax, pb.dim);
  return Index(new_qnscts, OUT);
}


inline Index GenBodyRightVirtBond(
    const Index &lvb, const Index &pb, const QN &zero_div, const long dmax) {
  std::vector<QNSector> new_qnscts;
  for (auto &lvqnsct : lvb.qnscts) {
    for (auto &pqnsct : pb.qnscts) {
      auto poss_rvb_qn = zero_div + lvqnsct.qn - pqnsct.qn;
      auto has_qn = false;
      for (auto &new_qnsct : new_qnscts) {
        if (poss_rvb_qn == new_qnsct.qn) {
          new_qnsct.dim += lvqnsct.dim;
          has_qn = true;
          break;
        }
      }
      if (!has_qn) {
        new_qnscts.push_back(QNSector(poss_rvb_qn, lvqnsct.dim));
      }
    }
  }
  DimCut(new_qnscts, dmax, pb.dim);
  return Index(new_qnscts, OUT);
}


inline Index GenTailLeftVirtBond(
    const Index &pb, const QN &zero_div, const long dmax) {
  std::vector<QNSector> new_qnscts;
  for (auto &qnsct : pb.qnscts) {
    auto new_qn = qnsct.qn - zero_div;
    auto has_qn = false;
    for (auto &new_qnsct : new_qnscts) {
      if (new_qnsct.qn == new_qn) {
        new_qnsct.dim += qnsct.dim;
        has_qn = true;
        break;
      }
    }
    if (!has_qn) {
      new_qnscts.push_back(QNSector(new_qn, qnsct.dim));
    }
  }
  DimCut(new_qnscts, dmax, pb.dim);
  return Index(new_qnscts, IN);
}


inline Index GenBodyLeftVirtBond(
    const Index &rvb, const Index &pb, const QN &zero_div, const long dmax) {
  std::vector<QNSector> new_qnscts;
  for (auto &rvqnsct : rvb.qnscts) {
    for (auto &pqnsct : pb.qnscts) {
      auto poss_lvb_qn = pqnsct.qn - zero_div + rvqnsct.qn;
      auto has_qn = false;
      for (auto &new_qnsct : new_qnscts) {
        if (poss_lvb_qn == new_qnsct.qn) {
          new_qnsct.dim += rvqnsct.dim;
          has_qn = true;
          break;
        }
      }
      if (!has_qn) {
        new_qnscts.push_back(QNSector(poss_lvb_qn, rvqnsct.dim));
      }
    }
  }
  DimCut(new_qnscts, dmax, pb.dim);
  return Index(new_qnscts, IN);
}


inline void DimCut(
    std::vector<QNSector> &qnscts, const long dmax, const long pdim) {
  std::sort(qnscts.begin(), qnscts.end(), GreaterQNSectorDim);
  auto kept_qn_cnt = 0;
  auto dim = 0;
  for (auto &qnsct : qnscts) {
    if (dim + qnsct.dim < dmax) {
      dim += qnsct.dim;
      kept_qn_cnt++;
    } else if (dim + qnsct.dim == dmax) {
      dim += qnsct.dim;
      kept_qn_cnt++;
      break;
    } else {
      qnsct.dim -= (dim + qnsct.dim - dmax);
      kept_qn_cnt++;
      break;
    }
  }
  qnscts.resize(kept_qn_cnt);
}


template <typename TenType>
void DirectStateInitMps(
    MPS<TenType> &mps, const std::vector<long> &stat_labs,
    const Index &pb_out, const QN &zero_div) {
  auto N = mps.size();
  assert(N == stat_labs.size());
  MpsFree(mps);
  Index lvb, rvb;

  // Calculate total quantum number.
  auto div = pb_out.CoorInterOffsetAndQnsct(stat_labs[0]).qnsct.qn;
  for (std::size_t i = 1; i < N; ++i) {
    div += pb_out.CoorInterOffsetAndQnsct(stat_labs[i]).qnsct.qn;
  }

  auto stat_lab = stat_labs[0];
  auto rvb_qn = div - pb_out.CoorInterOffsetAndQnsct(stat_lab).qnsct.qn;
  rvb = Index({QNSector(rvb_qn, 1)}, OUT);
  mps(0) = new TenType({pb_out, rvb});
  (mps[0])({stat_lab, 0}) = 1;

  for (std::size_t i = 1; i < N-1; ++i) {
    lvb = InverseIndex(rvb);
    stat_lab = stat_labs[i];
    rvb_qn = zero_div -
             pb_out.CoorInterOffsetAndQnsct(stat_lab).qnsct.qn +
             lvb.CoorInterOffsetAndQnsct(0).qnsct.qn;
    rvb = Index({QNSector(rvb_qn, 1)}, OUT);
    mps(i) = new TenType({lvb, pb_out, rvb});
    (mps[i])({0, stat_lab, 0}) = 1;
  }

  lvb = InverseIndex(rvb);
  mps(N-1) = new TenType({lvb, pb_out});
  stat_lab = stat_labs[N-1];
  (mps[N-1])({0, stat_lab}) = 1;

  // Centralize MPS.
  mps.Centralize(0);
}

/** Initialize MPS from a series of integer numbers,  for the non-uniform hilbert space cases
 *
 * @tparam TenType A realization of tensor class, GQTensor<GQTEN_Double> or GQTensor<GQTEN_Complex>
 * @param mps vectors contain the N pointers which point to TenType object, N is the system size
 * @param stat_labs label the local hilbert space states in order, with size N
 * @param pb_out_vector label the local hilbert space (physical bond) in order, with size N
 * @param zero_div left start quantum number
 */
//template <typename TenType>
//void DirectStateInitMps(
        //std::vector<TenType *> &mps, const std::vector<long> &stat_labs,
        //const std::vector<Index> &pb_out_vector, const QN &zero_div) {
  //auto N = mps.size();
  //assert(N == stat_labs.size());
  //assert(N == pb_out_vector.size());
  //MpsFree(mps);
  //Index lvb, rvb;

  //// Calculate total quantum number.
  //auto div = pb_out_vector.front().CoorInterOffsetAndQnsct(stat_labs[0]).qnsct.qn;
  //for (std::size_t i = 1; i < N; ++i) {
    //div += pb_out_vector[i].CoorInterOffsetAndQnsct(stat_labs[i]).qnsct.qn;
  //}

  //auto stat_lab = stat_labs[0];
  //auto rvb_qn = div - pb_out_vector.front().CoorInterOffsetAndQnsct(stat_lab).qnsct.qn;
  //rvb = Index({QNSector(rvb_qn, 1)}, OUT);
  //mps[0] = new TenType({pb_out_vector.front(), rvb});
  //(*mps[0])({stat_lab, 0}) = 1;

  //for (std::size_t i = 1; i < N-1; ++i) {
    //lvb = InverseIndex(rvb);
    //stat_lab = stat_labs[i];
    //rvb_qn = zero_div -
             //pb_out_vector[i].CoorInterOffsetAndQnsct(stat_lab).qnsct.qn +
             //lvb.CoorInterOffsetAndQnsct(0).qnsct.qn;
    //rvb = Index({QNSector(rvb_qn, 1)}, OUT);
    //mps[i] = new TenType({lvb, pb_out_vector[i], rvb});
    //(*mps[i])({0, stat_lab, 0}) = 1;
  //}

  //lvb = InverseIndex(rvb);
  //mps[N-1] = new TenType({lvb, pb_out_vector.back()});
  //stat_lab = stat_labs[N-1];
  //(*mps[N-1])({0, stat_lab}) = 1;
//}





template <typename TenType>
void ExtendDirectRandomInitMps(
    MPS<TenType> &mps,
    const std::vector<std::vector<long>> &stat_labs_set,
    const Index &pb, const QN &zero_div, const long enlarged_dim) {
  auto fusion_stats_num = stat_labs_set.size();
  assert(fusion_stats_num >= 1);
  auto N = mps.size();
  assert(N == stat_labs_set[0].size());
  MpsFree(mps);
  Index lvb, rvb;
  std::vector<QNSector> rvb_qnscts;

  // Calculate total quantum number.
  auto div = pb.CoorInterOffsetAndQnsct(stat_labs_set[0][0]).qnsct.qn;
  for (std::size_t i = 1; i < N; ++i) {
    div += pb.CoorInterOffsetAndQnsct(stat_labs_set[0][i]).qnsct.qn;
  }

  // Deal with MPS head local tensor.
  for (std::size_t i = 0; i < fusion_stats_num; ++i) {
    auto stat_lab = stat_labs_set[i][0];
    auto rvb_qn = div - pb.CoorInterOffsetAndQnsct(stat_lab).qnsct.qn;
    rvb_qnscts.push_back(QNSector(rvb_qn, enlarged_dim));
  }
  rvb = Index(rvb_qnscts, OUT);
  rvb_qnscts.clear();
  mps(0) = new TenType({pb, rvb});
  mps[0].Random(div);

  // Deal with MPS middle local tensors.
  for (std::size_t i = 1; i < N-1; ++i) {
    lvb = InverseIndex(rvb);
    for (std::size_t j = 0; j < fusion_stats_num; ++j) {
      auto stat_lab = stat_labs_set[j][i];
      auto rvb_qn = zero_div -
                    pb.CoorInterOffsetAndQnsct(stat_lab).qnsct.qn +
                    lvb.CoorInterOffsetAndQnsct(j*enlarged_dim).qnsct.qn;
      rvb_qnscts.push_back(QNSector(rvb_qn, enlarged_dim));
    }
    rvb = Index(rvb_qnscts, OUT);
    mps(i) = new TenType({lvb, pb, rvb});
    rvb_qnscts.clear();
    mps[i].Random(zero_div);
  }

  // Deal with MPS tail local tensor.
  lvb = InverseIndex(rvb);
  mps(N-1) = new TenType({lvb, pb});
  mps[N-1].Random(zero_div);

  // Centralize MPS.
  mps.Centralize(0);
}


/** ExtendDirectRandomInitMps
 *  Initialize MPS as a sum of some direct product states, for non-universal hilbert spaces.
 * @tparam TenType GQTensor<GQTEN_Double> or GQTensor<GQTEN_Complex>
 * @param mps A vector store the pointers pointing to GQTensors, but they must point to NULL
 * @param stat_labs_set A generalization of @param stat_labs in function DirectStateInitMps
 * @param pb_vector hilbert spaces set
 * @param zero_div the leftmost U1 quantum number
 * @param enlarged_dim stat_labs_set.size()
 */
//template <typename TenType>
//void ExtendDirectRandomInitMps(
  //std::vector<TenType *> &mps,
  //const std::vector<std::vector<long>> &stat_labs_set,
  //const std::vector<Index> &pb_vector, const QN &zero_div, const long enlarged_dim) {
  //auto fusion_stats_num = stat_labs_set.size();
  //assert(fusion_stats_num >= 1);
  //auto N = mps.size();
  //assert(N == stat_labs_set[0].size());
  //Index lvb, rvb;
  //std::vector<QNSector> rvb_qnscts;

  //// Calculate total quantum number.
  //auto pb = pb_vector.front();
  //auto div = pb.CoorInterOffsetAndQnsct(stat_labs_set[0][0]).qnsct.qn;
  //for (std::size_t i = 1; i < N; ++i) {
    //div += pb_vector[i].CoorInterOffsetAndQnsct(stat_labs_set[0][i]).qnsct.qn;
  //}

  //// Deal with MPS head local tensor.
  //for (std::size_t i = 0; i < fusion_stats_num; ++i) {
    //auto stat_lab = stat_labs_set[i][0];
    //auto rvb_qn = div - pb.CoorInterOffsetAndQnsct(stat_lab).qnsct.qn;
    //rvb_qnscts.push_back(QNSector(rvb_qn, enlarged_dim));
  //}
  //rvb = Index(rvb_qnscts, OUT);
  //rvb_qnscts.clear();
  //mps[0] = new TenType({pb, rvb});
  //mps[0]->Random(div);

  //// Deal with MPS middle local tensors.
  //for (std::size_t i = 1; i < N-1; ++i) {
    //auto pb = pb_vector[i];
    //lvb = InverseIndex(rvb);
    //for (std::size_t j = 0; j < fusion_stats_num; ++j) {
      //auto stat_lab = stat_labs_set[j][i];
      //auto rvb_qn = zero_div -
                    //pb.CoorInterOffsetAndQnsct(stat_lab).qnsct.qn +
                    //lvb.CoorInterOffsetAndQnsct(j*enlarged_dim).qnsct.qn;
      //rvb_qnscts.push_back(QNSector(rvb_qn, enlarged_dim));
    //}
    //rvb = Index(rvb_qnscts, OUT);
    //mps[i] = new TenType({lvb, pb, rvb});
    //rvb_qnscts.clear();
    //mps[i]->Random(zero_div);
  //}

  //// Deal with MPS tail local tensor.
  //lvb = InverseIndex(rvb);
  //mps[N-1] = new TenType({lvb, pb_vector.back()});
  //mps[N-1]->Random(zero_div);

  //// Centralize MPS.
  //auto temp_mps = MPS<TenType>(mps, -1);
  //RightNormalizeMps(temp_mps, temp_mps.N-1, 1);
//}
} /* gqmps2 */
#endif /* ifndef GQMPS2_ONE_DIM_TN_MPS_MPS_INIT_H */
