// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-29 22:11
* 
* Description: GraceQ/MPS2 project. Implementation details for MPS operations.
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <algorithm>
#include <cmath>

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


// MPS operations
// MPS initialization.
template <typename TenType>
void RandomInitMps(
    std::vector<TenType *> &mps,
    const Index &pb,
    const QN &tot_div,
    const QN &zero_div,
    const long dmax) {
  Index lvb, rvb;

  // Left to center.
  rvb = GenHeadRightVirtBond(pb, tot_div, dmax);
  mps[0] = new TenType({pb, rvb});
  mps[0]->Random(tot_div);
  assert(Div(*mps[0]) == tot_div);
  auto N = mps.size();
  for (std::size_t i = 1; i < N/2; ++i) {
    lvb = InverseIndex(rvb);
    rvb = GenBodyRightVirtBond(lvb, pb, zero_div, dmax);
    mps[i] = new TenType({lvb, pb, rvb});
    mps[i]->Random(zero_div);
    assert(Div(*mps[i]) == zero_div);
  }
  auto cent_bond = rvb;

  // Right to center.
  lvb = GenTailLeftVirtBond(pb, zero_div, dmax);
  mps[N-1] = new TenType({lvb, pb});
  mps[N-1]->Random(zero_div);
  assert(Div(*mps[N-1]) == zero_div);
  for (std::size_t i = N-2; i > N/2; --i) {
    rvb = InverseIndex(lvb);
    lvb = GenBodyLeftVirtBond(rvb, pb, zero_div, dmax);
    mps[i] = new TenType({lvb, pb, rvb});
    mps[i]->Random(zero_div);
    assert(Div(*mps[i]) == zero_div);
  }

  rvb = InverseIndex(lvb);
  lvb = InverseIndex(cent_bond);
  mps[N/2] = new TenType({lvb, pb, rvb});
  mps[N/2]->Random(zero_div);
  assert(Div(*mps[N/2]) == zero_div);
}


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


inline void DimCut(std::vector<QNSector> &qnscts, const long dmax, const long pdim) {
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
} /* gqmps2 */ 
