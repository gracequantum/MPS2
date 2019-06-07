/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-16 21:12
* 
* Description: GraceQ/mps2 project. MPS operations.
*/
#ifndef GQMPS2_MPS_OPS_H
#define GQMPS2_MPS_OPS_H


#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <vector>

namespace gqmps2 {
using namespace gqten;


// For random initialize MPS operation.
Index GenHeadRightVirtBond(const Index &, const QN &, const long);

Index GenBodyRightVirtBond(
    const Index &, const Index &, const QN &, const long);

Index GenTailLeftVirtBond(const Index &, const QN &, const long);

Index GenBodyLeftVirtBond(const Index &, const Index &, const QN &, const long);

void DimCut(std::vector<QNSector> &, const long, const long);

inline bool GreaterQNSectorDim(const QNSector &qnsct1, const QNSector &qnsct2) {
  return qnsct1.dim > qnsct2.dim;
}


// MPS centralization.
void CentralizeMps(MPS &, const long);

void LeftNormalizeMps(MPS &, const long, const long);

void RightNormalizeMps(MPS &, const long, const long);

void LeftNormalizeMpsTen(MPS &, const long);

void RightNormalizeMpsTen(MPS &, const long);
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_MPS_OPS_H */
