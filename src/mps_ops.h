/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-16 21:12
* 
* Description: GraceQ/mps2 project. MPS operations.
*/
#ifndef GQMPS2_MPS_OPS_H
#define GQMPS2_MPS_OPS_H


#include "gqten/gqten.h"

#include <vector>

namespace gqmps2 {
using namespace gqten;


Index GenHeadRightVirtBond(const Index &, const QN &, const long);

Index GenBodyRightVirtBond(
    const Index &, const Index &, const QN &, const long);

Index GenTailLeftVirtBond(const Index &, const QN &, const long);

Index GenBodyLeftVirtBond(const Index &, const Index &, const QN &, const long);

void DimCut(std::vector<QNSector> &, const long, const long);

inline bool GreaterQNSectorDim(const QNSector &qnsct1, const QNSector &qnsct2) {
  return qnsct1.dim > qnsct2.dim;
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_MPS_OPS_H */
