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


void DumpMps(const std::vector<GQTensor *> &);

void LoadMps(std::vector<GQTensor *> &);

void RandomInitMps(
    std::vector<GQTensor *> &, const Index &, const QN &, const QN &);

Index GenHeadVirtBond(const Index &, const QN &);

Index GenBodyVirtBond(const Index &, const Index &, const QN &);

} /* gqmps2 */ 
#endif /* ifndef GQMPS2_MPS_OPS_H */
