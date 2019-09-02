// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-13 14:34
* 
* Description: GraceQ/mps2 project. Private objects used by MPO generation.
*/
#ifndef GQMPS2_MPOGEN_H
#define GQMPS2_MPOGEN_H


#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <vector>


namespace gqmps2 {
using namespace gqten;


void AddOpToHeadMpoTen(GQTensor *, const GQTensor &, const long);

void AddOpToTailMpoTen(GQTensor *, const GQTensor &, const long);

void AddOpToCentMpoTen(GQTensor *, const GQTensor &, const long, const long);


// Helpers.
inline QN GetLvbTargetQN(const Index &index, const long coor) {
  auto coor_off_set_and_qnsct = index.CoorInterOffsetAndQnsct(coor);
  return coor_off_set_and_qnsct.qnsct.qn;
}


inline void ResortNodes(std::vector<FSMNode *> &sorted_nodes) {
  auto ref_idx = sorted_nodes.back()->mid_state_idx;
  for (auto it = sorted_nodes.begin(); it != sorted_nodes.end()-1; ++it) {
    if ((*it)->mid_state_idx >= ref_idx) { ++((*it)->mid_state_idx); }
  }
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_MPOGEN_H */
