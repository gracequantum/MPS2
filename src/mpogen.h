/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-13 14:34
* 
* Description: GraceQ/mps2 project. Private objects used by MPO generation.
*/
#ifndef GQMPS2_MPOGEN_H
#define GQMPS2_MPOGEN_H


#include "gqten/gqten.h"

namespace gqmps2 {
using namespace gqten;


void AddOpToHeadMpoTen(GQTensor *, const GQTensor &, const long);

void AddOpToTailMpoTen(GQTensor *, const GQTensor &, const long);

void AddOpToCentMpoTen(GQTensor *, const GQTensor &, const long, const long);

} /* gqmps2 */ 
#endif /* ifndef GQMPS2_MPOGEN_H */
