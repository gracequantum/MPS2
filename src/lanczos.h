/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 15:47
* 
* Description: GraceQ/mps2 project. Private objects for Lanczos solver.
*/
#ifndef GQMPS2_LANCZOS_H
#define GQMPS2_LANCZOS_H


#include "gqten/gqten.h"

#include <vector>


namespace gqmps2 {
using namespace gqten;

const int kLanczEnergyOutputPrecision = 16;

GQTensor *eff_ham_mul_state_cent(
    const std::vector<GQTensor *> &, const GQTensor *);

GQTensor *eff_ham_mul_state_lend(
    const std::vector<GQTensor *> &, const GQTensor *);

GQTensor *eff_ham_mul_state_rend(
    const std::vector<GQTensor *> &, const GQTensor *);

void TridiagGsSolver(
    const std::vector<double> &, const std::vector<double> &, const long,
    double &, double * &, const char);


// Helpers.
inline void InplaceContract(
    GQTensor * &lhs, const GQTensor &rhs,
    const std::vector<std::vector<long>> &axes) {
  auto res = Contract(*lhs, rhs, axes);
  delete lhs;
  lhs = res;
}

inline void LanczosFree(double * &a, std::vector<GQTensor *> &b) {
  if (a != nullptr) { delete [] a; }
  for (auto &ptr : b) { delete ptr; }
}

} /* gqmps2 */ 
#endif /* ifndef GQMPS2_LANCZOS_H */
