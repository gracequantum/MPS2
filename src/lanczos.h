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
const std::vector<long> kHamMulStateCentLCA0 = {0};
const std::vector<long> kHamMulStateCentLCA1 = {0, 2};
const std::vector<long> kHamMulStateCentLCA2 = {4, 1};
const std::vector<long> kHamMulStateCentLCA3 = {4, 1};
const std::vector<long> kHamMulStateCentRCA0 = {0};
const std::vector<long> kHamMulStateCentRCA1 = {0, 1};
const std::vector<long> kHamMulStateCentRCA2 = {0, 1};
const std::vector<long> kHamMulStateCentRCA3 = {1, 0};
const std::vector<long> kHamMulStateLendLCA0 = {0};
const std::vector<long> kHamMulStateLendLCA1 = {0, 2};
const std::vector<long> kHamMulStateLendLCA2 = {0, 3};
const std::vector<long> kHamMulStateLendRCA0 = {0};
const std::vector<long> kHamMulStateLendRCA1 = {1, 0};
const std::vector<long> kHamMulStateLendRCA2 = {0, 1};
const std::vector<long> kHamMulStateRendLCA0 = {0};
const std::vector<long> kHamMulStateRendLCA1 = {2, 0};
const std::vector<long> kHamMulStateRendLCA2 = {3, 0};
const std::vector<long> kHamMulStateRendRCA0 = {0};
const std::vector<long> kHamMulStateRendRCA1 = {0, 1};
const std::vector<long> kHamMulStateRendRCA2 = {1, 0};

GQTensor *eff_ham_mul_state_cent(
    const std::vector<GQTensor *> &, GQTensor *);

GQTensor *eff_ham_mul_state_lend(
    const std::vector<GQTensor *> &, GQTensor *);

GQTensor *eff_ham_mul_state_rend(
    const std::vector<GQTensor *> &, GQTensor *);

void TridiagGsSolver(
    const std::vector<double> &, const std::vector<double> &, const long,
    double &, double * &, const char);


// Helpers.
inline void LanczosFree(
    double * &a,
    std::vector<GQTensor *> &b,
    GQTensor * &last_mat_mul_vec_res) {
  if (a != nullptr) { delete [] a; }
  for (auto &ptr : b) { delete ptr; }
  delete last_mat_mul_vec_res;
}

} /* gqmps2 */ 
#endif /* ifndef GQMPS2_LANCZOS_H */
