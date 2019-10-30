// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-10-29 15:34
* 
* Description: GraceQ/MPS2 project. Unittests for algebra of MPO's coefficient and operator.
*/
#include "gqmps2/detail/mpogen/coef_op_alg.h"

#include <vector>

#include "gtest/gtest.h"


// Helpers.
std::vector<long> RandVec(size_t size) {
  std::vector<long> rand_vec;
  for (size_t i = 0; i < size; ++i) {
    rand_vec.push_back(rand());
  }
  return rand_vec;
}


std::vector<long> InverseVec(const std::vector<long> &v) {
  std::vector<long> inv_v;
  if (v.empty()) { return inv_v; }
  for (long i = v.size()-1; i >= 0; --i) {
    inv_v.push_back(v[i]);
  }
  return inv_v;
}


TEST(TestCoefRepr, Initialization) {
  CoefRepr null_coef_repr;
  EXPECT_EQ(null_coef_repr.GetCoefLabelList(), std::vector<CoefLabel>());

  CoefRepr id_coef_repr(kIdCoefLabel);
  std::vector<CoefLabel> id_coef_label_list = {kIdCoefLabel};
  EXPECT_EQ(id_coef_repr.GetCoefLabelList(), id_coef_label_list);

  auto rand_coef_labels = RandVec(5);
  CoefRepr rand_coef_repr(rand_coef_labels);
  EXPECT_EQ(rand_coef_repr.GetCoefLabelList(), rand_coef_labels);
}


void RunTestCoefReprEquivalentCase(size_t size) {
  auto rand_coef_labels1 = RandVec(size);
  auto rand_coef_labels1_inv = InverseVec(rand_coef_labels1);
  CoefRepr coef_repr1a(rand_coef_labels1);
  CoefRepr coef_repr1b(rand_coef_labels1_inv);
  EXPECT_EQ(coef_repr1a, coef_repr1b);
  if (size != 0) {
    auto rand_coef_labels2 = RandVec(size);
    CoefRepr coef_repr2(rand_coef_labels2);
    EXPECT_NE(coef_repr1a, coef_repr2);
  }
}


TEST(TestCoefRepr, Equivalent) {
  RunTestCoefReprEquivalentCase(0);
  RunTestCoefReprEquivalentCase(1);
  RunTestCoefReprEquivalentCase(3);
  RunTestCoefReprEquivalentCase(5);
}


void RunTestCoefReprAddCase(size_t lhs_size, size_t rhs_size) {
  auto lhs_rand_coef_labels = RandVec(lhs_size);
  auto rhs_rand_coef_labels = RandVec(rhs_size);
  std::vector<CoefLabel> added_rand_coef_labels;
  added_rand_coef_labels.reserve(lhs_size + rhs_size);
  added_rand_coef_labels.insert(
      added_rand_coef_labels.end(),
      lhs_rand_coef_labels.begin(), lhs_rand_coef_labels.end());
  added_rand_coef_labels.insert(
      added_rand_coef_labels.end(),
      rhs_rand_coef_labels.begin(), rhs_rand_coef_labels.end());
  CoefRepr lhs_coef_repr(lhs_rand_coef_labels);
  CoefRepr rhs_coef_repr(rhs_rand_coef_labels);
  CoefRepr added_coef_repr(added_rand_coef_labels);
  EXPECT_EQ(lhs_coef_repr + rhs_coef_repr, added_rand_coef_labels);
  EXPECT_EQ(rhs_coef_repr + lhs_coef_repr, added_rand_coef_labels);
}


TEST(TestCoefRepr, Add) {
  RunTestCoefReprAddCase(0, 0);
  RunTestCoefReprAddCase(1, 0);
  RunTestCoefReprAddCase(0, 1);
  RunTestCoefReprAddCase(1, 1);
  RunTestCoefReprAddCase(2, 1);
  RunTestCoefReprAddCase(1, 2);
  RunTestCoefReprAddCase(3, 3);
  RunTestCoefReprAddCase(5, 3);
  RunTestCoefReprAddCase(3, 5);
  RunTestCoefReprAddCase(5, 5);
}


TEST(TestOpRepr, Initialization){
  OpRepr null_op_repr;
  EXPECT_EQ(null_op_repr.GetCoefReprList(), std::vector<CoefRepr>());
}
