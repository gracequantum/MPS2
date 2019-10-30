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


std::vector<CoefRepr> GenCoefReprVec(
    const std::vector<CoefLabel> & coef_labels) {
  std::vector<CoefRepr> coef_reprs;
  for (auto label : coef_labels) { coef_reprs.push_back(CoefRepr(label)); }
  return coef_reprs;
}


// Testing representation of coefficient.
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
  EXPECT_EQ(lhs_coef_repr + rhs_coef_repr, added_coef_repr);
  EXPECT_EQ(rhs_coef_repr + lhs_coef_repr, added_coef_repr);
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


// Testing representation of operator.
TEST(TestOpRepr, Initialization){
  OpRepr null_op_repr;
  EXPECT_EQ(null_op_repr.GetCoefReprList(), std::vector<CoefRepr>());
  EXPECT_EQ(null_op_repr.GetOpLabelList(), std::vector<OpLabel>());

  OpLabel rand_op_label = rand();
  OpRepr nocoef_op_repr(rand_op_label);
  std::vector<CoefRepr> nocoef_op_coef_repr_list = {kIdCoefRepr};
  std::vector<OpLabel> nocoef_op_op_label_list = {rand_op_label};
  EXPECT_EQ(nocoef_op_repr.GetCoefReprList(), nocoef_op_coef_repr_list);
  EXPECT_EQ(nocoef_op_repr.GetOpLabelList(), nocoef_op_op_label_list);

  CoefRepr rand_coef_repr(rand());
  std::vector<CoefRepr> op_coef_repr_list = {rand_coef_repr};
  auto coef_op_op_label_list = nocoef_op_op_label_list;
  OpRepr coef_op_repr(rand_coef_repr, rand_op_label);
  EXPECT_EQ(coef_op_repr.GetCoefReprList(), op_coef_repr_list);
  EXPECT_EQ(coef_op_repr.GetOpLabelList(), coef_op_op_label_list);

  size_t size = 5;
  std::vector<CoefRepr> rand_coef_reprs;
  std::vector<OpLabel> rand_op_labels;
  for (size_t i = 0; i < size; ++i) {
    rand_coef_reprs.push_back(CoefRepr(rand()));
    rand_op_labels.push_back(rand());
  }
  OpRepr op_repr(rand_coef_reprs, rand_op_labels);
  EXPECT_EQ(op_repr.GetCoefReprList(), rand_coef_reprs);
  EXPECT_EQ(op_repr.GetOpLabelList(), rand_op_labels);
}


void RunTestOpReprEquivalentCase(size_t size) {
  auto rand_vec1a = RandVec(size);
  auto rand_vec1b = RandVec(size);
  auto rand_vec1a_inv = InverseVec(rand_vec1a);
  auto rand_vec1b_inv = InverseVec(rand_vec1b);
  std::vector<CoefRepr> coef_list1 = GenCoefReprVec(rand_vec1a);
  std::vector<CoefRepr> coef_list1_inv = GenCoefReprVec(rand_vec1a_inv);
  OpRepr op_repr1a(coef_list1, rand_vec1b);
  OpRepr op_repr1b(coef_list1_inv, rand_vec1b_inv);
  EXPECT_EQ(op_repr1a, op_repr1a);
  EXPECT_EQ(op_repr1a, op_repr1b);
  if (size != 0) {
    auto rand_vec2a = RandVec(size);
    auto rand_vec2b = RandVec(size);
    std::vector<CoefRepr> coef_list2 = GenCoefReprVec(rand_vec2a);
    OpRepr op_repr2(coef_list2, rand_vec2b);
    EXPECT_NE(op_repr2, op_repr1a);
  }
}


TEST(TestOpRepr, TestOpReprEquivalent) {
  RunTestOpReprEquivalentCase(0);
  RunTestOpReprEquivalentCase(1);
  RunTestOpReprEquivalentCase(3);
  RunTestOpReprEquivalentCase(5);
}


void RunTestOpReprAddCase(size_t lhs_size, size_t rhs_size) {
  auto lhs_rand_coef_reprs = GenCoefReprVec(RandVec(lhs_size));
  auto rhs_rand_coef_reprs = GenCoefReprVec(RandVec(rhs_size));
  auto lhs_rand_op_labels = RandVec(lhs_size);
  auto rhs_rand_op_labels = RandVec(rhs_size);
  std::vector<CoefRepr> added_rand_coef_reprs;
  added_rand_coef_reprs.reserve(lhs_size + rhs_size);
  added_rand_coef_reprs.insert(
      added_rand_coef_reprs.end(),
      lhs_rand_coef_reprs.begin(), lhs_rand_coef_reprs.end());
  added_rand_coef_reprs.insert(
      added_rand_coef_reprs.end(),
      rhs_rand_coef_reprs.begin(), rhs_rand_coef_reprs.end());
  std::vector<OpLabel> added_rand_op_labels;
  added_rand_op_labels.reserve(lhs_size + rhs_size);
  added_rand_op_labels.insert(
      added_rand_op_labels.end(),
      lhs_rand_op_labels.begin(), lhs_rand_op_labels.end());
  added_rand_op_labels.insert(
      added_rand_op_labels.end(),
      rhs_rand_op_labels.begin(), rhs_rand_op_labels.end());
  OpRepr lhs_op_repr(lhs_rand_coef_reprs, lhs_rand_op_labels);
  OpRepr rhs_op_repr(rhs_rand_coef_reprs, rhs_rand_op_labels);
  OpRepr added_op_repr(added_rand_coef_reprs, added_rand_op_labels);
  EXPECT_EQ(lhs_op_repr + rhs_op_repr, added_op_repr);
  EXPECT_EQ(rhs_op_repr + lhs_op_repr, added_op_repr);
}


TEST(TestOpRepr, TestOpReprAdd) {
  RunTestOpReprAddCase(0, 0);
  RunTestOpReprAddCase(1, 0);
  RunTestOpReprAddCase(0, 1);
  RunTestOpReprAddCase(1, 1);
  RunTestOpReprAddCase(2, 1);
  RunTestOpReprAddCase(1, 2);
  RunTestOpReprAddCase(3, 3);
  RunTestOpReprAddCase(5, 3);
  RunTestOpReprAddCase(3, 5);
  RunTestOpReprAddCase(5, 5);
}
