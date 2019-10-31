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


void RunTestSparCoefReprMatInitializationCase(size_t row_num, size_t col_num) {
  SparCoefReprMat spar_mat(row_num, col_num);
  EXPECT_EQ(spar_mat.rows, row_num);
  EXPECT_EQ(spar_mat.cols, col_num);
  EXPECT_TRUE(spar_mat.data.empty());
  auto size = row_num * col_num;
  auto indexes = spar_mat.indexes;
  EXPECT_EQ(indexes.size(), size);
  for (size_t i = 0; i < size; ++i) { EXPECT_EQ(indexes[i], -1); }
}


TEST(TestSparCoefReprMat, Initialization) {
  SparCoefReprMat null_coef_repr_mat;
  EXPECT_EQ(null_coef_repr_mat.rows, 0);
  EXPECT_EQ(null_coef_repr_mat.cols, 0);
  EXPECT_TRUE(null_coef_repr_mat.data.empty());
  EXPECT_TRUE(null_coef_repr_mat.indexes.empty());

  RunTestSparCoefReprMatInitializationCase(1, 1);
  RunTestSparCoefReprMatInitializationCase(5, 1);
  RunTestSparCoefReprMatInitializationCase(1, 5);
  RunTestSparCoefReprMatInitializationCase(5, 3);
  RunTestSparCoefReprMatInitializationCase(3, 5);
  RunTestSparCoefReprMatInitializationCase(5, 5);
}


void RunTestSparCoefReprMatElemGetterAndSetterCase(
    size_t row_num, size_t col_num) {
  auto size = row_num * col_num;
  SparCoefReprMat spar_mat;
  if (size == 0) {
    spar_mat = SparCoefReprMat();
  } else {
    spar_mat = SparCoefReprMat(row_num, col_num);
  }
  auto null_coef_repr = CoefRepr();
  for (size_t x = 0; x < row_num; ++x) {
    for (size_t y = 0; y < col_num; ++y) {
      EXPECT_EQ(spar_mat(x, y), null_coef_repr);
    }
  }
  if (size > 0) {
    auto x1 = rand() % row_num;
    auto y1 = rand() % col_num;
    CoefRepr coef1(rand());
    spar_mat.SetElem(x1, y1, coef1);
    EXPECT_EQ(spar_mat(x1, y1), coef1);
    CoefRepr coef2(rand());
    spar_mat.SetElem(x1, y1, coef2);
    EXPECT_EQ(spar_mat(x1, y1), coef2);
    if (size > 1) {
      auto x2 = rand() % row_num;
      auto y2 = rand() % col_num;
      CoefRepr coef3(rand());
      spar_mat.SetElem(x2, y2, coef3);
      EXPECT_EQ(spar_mat(x2, y2), coef3);
      CoefRepr coef4(rand());
      spar_mat.SetElem(x2, y2, coef4);
      EXPECT_EQ(spar_mat(x2, y2), coef4);
    }
  }
}


TEST(TestSparCoefReprMat, ElemGetterAndSetter) {
  RunTestSparCoefReprMatElemGetterAndSetterCase(0, 0);
  RunTestSparCoefReprMatElemGetterAndSetterCase(1, 1);
  RunTestSparCoefReprMatElemGetterAndSetterCase(5, 1);
  RunTestSparCoefReprMatElemGetterAndSetterCase(1, 5);
  RunTestSparCoefReprMatElemGetterAndSetterCase(5, 3);
  RunTestSparCoefReprMatElemGetterAndSetterCase(3, 5);
  RunTestSparCoefReprMatElemGetterAndSetterCase(5, 5);
  RunTestSparCoefReprMatElemGetterAndSetterCase(100, 100);
}


void RunTestSparCoefReprMatRowAndColGetter(
    const size_t row_num, const size_t col_num) {
  auto size = row_num * col_num;
  SparCoefReprMat spar_mat(row_num, col_num);
  auto null_coef_repr = CoefRepr();
  std::vector<CoefRepr> null_row(col_num, null_coef_repr);
  std::vector<CoefRepr> null_col(row_num, null_coef_repr);
  for (size_t row_idx = 0; row_idx < row_num; ++row_idx) {
    EXPECT_EQ(spar_mat.GetRow(row_idx), null_row);
  }
  for (size_t col_idx = 0; col_idx < col_num; ++col_idx) {
    EXPECT_EQ(spar_mat.GetCol(col_idx), null_col);
  }
  if (size > 0) {
    auto x = rand() % row_num;
    auto y = rand() % col_num;
    CoefRepr coef(rand());
    spar_mat.SetElem(x, y, coef);
    auto x_row = spar_mat.GetRow(x);
    for (size_t i = 0;i < col_num; ++i) {
      if (i == y) {
        EXPECT_EQ(x_row[i], coef);
      } else {
        EXPECT_EQ(x_row[i], null_coef_repr);
      }
    }
    auto y_col = spar_mat.GetCol(y);
    for (size_t i = 0; i < row_num; ++i) {
      if (i == x) {
        EXPECT_EQ(y_col[i], coef);
      } else {
        EXPECT_EQ(y_col[i], null_coef_repr);
      }
    }
  }
}


TEST(TestSparCoefReprMat, RowAndColGetter) {
  RunTestSparCoefReprMatRowAndColGetter(0, 0);
  RunTestSparCoefReprMatRowAndColGetter(1, 1);
  RunTestSparCoefReprMatRowAndColGetter(5, 1);
  RunTestSparCoefReprMatRowAndColGetter(1, 5);
  RunTestSparCoefReprMatRowAndColGetter(3, 5);
  RunTestSparCoefReprMatRowAndColGetter(5, 3);
  RunTestSparCoefReprMatRowAndColGetter(5, 5);
}


void RunTestSparCoefReprMatRemoveRowAndColCase(
    const size_t row_num, const size_t col_num) {
  SparCoefReprMat spar_mat(row_num, col_num);
  auto x = rand() % row_num;
  auto y = rand() % col_num;
  CoefRepr coef(rand());
  spar_mat.SetElem(x, y, coef);

  auto spar_mat_to_rmv_row = spar_mat;
  spar_mat_to_rmv_row.RemoveRow(x);
  if (row_num > 1) {
    auto new_rows = row_num - 1;
    EXPECT_EQ(spar_mat_to_rmv_row.rows, new_rows);
    EXPECT_EQ(spar_mat_to_rmv_row.cols, col_num);
    auto new_size = new_rows * col_num;
    EXPECT_EQ(spar_mat_to_rmv_row.indexes, std::vector<long>(new_size, -1));
  } else {
    EXPECT_EQ(spar_mat_to_rmv_row.rows, 0);
    EXPECT_EQ(spar_mat_to_rmv_row.cols, 0);
    EXPECT_TRUE(spar_mat_to_rmv_row.indexes.empty());
  }

  auto spar_mat_to_rmv_col = spar_mat;
  spar_mat_to_rmv_col.RemoveCol(y);
  if (col_num > 1) {
    auto new_cols = col_num - 1;
    EXPECT_EQ(spar_mat_to_rmv_col.rows, row_num);
    EXPECT_EQ(spar_mat_to_rmv_col.cols, new_cols);
    auto new_size = row_num * new_cols;
    EXPECT_EQ(spar_mat_to_rmv_col.indexes, std::vector<long>(new_size, -1));
  } else {
    EXPECT_EQ(spar_mat_to_rmv_col.rows, 0);
    EXPECT_EQ(spar_mat_to_rmv_col.cols, 0);
    EXPECT_TRUE(spar_mat_to_rmv_col.indexes.empty());
  }
}


TEST(TestSparCoefReprMat, RemoveRowAndCol) {
  RunTestSparCoefReprMatRemoveRowAndColCase(1, 1);
  RunTestSparCoefReprMatRemoveRowAndColCase(1, 5);
  RunTestSparCoefReprMatRemoveRowAndColCase(5, 1);
  RunTestSparCoefReprMatRemoveRowAndColCase(3, 5);
  RunTestSparCoefReprMatRemoveRowAndColCase(5, 3);
  RunTestSparCoefReprMatRemoveRowAndColCase(5, 5);
  RunTestSparCoefReprMatRemoveRowAndColCase(100, 100);
}


void RunTestSparCoefReprMatSwapTwoRowsAndColsCase(
    const size_t row_num, const size_t col_num) {
  SparCoefReprMat spar_mat(row_num, col_num);
  auto x = rand() % row_num;
  auto y = rand() % col_num;
  CoefRepr coef(rand());
  spar_mat.SetElem(x, y, coef);

  auto row_idx2 = rand() % row_num;
  auto row1 = spar_mat.GetRow(x);
  auto row2 = spar_mat.GetRow(row_idx2);
  auto spar_mat_to_swap_rows = spar_mat;
  spar_mat_to_swap_rows.SwapTwoRows(x, row_idx2);
  EXPECT_EQ(spar_mat_to_swap_rows.GetRow(x), row2);
  EXPECT_EQ(spar_mat_to_swap_rows.GetRow(row_idx2), row1);

  auto col_idx2  = rand() % col_num;
  auto col1 = spar_mat.GetCol(y);
  auto col2 = spar_mat.GetCol(col_idx2);
  auto spar_mat_to_swap_cols = spar_mat;
  spar_mat_to_swap_cols.SwapTwoCols(y, col_idx2);
  EXPECT_EQ(spar_mat_to_swap_cols.GetCol(y), col2);
  EXPECT_EQ(spar_mat_to_swap_cols.GetCol(col_idx2), col1);
}


TEST(TestSparCoefReprMat, SwapTwoRowsAndCols) {
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(1, 1);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(1, 5);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(5, 1);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(3, 5);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(5, 3);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(5, 5);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(100, 100);
}


void RunTestSparCoefReprMatTransposeRowsAndCols(
    const std::vector<size_t> &transposed_row_idxs,
    const std::vector<size_t> &transposed_col_idxs) {
  auto row_num = transposed_row_idxs.size();
  auto col_num = transposed_col_idxs.size();
  SparCoefReprMat spar_mat(row_num, col_num);
  int nonull_elem_num = 3;
  for (int i = 0; i < nonull_elem_num; ++i) {
    auto x = rand() % row_num;
    auto y = rand() % col_num;
    CoefRepr coef(rand());
    spar_mat.SetElem(x, y, coef);
  }

  auto spar_mat_to_tsps_rows = spar_mat;
  spar_mat_to_tsps_rows.TransposeRows(transposed_row_idxs);
  for (size_t i = 0; i < row_num; ++i) {
    EXPECT_EQ(
        spar_mat_to_tsps_rows.GetRow(i),
        spar_mat.GetRow(transposed_row_idxs[i]));
  }

  auto spar_mat_to_tsps_cols = spar_mat;
  spar_mat_to_tsps_cols.TransposeCols(transposed_col_idxs);
  for (size_t i = 0; i < col_num; ++i) {
    EXPECT_EQ(
        spar_mat_to_tsps_cols.GetCol(i),
        spar_mat.GetCol(transposed_col_idxs[i]));
  }
}


TEST(TestSparCoefReprMat, TransposeRowsAndCols) {
  RunTestSparCoefReprMatTransposeRowsAndCols({0, 1}, {0, 1});
  RunTestSparCoefReprMatTransposeRowsAndCols({1, 0}, {0, 1});
  RunTestSparCoefReprMatTransposeRowsAndCols({0, 1}, {1, 0});
  RunTestSparCoefReprMatTransposeRowsAndCols({1, 0}, {1, 0});
  RunTestSparCoefReprMatTransposeRowsAndCols({2, 1, 0}, {4, 3, 1, 0, 2});
  RunTestSparCoefReprMatTransposeRowsAndCols({4, 3, 1, 0, 2}, {1, 0, 2});
}


void RunTestSparOpReprMatInitializationCase(size_t row_num, size_t col_num) {
  SparOpReprMat spar_mat(row_num, col_num);
  EXPECT_EQ(spar_mat.rows, row_num);
  EXPECT_EQ(spar_mat.cols, col_num);
  EXPECT_TRUE(spar_mat.data.empty());
  auto size = row_num * col_num;
  auto indexes = spar_mat.indexes;
  EXPECT_EQ(indexes.size(), size);
  for (size_t i = 0; i < size; ++i) { EXPECT_EQ(indexes[i], -1); }
}


TEST(TestSparOpReprMat, Initialization) {
  SparOpReprMat null_op_repr_mat;
  EXPECT_EQ(null_op_repr_mat.rows, 0);
  EXPECT_EQ(null_op_repr_mat.cols, 0);
  EXPECT_TRUE(null_op_repr_mat.data.empty());
  EXPECT_TRUE(null_op_repr_mat.indexes.empty());

  RunTestSparOpReprMatInitializationCase(1, 1);
  RunTestSparOpReprMatInitializationCase(5, 1);
  RunTestSparOpReprMatInitializationCase(1, 5);
  RunTestSparOpReprMatInitializationCase(5, 3);
  RunTestSparOpReprMatInitializationCase(3, 5);
  RunTestSparOpReprMatInitializationCase(5, 5);
}
