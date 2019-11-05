// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-10-30 18:32
* 
* Description: GraceQ/MPS2 project. Data structure of a special sparse matrix.
*/
#ifndef GQMPS2_DETAIL_MPOGEN_SPARSE_MAT_H
#define GQMPS2_DETAIL_MPOGEN_SPARSE_MAT_H


#include <vector>


template <typename ElemType>
class SparMat {
public:
  SparMat(void) : rows(0), cols(0), data(), indexes() {}

  SparMat(const size_t row_num, const size_t col_num) :
      rows(row_num), cols(col_num),
      data(), indexes(row_num*col_num, -1) {}

  SparMat(const SparMat<ElemType> &spar_mat) :
      rows(spar_mat.rows), cols(spar_mat.cols),
      data(spar_mat.data), indexes(spar_mat.indexes) {}

  SparMat<ElemType> &operator=(const SparMat<ElemType> &spar_mat) {
    rows = spar_mat.rows;
    cols = spar_mat.cols;
    data = spar_mat.data;
    indexes = spar_mat.indexes; 
    return *this;
  }

  // Element getter and setter.
  const ElemType &operator()(const size_t x, const size_t y) const {
    auto offset = CalcOffset(x, y);
    if (indexes[offset] == -1) {
      return nullelem; 
    } else {
      return data[indexes[offset]];
    }
  }

  void SetElem(const size_t x, const size_t y, const ElemType &elem) {
    if (elem == nullelem) { return; }
    auto offset = CalcOffset(x, y);
    if (indexes[offset] == -1) {
      data.push_back(elem);
      long idx = data.size() - 1;
      indexes[offset] = idx;
    } else {
      data[indexes[offset]] = elem;
    }
  }

  // Get row and column.
  std::vector<ElemType> GetRow(const size_t row_idx) const {
    assert(row_idx < rows); 
    std::vector<ElemType> row;
    row.reserve(cols);
    for (size_t y = 0; y < cols; ++y) {
      row.push_back((*this)(row_idx, y));
    }
    return row;
  }
  
  std::vector<ElemType> GetCol(const size_t col_idx) const {
    assert(col_idx < cols); 
    std::vector<ElemType> col;
    col.reserve(rows);
    for (size_t x = 0; x < rows; ++x) {
      col.push_back((*this)(x, col_idx));
    }
    return col;
  }

  // Remove row and column.
  void RemoveRow(const size_t row_idx) {
    assert(row_idx < rows); 
    if (rows == 1) {
      *this = SparMat<ElemType>();
      return;
    }
    auto new_rows = rows - 1;
    auto new_size = new_rows * cols;
    std::vector<long> new_indexes(new_size);
    for (size_t x = 0; x < rows; ++x) {
      for (size_t y = 0; y < cols; ++y) {
        if (x < row_idx) {
          new_indexes[CalcOffset(x, y)] = indexes[CalcOffset(x, y)];
        } else if (x > row_idx) {
          new_indexes[CalcOffset(x-1, y)] = indexes[CalcOffset(x, y)];
        }
      }
    }
    rows = new_rows;
    indexes = new_indexes;
  }

  void RemoveCol(const size_t col_idx) {
    assert(col_idx < cols);
    if (cols == 1) {
      *this = SparMat<ElemType>();
      return;
    }
    auto new_cols = cols - 1;
    auto new_size = rows * new_cols;
    std::vector<long> new_indexes(new_size);
    for (size_t x = 0; x < rows; ++x) {
      for (size_t y = 0; y < cols; ++y) {
        if (y < col_idx) {
          new_indexes[CalcOffset_(x, y, new_cols)] = indexes[CalcOffset(x, y)];
        } else if (y > col_idx) {
          new_indexes[CalcOffset_(x, y-1, new_cols)] =
              indexes[CalcOffset(x, y)];
        }
      }
    }
    cols = new_cols;
    indexes = new_indexes;
  }

  // Swap two rows and columns.
  void SwapTwoRows(const size_t row_idx1, const size_t row_idx2) {
    assert(row_idx1 < rows && row_idx2 < rows);
    if (row_idx1 == row_idx2) { return; }
    for (size_t y = 0; y < cols; ++y) {
      auto offset1 = CalcOffset(row_idx1, y);
      auto offset2 = CalcOffset(row_idx2, y);
      auto temp = indexes[offset1];
      indexes[offset1] = indexes[offset2];
      indexes[offset2] = temp;
    }
  }

  void SwapTwoCols(const size_t col_idx1, const size_t col_idx2) {
    assert(col_idx1 < cols && col_idx2 < cols);
    if (col_idx1 == col_idx2) { return; }
    for (size_t x = 0; x < rows; ++x) {
      auto offset1 = CalcOffset(x, col_idx1);
      auto offset2 = CalcOffset(x, col_idx2);
      auto temp = indexes[offset1];
      indexes[offset1] = indexes[offset2];
      indexes[offset2] = temp;
    }
  }

  // Transpose rows and columns.
  void TransposeRows(const std::vector<size_t> &transposed_row_idxs) {
    assert(transposed_row_idxs.size() == rows);
    std::vector<long> new_indexes(indexes.size());
    for (size_t i = 0; i < rows; ++i) {
      auto transposed_row_idx = transposed_row_idxs[i];
      for (size_t y = 0; y < cols; ++y) {
        new_indexes[CalcOffset(i, y)] =
            indexes[CalcOffset(transposed_row_idx, y)];
      }
    }
    indexes = new_indexes;
  }

  void TransposeCols(const std::vector<size_t> &transposed_col_idxs) {
    assert(transposed_col_idxs.size() == cols);
    std::vector<long> new_indexes(indexes.size());
    for (size_t i = 0; i < cols; ++i) {
      auto transposed_col_idx = transposed_col_idxs[i];
      for (size_t x = 0; x < rows; ++x) {
        new_indexes[CalcOffset(x, i)] =
            indexes[CalcOffset(x, transposed_col_idx)];
      }
    }
    indexes = new_indexes;
  }
  
  size_t CalcOffset(const size_t x, const size_t y) const {
    return x*cols + y;
  }

  size_t rows;
  size_t cols;
  std::vector<ElemType> data;
  std::vector<long> indexes;

private:
  size_t CalcOffset_(
      const size_t x, const size_t y, const size_t new_cols) const {
    return x*new_cols + y; 
  }

  static ElemType nullelem;
};


template<typename ElemType>
ElemType SparMat<ElemType>::nullelem = ElemType();
#endif /* ifndef GQMPS2_DETAIL_MPOGEN_SPARSE_MAT_H */
