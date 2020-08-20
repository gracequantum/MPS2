// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-20 17:29
* 
* Description: GraceQ/MPS2 project. Unittests for TenVec .
*/
#include "gqmps2/one_dim_tn/framework/ten_vec.h"
#include "gqten/gqten.h"

#include "gtest/gtest.h"

using namespace gqmps2;
using namespace gqten;

using Tensor = DGQTensor;


TEST(TestTenVec, TestIO) {
  QN qn0 = QN({QNNameVal("N", 0)});
  QN qn1 = QN({QNNameVal("N", 1)});
  QN qnm1 = QN({QNNameVal("N", -1)});
  Index idx_out = Index({QNSector(qn0, 2), QNSector(qn1, 2)}, OUT);
  auto idx_in = InverseIndex(idx_out);
  Tensor ten0({idx_in, idx_out});
  Tensor ten1({idx_in, idx_out});
  Tensor ten2({idx_in, idx_out});
  ten0.Random(qn0);
  ten1.Random(qn1);
  ten2.Random(qnm1);

  TenVec<Tensor> tenvec(3);
  tenvec[0] = ten0;
  tenvec[1] = ten1;
  tenvec[2] = ten2;
  tenvec.DumpTen(0, "ten0." + kGQTenFileSuffix);
  tenvec.DumpTen(1, "ten1." + kGQTenFileSuffix);
  tenvec.DumpTen(2, "ten2." + kGQTenFileSuffix);
  tenvec.dealloc(0);
  tenvec.dealloc(1);
  tenvec.dealloc(2);
  tenvec.LoadTen(0, "ten2." + kGQTenFileSuffix);
  tenvec.LoadTen(1, "ten0." + kGQTenFileSuffix);
  tenvec.LoadTen(2, "ten1." + kGQTenFileSuffix);
  EXPECT_EQ(tenvec[0], ten2);
  EXPECT_EQ(tenvec[1], ten0);
  EXPECT_EQ(tenvec[2], ten1);
}
