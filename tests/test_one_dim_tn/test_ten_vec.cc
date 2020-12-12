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

using U1QN = QN<U1QNVal>;
using QNT = U1QN;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using Tensor = DGQTensor;


TEST(TestTenVec, TestIO) {
  QNT qn0 = QNT({QNCard("N",  U1QNVal( 0))});
  QNT qn1 = QNT({QNCard("N",  U1QNVal( 1))});
  QNT qnm1 = QNT({QNCard("N", U1QNVal(-1))});
  IndexT idx_out = IndexT(
                       {QNSctT(qn0, 2), QNSctT(qn1, 2)},
                       GQTenIndexDirType::OUT
                   );
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
