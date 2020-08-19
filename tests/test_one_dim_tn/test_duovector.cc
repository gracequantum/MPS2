// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-19 17:33
* 
* Description: GraceQ/MPS2 project. Unittests for DuoVector .
*/
#include "gqmps2/one_dim_tn/framework/duovector.h"

#include "gtest/gtest.h"


using namespace gqmps2;


template <typename ElemT>
void RunTestDuoVectorConstructorsCase(const size_t size) {
  DuoVector<ElemT> duovec(size);
  EXPECT_EQ(duovec.size(), size);
  
  auto craw_data = duovec.cdata();
  for (auto &rpelem : craw_data) {
    EXPECT_EQ(rpelem, nullptr);
  }
}

TEST(TestDuoVector, TestConstructors) {
  RunTestDuoVectorConstructorsCase<int>(0);
  RunTestDuoVectorConstructorsCase<int>(1);
  RunTestDuoVectorConstructorsCase<int>(3);

  RunTestDuoVectorConstructorsCase<double>(0);
  RunTestDuoVectorConstructorsCase<double>(1);
  RunTestDuoVectorConstructorsCase<double>(3);
}


TEST(TestDuoVector, TestElemAccess) {
  DuoVector<int> intduovec(1);

  intduovec[0] = 3;
  EXPECT_EQ(intduovec[0], 3);

  auto pelem = intduovec.cdata()[0];
  intduovec[0] = 5;
  EXPECT_EQ(intduovec.cdata()[0], pelem);
  EXPECT_EQ(intduovec[0], 5);

  auto pelem2 = new int(4);
  delete intduovec(0);
  intduovec(0) = pelem2;
  EXPECT_EQ(intduovec[0], 4);
  EXPECT_NE(intduovec.cdata()[0], pelem);
  EXPECT_EQ(intduovec.cdata()[0], pelem2);
}


TEST(TestDuoVector, TestElemAllocDealloc) {
  DuoVector<int> intduovec(2);

  intduovec.alloc(0);
  EXPECT_NE(intduovec.cdata()[0], nullptr);
  EXPECT_EQ(intduovec.cdata()[1], nullptr);

  intduovec[0] = 3;
  EXPECT_EQ(intduovec[0], 3);

  intduovec.dealloc(0);
  EXPECT_EQ(intduovec.cdata()[0], nullptr);

  intduovec.dealloc(1);
  EXPECT_EQ(intduovec.cdata()[1], nullptr);
}
