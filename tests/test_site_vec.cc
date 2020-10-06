// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-29 16:48
* 
* Description: GraceQ/MPS2 project. Unittests for SiteVec .
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include "gtest/gtest.h"


using namespace gqmps2;
using namespace gqten;


using Tensor = DGQTensor;


template <typename TenT>
void TestIsIdOp(const TenT &ten) {
  EXPECT_EQ(ten.indexes.size(), 2);
  EXPECT_EQ(ten.shape[0], ten.shape[1]);
  EXPECT_EQ(ten.indexes[0].dir, IN);
  EXPECT_EQ(InverseIndex(ten.indexes[0]), ten.indexes[1]);

  auto dim = ten.shape[0];
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      if (i == j) {
        EXPECT_EQ(ten.Elem({i, j}), 1.0);
      } else {
        EXPECT_EQ(ten.Elem({i, j}), 0.0);
      }
    }
  }
}


void RunTestSiteVecBasicFeatures(
    const int N,
    const Index &local_hilbert_space
) {
  SiteVec<Tensor> site_vec(N, local_hilbert_space);
  EXPECT_EQ(site_vec.size, N);
  Index site;
  if (local_hilbert_space.dir == OUT) {
    site = local_hilbert_space;
  } else {
    site = InverseIndex(local_hilbert_space);
  }
  EXPECT_EQ(site_vec.sites, IndexVec(N, site));
  for (int i = 0; i < site_vec.size; ++i) {
    TestIsIdOp(site_vec.id_ops[i]);
  }
}


void RunTestSiteVecBasicFeatures(const IndexVec &local_hilbert_spaces) {
  SiteVec<Tensor> site_vec(local_hilbert_spaces);
  EXPECT_EQ(site_vec.size, local_hilbert_spaces.size());
  for (int i = 0; i < site_vec.size; ++i) {
    if (local_hilbert_spaces[i].dir == OUT) {
      EXPECT_EQ(site_vec.sites[i], local_hilbert_spaces[i]);
    } else {
      EXPECT_EQ(site_vec.sites[i], InverseIndex(local_hilbert_spaces[i]));
    }
    TestIsIdOp(site_vec.id_ops[i]);
  }
}


TEST(TestSiteVec, TestBasicFeatures) {
  Index pb_out1 = Index({
                      QNSector(QN({QNNameVal("N", 0)}), 1),
                      QNSector(QN({QNNameVal("N", 1)}), 1)},
                      OUT
                  );
  Index pb_in1 = InverseIndex(pb_out1);
  Index pb_out2 = Index({
                      QNSector(QN({QNNameVal("N", 1)}), 3)},
                      OUT
                  );
  Index pb_in2 = InverseIndex(pb_out2);
  RunTestSiteVecBasicFeatures(1, pb_out1);
  RunTestSiteVecBasicFeatures(1, pb_in1);
  RunTestSiteVecBasicFeatures(3, pb_out1);
  RunTestSiteVecBasicFeatures(3, pb_in1);

  RunTestSiteVecBasicFeatures({pb_out1});
  RunTestSiteVecBasicFeatures({pb_out2, pb_out2});
  RunTestSiteVecBasicFeatures({pb_out1, pb_out2, pb_out1});
  RunTestSiteVecBasicFeatures({pb_in2, pb_out1, pb_out1, pb_in1, pb_out2});
}
