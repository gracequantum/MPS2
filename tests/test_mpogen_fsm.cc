// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-11-18 18:25
* 
* Description: GraceQ/MPS2 project. Unittests for finite state machine used by MPO generator.
*/
#include "gqmps2/detail/mpogen/fsm.h"

#include "gtest/gtest.h"


void RunTestFSMInitializationCase(const size_t N) {
  if (N == 0) {
    FSM fsm;
    EXPECT_EQ(fsm.phys_size(), 0);
    EXPECT_EQ(fsm.fsm_size(), 1);
  }

  FSM fsm(N);
  EXPECT_EQ(fsm.phys_size(), N);
  EXPECT_EQ(fsm.fsm_size(), N+1);
}


TEST(TestFSM, Initialization) {
  RunTestFSMInitializationCase(0);  
  RunTestFSMInitializationCase(1);  
  RunTestFSMInitializationCase(5);  
  RunTestFSMInitializationCase(20);  
}


void RunTestAddPathCase1(void) {
  FSM fsm0(1);
  EXPECT_TRUE(fsm0.GetFSMPaths().empty());

  FSM fsm1(1);
  fsm1.AddPath(0, 0, {kIdOpRepr});
  auto fsm_paths = fsm1.GetFSMPaths();
  EXPECT_EQ(fsm_paths.size(), 1);
  FSMNode fsm_node1, fsm_node2;
  fsm_node1.fsm_site_idx = 0;
  fsm_node1.fsm_stat_idx = kFSMReadyStatIdx;
  fsm_node2.fsm_site_idx = 1;
  fsm_node2.fsm_stat_idx = kFSMFinalStatIdx;
  EXPECT_EQ(
      fsm_paths[0].fsm_nodes,
      FSMNodeVec({fsm_node1, fsm_node2}));
  EXPECT_EQ(fsm_paths[0].op_reprs, OpReprVec({kIdOpRepr}));

  FSM fsm2(1);
  OpLabel s(1);
  fsm2.AddPath(0, 0, {OpRepr(s)});
  fsm_paths = fsm2.GetFSMPaths();
  EXPECT_EQ(fsm_paths.size(), 1);
  EXPECT_EQ(
      fsm_paths[0].fsm_nodes,
      FSMNodeVec({fsm_node1, fsm_node2}));
  EXPECT_EQ(fsm_paths[0].op_reprs, OpReprVec({OpRepr(s)}));
}


void RunTestAddPathCase2(void) {
  auto s = OpRepr(1);
  FSM fsm(2);
  fsm.AddPath(0, 0, {s});
  fsm.AddPath(1, 1, {s});
  FSMNode fsm_node1, fsm_node2, fsm_node3, fsm_node4;
  fsm_node1.fsm_site_idx = 0;
  fsm_node1.fsm_stat_idx = 0;
  fsm_node2.fsm_site_idx = 1;
  fsm_node2.fsm_stat_idx = -1;
  fsm_node3.fsm_site_idx = 2;
  fsm_node3.fsm_stat_idx = -1;
  fsm_node4.fsm_site_idx = 1;
  fsm_node4.fsm_stat_idx = 0;

  auto fsm_paths = fsm.GetFSMPaths();
  EXPECT_EQ(fsm_paths.size(), 2);
  EXPECT_EQ(
      fsm_paths[0].fsm_nodes,
      FSMNodeVec({fsm_node1, fsm_node2, fsm_node3}));
  EXPECT_EQ(
      fsm_paths[0].op_reprs,
      OpReprVec({s, kIdOpRepr}));
  EXPECT_EQ(
      fsm_paths[1].fsm_nodes,
      FSMNodeVec({fsm_node1, fsm_node4, fsm_node3}));
  EXPECT_EQ(
      fsm_paths[1].op_reprs,
      OpReprVec({kIdOpRepr, s}));
}


void RunTestAddPathCase3(void) {
  auto s = OpRepr(1);
  FSM fsm(2);
  fsm.AddPath(0, 1, {s, s});
  fsm.AddPath(0, 1, {s, s});
  FSMNode n1, n2, n3, n4;
  n1.fsm_site_idx = 0;
  n1.fsm_stat_idx = 0;
  n2.fsm_site_idx = 1;
  n2.fsm_stat_idx = 1;
  n3.fsm_site_idx = 2;
  n3.fsm_stat_idx = -1;
  n4.fsm_site_idx = 1;
  n4.fsm_stat_idx = 2;

  auto paths = fsm.GetFSMPaths();
  EXPECT_EQ(paths[0].op_reprs, OpReprVec({s, s}));
  EXPECT_EQ(paths[0].fsm_nodes, FSMNodeVec({n1, n2, n3}));
  EXPECT_EQ(paths[1].op_reprs, OpReprVec({s, s}));
  EXPECT_EQ(paths[1].fsm_nodes, FSMNodeVec({n1, n4, n3}));
}


void RunTestAddPathCase4(void) {
  auto s = OpRepr(1);
  FSM fsm(4);
  fsm.AddPath(0, 1, {s, s});
  fsm.AddPath(1, 2, {s, s});
  fsm.AddPath(2, 3, {s, s});
  FSMNode n1, n2, n3, n4, n5, n6, n7, n8, n9;
  n1.fsm_site_idx = 0;
  n1.fsm_stat_idx = 0;
  n2.fsm_site_idx = 1;
  n2.fsm_stat_idx = 1;
  n3.fsm_site_idx = 2;
  n3.fsm_stat_idx = -1;
  n4.fsm_site_idx = 3;
  n4.fsm_stat_idx = -1;
  n5.fsm_site_idx = 4;
  n5.fsm_stat_idx = -1;
  n6.fsm_site_idx = 1;
  n6.fsm_stat_idx = 0;
  n7.fsm_site_idx = 2;
  n7.fsm_stat_idx = 1;
  n8.fsm_site_idx = 2;
  n8.fsm_stat_idx = 0;
  n9.fsm_site_idx = 3;
  n9.fsm_stat_idx = 1;

  auto paths = fsm.GetFSMPaths();
  EXPECT_EQ(paths[0].op_reprs, OpReprVec({s, s, kIdOpRepr, kIdOpRepr}));
  EXPECT_EQ(paths[0].fsm_nodes, FSMNodeVec({n1, n2, n3, n4, n5}));
  EXPECT_EQ(paths[1].op_reprs, OpReprVec({kIdOpRepr, s, s, kIdOpRepr}));
  EXPECT_EQ(paths[1].fsm_nodes, FSMNodeVec({n1, n6, n7, n4, n5}));
  EXPECT_EQ(paths[2].op_reprs, OpReprVec({kIdOpRepr, kIdOpRepr, s, s}));
  EXPECT_EQ(paths[2].fsm_nodes, FSMNodeVec({n1, n6, n8, n9, n5}));
}


void RunTestAddPathCase5(void) {
  auto s = OpRepr(1);
  FSM fsm(5);
 fsm.AddPath(0, 4, {s, kIdOpRepr, kIdOpRepr, kIdOpRepr, s}); 
 fsm.AddPath(1, 4, {s, s, kIdOpRepr, s});
 fsm.AddPath(1, 3, {s, s, s});
 FSMNode n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13;
 n1.fsm_site_idx = 0;
 n1.fsm_stat_idx = 0;
 n2.fsm_site_idx = 1;
 n2.fsm_stat_idx = 1;
 n3.fsm_site_idx = 2;
 n3.fsm_stat_idx = 1;
 n4.fsm_site_idx = 3;
 n4.fsm_stat_idx = 1;
 n5.fsm_site_idx = 4;
 n5.fsm_stat_idx = 1;
 n6.fsm_site_idx = 5;
 n6.fsm_stat_idx = -1;
 n7.fsm_site_idx = 1;
 n7.fsm_stat_idx = 0;
 n8.fsm_site_idx = 2;
 n8.fsm_stat_idx = 2;
 n9.fsm_site_idx = 3;
 n9.fsm_stat_idx = 2;
 n10.fsm_site_idx = 4;
 n10.fsm_stat_idx = 2;
 n11.fsm_site_idx = 2;
 n11.fsm_stat_idx = 3;
 n12.fsm_site_idx = 3;
 n12.fsm_stat_idx = 3;
 n13.fsm_site_idx = 4;
 n13.fsm_stat_idx = -1;

 auto paths = fsm.GetFSMPaths();
 EXPECT_EQ(paths[0].op_reprs, OpReprVec({s, kIdOpRepr, kIdOpRepr, kIdOpRepr, s}));
 EXPECT_EQ(paths[0].fsm_nodes, FSMNodeVec({n1, n2, n3, n4, n5, n6}));
 EXPECT_EQ(paths[1].op_reprs, OpReprVec({kIdOpRepr, s, s, kIdOpRepr, s}));
 EXPECT_EQ(paths[1].fsm_nodes, FSMNodeVec({n1, n7, n8, n9, n10, n6}));
 EXPECT_EQ(paths[2].op_reprs, OpReprVec({kIdOpRepr, s, s, s, kIdOpRepr}));
 EXPECT_EQ(paths[2].fsm_nodes, FSMNodeVec({n1, n7, n11, n12, n13, n6}));
}


TEST(TestFSM, TestAddPath) {
  RunTestAddPathCase1();
  RunTestAddPathCase2();
  RunTestAddPathCase3();
  RunTestAddPathCase4();
  RunTestAddPathCase5();
}
