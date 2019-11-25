// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-11-18 18:28
* 
* Description: GraceQ/MPS2 project. Finite state machine used by MPO generator.
*/
#ifndef GQMPS2_DETAIL_MPOGEN_FSM
#define GQMPS2_DETAIL_MPOGEN_FSM


#include "gqmps2/detail/mpogen/coef_op_alg.h"

#include <vector>

#include <assert.h>

#ifdef Release
  #define NDEBUG
#endif


struct FSMNode {
  size_t fsm_site_idx;
  long fsm_stat_idx;
};


bool operator==(const FSMNode &lhs, const FSMNode &rhs) {
  return (lhs.fsm_site_idx == rhs.fsm_site_idx) &&
         (lhs.fsm_stat_idx == rhs.fsm_stat_idx);
}


bool operator!=(const FSMNode &lhs, const FSMNode &rhs) {
  return !(lhs == rhs);
}


using FSMNodeVec = std::vector<FSMNode>;


struct FSMPath {
  FSMPath(const size_t phys_site_num, const size_t fsm_site_num) :
      fsm_nodes(fsm_site_num), op_reprs(phys_site_num) {
    assert(fsm_nodes.size() == op_reprs.size() + 1); 
  }
  FSMNodeVec fsm_nodes;
  OpReprVec op_reprs;
};


class FSM {
public:
  FSM(const size_t phys_site_num) :
      phys_site_num_(phys_site_num),
      fsm_site_num_(phys_site_num+1),
      mid_stat_nums_(phys_site_num+1, 0),
      has_readys_(phys_site_num+1, false) {
    assert(fsm_site_num_ == phys_site_num_ + 1); 
  }
  
  FSM(void) : FSM(0) {}

  size_t phys_size(void) const { return phys_site_num_; }

  size_t fsm_size(void) const { return fsm_site_num_; }

  void AddPath(const size_t, const size_t, const OpReprVec &);

  std::vector<FSMPath> GetFSMPaths(void) const { return fsm_paths_; }

  std::vector<SparOpReprMat> GenMatRepr(void) const;

  std::vector<SparOpReprMat> GenCompressedMatRepr(void) const;


private:
  size_t phys_site_num_;
  size_t fsm_site_num_;
  std::vector<size_t> mid_stat_nums_;
  std::vector<bool> has_readys_;
  std::vector<FSMPath> fsm_paths_;
};


const long kFSMReadyStatIdx = 0;

const long kFSMFinalStatIdx = -1;


void FSM::AddPath(
      const size_t head_ntrvl_site_idx, const size_t tail_ntrvl_site_idx,
      const OpReprVec &ntrvl_ops) {
  assert(
      head_ntrvl_site_idx + ntrvl_ops.size() +
      (phys_site_num_ - tail_ntrvl_site_idx - 1) == phys_site_num_);
  FSMPath fsm_path(phys_site_num_, fsm_site_num_);
  // Set operator representations.
  for (size_t i = 0; i < phys_site_num_; ++i) {
    if (i < head_ntrvl_site_idx) {
      fsm_path.op_reprs[i] = kIdOpRepr;
    } else if (i > tail_ntrvl_site_idx) {
      fsm_path.op_reprs[i] = kIdOpRepr;
    } else {
      fsm_path.op_reprs[i] = ntrvl_ops[i-head_ntrvl_site_idx];
    }
  }
  // Set FSM nodes.
  fsm_path.fsm_nodes[0].fsm_site_idx = 0;
  fsm_path.fsm_nodes[0].fsm_stat_idx = kFSMReadyStatIdx;
  has_readys_[0] = true;
  for (size_t i = 0; i < phys_site_num_; ++i) {
    size_t tgt_fsm_site_idx = i + 1;
    if (i < head_ntrvl_site_idx) {
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_site_idx = tgt_fsm_site_idx;
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_stat_idx = kFSMReadyStatIdx;
      has_readys_[tgt_fsm_site_idx] = true;
    } else if (i >= tail_ntrvl_site_idx) {
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_site_idx = tgt_fsm_site_idx;
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_stat_idx = kFSMFinalStatIdx;
    } else {
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_site_idx = tgt_fsm_site_idx;
      mid_stat_nums_[tgt_fsm_site_idx]++;
      fsm_path.fsm_nodes[tgt_fsm_site_idx].fsm_stat_idx = 
          mid_stat_nums_[tgt_fsm_site_idx];
    }
  }
  fsm_paths_.push_back(fsm_path);
}
#endif /* ifndef GQMPS2_DETAIL_MPOGEN_FSM */
