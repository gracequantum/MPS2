// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-27 17:38
* 
* Description: GraceQ/MPS2 project. Implantation details for MPO generator.
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

#include <assert.h>

#ifdef Release
  #define NDEBUG
#endif


namespace gqmps2 {
using namespace gqten;


// Forward declarations.
template <typename TenElemType>
void AddOpToHeadMpoTen(
    GQTensor<TenElemType> *, const GQTensor<TenElemType> &, const long);

template <typename TenElemType>
void AddOpToTailMpoTen(
    GQTensor<TenElemType> *, const GQTensor<TenElemType> &, const long);

template <typename TenElemType>
void AddOpToCentMpoTen(
    GQTensor<TenElemType> *, const GQTensor<TenElemType> &,
    const long, const long);


// Helpers.
inline QN GetLvbTargetQN(const Index &index, const long coor) {
  auto coor_off_set_and_qnsct = index.CoorInterOffsetAndQnsct(coor);
  return coor_off_set_and_qnsct.qnsct.qn;
}


template <typename TenElemType>
inline void ResortNodes(std::vector<FSMNode<TenElemType> *> &sorted_nodes) {
  auto ref_idx = sorted_nodes.back()->mid_state_idx;
  for (auto it = sorted_nodes.begin(); it != sorted_nodes.end()-1; ++it) {
    if ((*it)->mid_state_idx >= ref_idx) { ++((*it)->mid_state_idx); }
  }
}


// MPO generator.
template <typename TenElemType>
MPOGenerator<TenElemType>::MPOGenerator(
    const long N, const Index &pb, const QN &zero_div) :
    N_(N),
    pb_out_(pb),
    ready_nodes_(N+1),
    final_nodes_(N+1),
    middle_nodes_set_(N+1),
    edges_set_(N),
    fsm_graph_merged_(false),
    relable_to_end_(false),
    zero_div_(zero_div),
    rvbs_(N-1),
    fsm_graph_sorted_(false) {
  pb_in_ = InverseIndex(pb_out_);    
  // Generate identity operator.
  auto id_op = GQTensor<TenElemType>({pb_in_, pb_out_});
  for (long i = 0; i < pb_out_.dim; ++i) { id_op({i, i}) = 1; }
  id_op_ = id_op;
  // Generate ready nodes and final nodes.
  for (long i = 0; i < N_+1; ++i) {
    auto ready_node = new FSMNode<TenElemType>(i);
    ready_node->is_ready = true;
    ready_node->mid_state_idx = 0;        // For MPO generation process.
    ready_nodes_[i] = ready_node;
    auto final_node = new FSMNode<TenElemType>(i);
    final_node->is_final = true;
    final_node->mid_state_idx = -1;       // For MPO generation process.
    final_nodes_[i] = final_node;
  }
  // Generate R -> R and F -> F identity finite state machine edges.
  for (long i = 0; i < N_; ++i) {
    if (i != N_-1) {
      auto r2r_edge = new FSMEdge<TenElemType>(
                              id_op_,
                              ready_nodes_[i],
                              ready_nodes_[i+1],
                              i);
      edges_set_[i].push_back(r2r_edge);
    }
    if (i != 0) {
      auto f2f_edge = new FSMEdge<TenElemType>(
                              id_op_,
                              final_nodes_[i],
                              final_nodes_[i+1],
                              i);
      edges_set_[i].push_back(f2f_edge);
    }
  }
}


template <typename TenElemType>
void MPOGenerator<TenElemType>::AddTerm(
    const double coef,
    const std::vector<OpIdx<TenElemType>> &opidxs,
    const GQTensor<TenElemType> &inter_op) {
  switch (opidxs.size()) {
    case 1:
      AddOneSiteTerm(coef, opidxs[0]);
      break;
    case 2:
      AddTwoSiteTerm(coef, opidxs[0], opidxs[1], inter_op);
      break;
    default:
      std::cout << "Unsupport term type." << std::endl;
     exit(1); 
  }
  fsm_graph_merged_ = false;
}


template <typename TenElemType>
std::vector<GQTensor<TenElemType> *> MPOGenerator<TenElemType>::Gen(void) {
  if (!fsm_graph_merged_) {
    FSMGraphMerge();
    fsm_graph_merged_ = true;
    fsm_graph_sorted_ = false;
  }
  // Print MPO tensors virtual bond dimension.
  for (auto &middle_nodes : middle_nodes_set_) {
    std::cout << std::setw(3) << middle_nodes.size() + 2 << std::endl;
  }
  // Generation process.
  if (!fsm_graph_sorted_) {
    FSMGraphSort();
    fsm_graph_sorted_ = true;
  }
  std::vector<GQTensor<TenElemType> *> mpo(N_);
  for (long i = 0; i < N_; ++i) {
    if (i == 0) {
      mpo[i] = GenHeadMpo();
    } else if (i == N_-1) {
      mpo[i] = GenTailMpo();
    } else {
      mpo[i] = GenCentMpo(i);
    }
  }
  return mpo;
}


template <typename TenElemType>
void MPOGenerator<TenElemType>::AddOneSiteTerm(
    const double coef, const OpIdx<TenElemType> &opidx) {
  auto new_edge = new FSMEdge<TenElemType>(
                          coef*opidx.op,
                          ready_nodes_[opidx.idx],
                          final_nodes_[opidx.idx+1],
                          opidx.idx);
  edges_set_[opidx.idx].push_back(new_edge);
}


template <typename TenElemType>
void MPOGenerator<TenElemType>::AddTwoSiteTerm(
    const double coef,
    const OpIdx<TenElemType> &opidx1, const OpIdx<TenElemType> &opidx2,
    const GQTensor<TenElemType> &inter_op) {
  assert(opidx1.idx < opidx2.idx);
    GQTensor<TenElemType> itrop;   // Inter operator.
    if (inter_op == kNullOperator<TenElemType>) {
      itrop  = id_op_;
    } else {
      itrop = inter_op;
  }

  auto last_node = ready_nodes_[opidx1.idx];
  auto next_node = new FSMNode<TenElemType>(opidx1.idx+1);
  next_node->mid_state_idx = middle_nodes_set_[opidx1.idx+1].size() + 1;
  auto new_edge = new FSMEdge<TenElemType>(
                          coef*opidx1.op, last_node, next_node, opidx1.idx);
  next_node->ledges.push_back(new_edge);
  edges_set_[opidx1.idx].push_back(new_edge);
  middle_nodes_set_[opidx1.idx+1].push_back(next_node);

  for (long i = opidx1.idx+1; i < opidx2.idx; ++i) {
    last_node = next_node;
    next_node = new FSMNode<TenElemType>(i+1);
    next_node->mid_state_idx = middle_nodes_set_[i+1].size() + 1;
    new_edge = new FSMEdge<TenElemType>(itrop, last_node, next_node, i);
    last_node->redges.push_back(new_edge);
    next_node->ledges.push_back(new_edge);
    edges_set_[i].push_back(new_edge);
    middle_nodes_set_[i+1].push_back(next_node);
  }

  last_node = next_node;
  next_node = final_nodes_[opidx2.idx+1];
  new_edge = new FSMEdge<TenElemType>(
                     opidx2.op, last_node, next_node, opidx2.idx);
  last_node->redges.push_back(new_edge);
  edges_set_[opidx2.idx].push_back(new_edge);
}


// Merge finite state machine graph.
template <typename TenElemType>
void MPOGenerator<TenElemType>::FSMGraphMerge(void) {
  for (long i = 1; i < N_; ++i) {
    FSMGraphMergeAt(i);
    if (relable_to_end_) {
      for (long j = i+1; j < N_; ++j) {
        RelabelMidNodesIdx(j);
      }
      relable_to_end_ = false;
    }
  }
  RemoveNullEdges();
}


template <typename TenElemType>
void MPOGenerator<TenElemType>::FSMGraphMergeAt(const long nodes_set_idx) {
  std::vector<FSMNode<TenElemType> *> &rmiddle_nodes =
                                      middle_nodes_set_[nodes_set_idx];
  auto merge_happend = false;
  for (std::size_t i = 0; i < rmiddle_nodes.size(); i++) {
    merge_happend = FSMGraphMergeNodesTo(i, rmiddle_nodes);
    if (merge_happend) { break; }
  }
  if (merge_happend) {
      RelabelMidNodesIdx(nodes_set_idx);
      FSMGraphMergeAt(nodes_set_idx);
  }
}


template <typename TenElemType>
bool MPOGenerator<TenElemType>::FSMGraphMergeNodesTo(
    const long target_mid_node_idx,
    std::vector<FSMNode<TenElemType> *> &middle_nodes) {
  auto merge_happend = false;
  for (long i = target_mid_node_idx+1; i < middle_nodes.size(); ++i) {
    merge_happend = FSMGraphMergeTwoNodes(
                        middle_nodes[target_mid_node_idx], middle_nodes[i]);
    if (merge_happend) {
      return merge_happend;
    }
  }
  return merge_happend;
}


template <typename TenElemType>
bool MPOGenerator<TenElemType>::FSMGraphMergeTwoNodes(
    FSMNode<TenElemType> *&target, FSMNode<TenElemType> *&from) {
  // Check possible merged left edges.
  std::vector<std::size_t> lposs_merge_edge_idxs;
  for (std::size_t i = 0; i < target->ledges.size(); ++i) {
    for (std::size_t j = 0; j < from->ledges.size(); ++j) {
      auto can_merge = CheckMergeableLeftEdgePair(
                           target->ledges[i],
                           from->ledges[j]);
      if (can_merge) {
        lposs_merge_edge_idxs.push_back(j);
      }
    }
  }
  // Check possible merged right edges.
  std::vector<std::size_t> rposs_merge_edge_idxs;
  for (std::size_t i = 0; i < target->redges.size(); ++i) {
    for (std::size_t j = 0; j < from->redges.size(); ++j) {
      auto can_merge = CheckMergeableRightEdgePair(
                           target->redges[i],
                           from->redges[j]);
      if (can_merge) {
        rposs_merge_edge_idxs.push_back(j);
      }
    }
  }
  // Check whether two nodes can merge and merge them.
  if ((target->ledges.size()*target->redges.size() + from->ledges.size()*from->redges.size()) ==
      ((target->ledges.size() + from->ledges.size() - lposs_merge_edge_idxs.size())*
       (target->redges.size() + from->redges.size() - rposs_merge_edge_idxs.size()))) {
    // Merge two nodes.
    // Deal with left edges.
    for (std::size_t i = 0; i < from->ledges.size(); ++i) {
      auto working_edge = from->ledges[i];
      if (std::find(lposs_merge_edge_idxs.cbegin(),
                    lposs_merge_edge_idxs.cend(), i) !=
                    lposs_merge_edge_idxs.cend()) {
        auto working_site_idx = working_edge->loc;
        auto working_edge_it = std::find(edges_set_[working_site_idx].begin(),
                                         edges_set_[working_site_idx].end(),
                                         working_edge);
        delete *working_edge_it;
        *working_edge_it = nullptr;
      } else {
        working_edge->next_node = target;
        target->ledges.push_back(working_edge);
      }
    }
    // Deal with right edges.
    for (std::size_t i = 0; i < from->redges.size(); ++i) {
      auto working_edge = from->redges[i];
      if (std::find(rposs_merge_edge_idxs.cbegin(),
                    rposs_merge_edge_idxs.cend(), i) !=
                    rposs_merge_edge_idxs.cend()) {
        DeletePathToRightEnd(working_edge);
      } else {
        working_edge->last_node = target;
        target->redges.push_back(working_edge);
      }
    }
    delete from; from = nullptr;
    return true;
    // Not merge.
  } else {
    return false;
  }
}


template <typename TenElemType>
inline bool MPOGenerator<TenElemType>::CheckMergeableLeftEdgePair(
    const FSMEdge<TenElemType> *ledge1, const FSMEdge<TenElemType> *ledge2) {
  if (ledge1->op == ledge2->op && ledge1->last_node == ledge2->last_node) {
    return true;
  } else {
    return false;
  }
}


template <typename TenElemType>
inline bool MPOGenerator<TenElemType>::CheckMergeableRightEdgePair(
    const FSMEdge<TenElemType> *redge1, const FSMEdge<TenElemType> *redge2) {
  const FSMEdge<TenElemType> *pnext_edge1 = redge1;
  const FSMEdge<TenElemType> *pnext_edge2 = redge2;
  while (true) {
    if (pnext_edge1->op != pnext_edge2->op) {
      return false;
    }
    if (pnext_edge1->next_node->is_final &&
        pnext_edge2->next_node->is_final) {
      return true; 
    } else if (!pnext_edge1->next_node->is_final &&
               !pnext_edge2->next_node->is_final) {
      assert(pnext_edge1->next_node->redges.size() == 1);
      assert(pnext_edge2->next_node->redges.size() == 1);
      pnext_edge1 = pnext_edge1->next_node->redges[0];
      pnext_edge2 = pnext_edge2->next_node->redges[0];
    } else {
      return false;
    }
  }
}


template <typename TenElemType>
void MPOGenerator<TenElemType>::DeletePathToRightEnd(
    FSMEdge<TenElemType> *edge) {
  auto this_edge = edge;
  while(true) {
    if (this_edge->next_node->is_final) {
      auto this_edge_loc = this_edge->loc;
      auto this_edge_it = std::find(edges_set_[this_edge_loc].begin(),
                                    edges_set_[this_edge_loc].end(),
                                    this_edge);
      delete *this_edge_it;
      *this_edge_it = nullptr;
      break;
    } else {
      assert(this_edge->next_node->redges.size() == 1);
      auto next_edge = this_edge->next_node->redges[0];
      auto next_node_loc = this_edge->next_node->loc;
      auto next_node_it = std::find(middle_nodes_set_[next_node_loc].begin(),
                                    middle_nodes_set_[next_node_loc].end(),
                                    this_edge->next_node);
      delete *next_node_it;
      *next_node_it = nullptr;
      auto this_edge_loc = this_edge->loc;
      auto this_edge_it = std::find(edges_set_[this_edge_loc].begin(),
                                    edges_set_[this_edge_loc].end(),
                                    this_edge);
      delete *this_edge_it;
      *this_edge_it = nullptr;
      this_edge = next_edge;
    }
  }
  relable_to_end_ = true;
}


template <typename TenElemType>
void MPOGenerator<TenElemType>::RelabelMidNodesIdx(const long nodes_set_idx) {
  std::vector<FSMNode<TenElemType> *> new_middle_nodes;
  long new_mid_state_idx = 1;
  for (auto &pmid_node : middle_nodes_set_[nodes_set_idx]) {
    if (pmid_node != nullptr) {
      pmid_node->mid_state_idx = new_mid_state_idx;
      new_middle_nodes.push_back(pmid_node);
      ++new_mid_state_idx;
    }
  }
  middle_nodes_set_[nodes_set_idx] = new_middle_nodes;
}


template <typename TenElemType>
void MPOGenerator<TenElemType>::RemoveNullEdges(void) {
  for (auto &edges : edges_set_) {
    std::vector<FSMEdge<TenElemType> *> new_edges;
    for (auto &edge : edges) {
      if (edge != nullptr) { new_edges.push_back(edge); }
    }
    edges = new_edges;
  }
}


// Gnerator MPO tensors.
// Sort finite state machine nodes by quantum numbers.
template <typename TenElemType>
void MPOGenerator<TenElemType>::FSMGraphSort(void) {
  for (long i = 0; i < N_ - 1; ++i) {
    rvbs_[i] = FSMGraphSortAt(i);
  }
}


template <typename TenElemType>
Index MPOGenerator<TenElemType>::FSMGraphSortAt(const long site_idx) {
  // Sort R and F states.
  ready_nodes_[site_idx+1]->mid_state_idx = 0;
  final_nodes_[site_idx+1]->mid_state_idx = 1;
  std::vector<FSMNode<TenElemType> *> temp_sorted_nodes;
  temp_sorted_nodes.push_back(ready_nodes_[site_idx+1]);
  temp_sorted_nodes.push_back(final_nodes_[site_idx+1]);
  std::vector<QNSector> rvb_qnscts;
  rvb_qnscts.push_back(QNSector(zero_div_, 2));
  for (auto &edge : edges_set_[site_idx]) {
    if (std::find(temp_sorted_nodes.begin(),
                  temp_sorted_nodes.end(),
                  edge->next_node) == temp_sorted_nodes.end()) {
      QN rvb_qn;
      if (site_idx == 0) {
        rvb_qn = zero_div_ - Div(edge->op);
      } else {
        rvb_qn = zero_div_ - Div(edge->op) +
                 GetLvbTargetQN(
                     rvbs_[site_idx-1],
                     edge->last_node->mid_state_idx);
      }
      auto has_qn = false;
      long offset = 0;
      for (auto &qnsct : rvb_qnscts) {
        if (qnsct.qn == rvb_qn) {
          edge->next_node->mid_state_idx = offset + qnsct.dim;
          temp_sorted_nodes.push_back(edge->next_node);
          ResortNodes(temp_sorted_nodes);
          qnsct.dim += 1;
          has_qn = true;
          break;
        } else {
          offset += qnsct.dim;
        }
      }
      if (!has_qn) {
        rvb_qnscts.push_back(QNSector(rvb_qn, 1));
        edge->next_node->mid_state_idx = offset;
        temp_sorted_nodes.push_back(edge->next_node);
      }
    }
  }
  return Index(rvb_qnscts, OUT);
}


template <typename TenElemType>
GQTensor<TenElemType> *MPOGenerator<TenElemType>::GenHeadMpo(void) {
  auto pmpo_ten = new GQTensor<TenElemType>({pb_in_, rvbs_[0], pb_out_});
  for (auto &edge : edges_set_[0]) {
    AddOpToHeadMpoTen(pmpo_ten, edge->op, edge->next_node->mid_state_idx);
  }
  return pmpo_ten;
}


template <typename TenElemType>
GQTensor<TenElemType> *MPOGenerator<TenElemType>::GenTailMpo(void) {
  auto lvb = InverseIndex(rvbs_.back());
  auto pmpo_ten = new GQTensor<TenElemType>({pb_in_, lvb, pb_out_});
  for (auto &edge : edges_set_[N_-1]) {
    AddOpToTailMpoTen(pmpo_ten, edge->op, edge->last_node->mid_state_idx);
  }
  return pmpo_ten;
}


template <typename TenElemType>
GQTensor<TenElemType> *MPOGenerator<TenElemType>::GenCentMpo(const long site_idx) {
  auto lvb = InverseIndex(rvbs_[site_idx-1]);
  auto pmpo_ten = new GQTensor<TenElemType>({lvb, pb_in_, pb_out_, rvbs_[site_idx]});
  for (auto &edge : edges_set_[site_idx]) {
    AddOpToCentMpoTen(
        pmpo_ten, edge->op,
        edge->last_node->mid_state_idx, edge->next_node->mid_state_idx);
  }
  return pmpo_ten;
}


template <typename TenElemType>
void AddOpToHeadMpoTen(
    GQTensor<TenElemType> *pmpo_ten, const GQTensor<TenElemType> &rop, const long rvb_coor) {
  for (long bpb_coor = 0; bpb_coor < rop.indexes[0].dim; ++bpb_coor) {
    for (long tpb_coor = 0; tpb_coor < rop.indexes[1].dim; ++tpb_coor) {
      auto elem = rop.Elem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)({bpb_coor, rvb_coor, tpb_coor}) = elem;
      }
    }
  }
}


template <typename TenElemType>
void AddOpToTailMpoTen(
    GQTensor<TenElemType> *pmpo_ten, const GQTensor<TenElemType> &rop, const long lvb_coor) {
  for (long bpb_coor = 0; bpb_coor < rop.indexes[0].dim; ++bpb_coor) {
    for (long tpb_coor = 0; tpb_coor < rop.indexes[1].dim; ++tpb_coor) {
      auto elem = rop.Elem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)({bpb_coor, lvb_coor, tpb_coor}) = elem;
      }
    }
  }
}


template <typename TenElemType>
void AddOpToCentMpoTen(
    GQTensor<TenElemType> *pmpo_ten, const GQTensor<TenElemType> &rop,
    const long lvb_coor, const long rvb_coor) {
  for (long bpb_coor = 0; bpb_coor < rop.indexes[0].dim; ++bpb_coor) {
    for (long tpb_coor = 0; tpb_coor < rop.indexes[1].dim; ++tpb_coor) {
      auto elem = rop.Elem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)({lvb_coor, bpb_coor, tpb_coor, rvb_coor}) = elem;
      }
    }
  }
}
} /* gqmps2 */ 
