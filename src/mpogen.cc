/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-13 15:13
* 
* Description: GraceQ/mps2 project. Private objects used by MPO generation, implementation.
*/
#include "mpogen.h"
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <iostream>
#include <iomanip>
#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>


namespace gqmps2 {
using namespace gqten;


MPOGenerator::MPOGenerator(const long N, const Index &pb) :
    N_(N),
    pb_out_(pb),
    ready_nodes_(N+1),
    final_nodes_(N+1),
    middle_nodes_set_(N+1),
    edges_set_(N),
    fsm_graph_merged_(false),
    relable_to_end_(false) {
  pb_in_ = InverseIndex(pb_out_);    
  auto id_op = GQTensor({pb_in_, pb_out_});
  for (long i = 0; i < pb_out_.dim; ++i) { id_op({i, i}) = 1; }
  id_op_ = id_op;
  for (long i = 0; i < N_+1; ++i) {
    auto ready_node = new FSMNode(i);
    ready_node->is_ready = true;
    ready_nodes_[i] = ready_node;
    auto final_node = new FSMNode(i);
    final_node->is_final = true;
    final_nodes_[i] = final_node;
  }
}

void MPOGenerator::AddTerm(
    const double coef,
    const std::vector<OpIdx> &opidxs,
    const GQTensor &inter_op) {
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


std::vector<GQTensor *> MPOGenerator::Gen(void) {
  if (!fsm_graph_merged_) {
    FSMGraphMerge();
    fsm_graph_merged_ = true;
  }
  // Print MPO tensors virtual bond dimension.
  for (auto &middle_nodes : middle_nodes_set_) {
    std::cout << std::setw(3) << middle_nodes.size() + 2 << std::endl;
  }
  std::vector<GQTensor *> mpo(N_);
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


void MPOGenerator::AddOneSiteTerm(const double coef, const OpIdx &opidx) {
  auto new_edge = new FSMEdge(
                          coef*opidx.op,
                          ready_nodes_[opidx.idx],
                          final_nodes_[opidx.idx+1],
                          opidx.idx);
  edges_set_[opidx.idx].push_back(new_edge);
}


void MPOGenerator::AddTwoSiteTerm(
    const double coef,
    const OpIdx &opidx1, const OpIdx &opidx2,
    const GQTensor &inter_op) {
  assert(opidx1.idx < opidx2.idx);
    GQTensor itrop;   // Inter operator.
    if (inter_op == kNullOperator) {
      itrop  = id_op_;
    } else {
      itrop = inter_op;
  }

  auto last_node = ready_nodes_[opidx1.idx];
  auto next_node = new FSMNode(opidx1.idx+1);
  next_node->mid_state_idx = middle_nodes_set_[opidx1.idx+1].size() + 1;
  auto new_edge = new FSMEdge(coef*opidx1.op, last_node, next_node, opidx1.idx);
  next_node->ledges.push_back(new_edge);
  edges_set_[opidx1.idx].push_back(new_edge);
  middle_nodes_set_[opidx1.idx+1].push_back(next_node);

  for (long i = opidx1.idx+1; i < opidx2.idx; ++i) {
    last_node = next_node;
    next_node = new FSMNode(i+1);
    next_node->mid_state_idx = middle_nodes_set_[i+1].size() + 1;
    new_edge = new FSMEdge(itrop, last_node, next_node, i);
    last_node->redges.push_back(new_edge);
    next_node->ledges.push_back(new_edge);
    edges_set_[i].push_back(new_edge);
    middle_nodes_set_[i+1].push_back(next_node);
  }

  last_node = next_node;
  next_node = final_nodes_[opidx2.idx+1];
  new_edge = new FSMEdge(opidx2.op, last_node, next_node, opidx2.idx);
  last_node->redges.push_back(new_edge);
  edges_set_[opidx2.idx].push_back(new_edge);
}


// Merge finite state machine graph.
void MPOGenerator::FSMGraphMerge(void) {
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


void MPOGenerator::FSMGraphMergeAt(const long nodes_set_idx) {
  std::vector<FSMNode *> &rmiddle_nodes = middle_nodes_set_[nodes_set_idx];
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


bool MPOGenerator::FSMGraphMergeNodesTo(
    const long target_mid_node_idx, std::vector<FSMNode *> &middle_nodes) {
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


bool MPOGenerator::FSMGraphMergeTwoNodes(FSMNode *&target, FSMNode *&from) {
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


inline bool MPOGenerator::CheckMergeableLeftEdgePair(
    const FSMEdge *ledge1, const FSMEdge *ledge2) {
  if (ledge1->op == ledge2->op && ledge1->last_node == ledge2->last_node) {
    return true;
  } else {
    return false;
  }
}


inline bool MPOGenerator::CheckMergeableRightEdgePair(
    const FSMEdge *redge1, const FSMEdge *redge2) {
  const FSMEdge *pnext_edge1 = redge1;
  const FSMEdge *pnext_edge2 = redge2;
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


void MPOGenerator::DeletePathToRightEnd(FSMEdge *edge) {
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


void MPOGenerator::RelabelMidNodesIdx(const long nodes_set_idx) {
  std::vector<FSMNode *> new_middle_nodes;
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


void MPOGenerator::RemoveNullEdges(void) {
  for (auto &edges : edges_set_) {
    std::vector<FSMEdge *> new_edges;
    for (auto &edge : edges) {
      if (edge != nullptr) { new_edges.push_back(edge); }
    }
    edges = new_edges;
  }
}


// Gnerator MPO tensors.
GQTensor *MPOGenerator::GenHeadMpo(void) {
  auto rvb_dim = middle_nodes_set_[1].size() + 2;
  auto rvb = Index({QNSector(QN(), rvb_dim)});
  auto mpo_ten = new GQTensor({pb_in_, rvb, pb_out_});
  // Add the R --> R identity edge.
  AddOpToHeadMpoTen(mpo_ten, id_op_, 0);
  // Working for other edges.
  for (auto &edge : edges_set_[0]) {
    if (edge->next_node->is_final) {
      AddOpToHeadMpoTen(mpo_ten, edge->op, rvb_dim-1);
    } else {
      AddOpToHeadMpoTen(mpo_ten, edge->op, edge->next_node->mid_state_idx);
    }
  }
  return mpo_ten;
}


GQTensor *MPOGenerator::GenTailMpo(void) {
  auto lvb_dim = middle_nodes_set_[N_-1].size() + 2;
  auto lvb = Index({QNSector(QN(), lvb_dim)});
  auto mpo_ten = new GQTensor({pb_in_, lvb, pb_out_});
  // Add the F --> F idensity edge.
  AddOpToTailMpoTen(mpo_ten, id_op_, lvb_dim-1);
  // Working for other edges.
  for (auto &edge : edges_set_[N_-1]) {
    if (edge->last_node->is_ready) {
      AddOpToTailMpoTen(mpo_ten, edge->op, 0);
    } else {
      AddOpToTailMpoTen(mpo_ten, edge->op, edge->last_node->mid_state_idx);
    }
  }
  return mpo_ten;
}


GQTensor *MPOGenerator::GenCentMpo(const long site_idx) {
  auto lvb_dim = middle_nodes_set_[site_idx].size() + 2;
  auto lvb = Index({QNSector(QN(), lvb_dim)});
  auto rvb_dim = middle_nodes_set_[site_idx+1].size() + 2;
  auto rvb = Index({QNSector(QN(), rvb_dim)});
  auto mpo_ten = new GQTensor({lvb, pb_in_, pb_out_, rvb});
  // Add the R --> R identity edge.
  AddOpToCentMpoTen(mpo_ten, id_op_, 0, 0);
  // Add the F --> F identity edge.
  AddOpToCentMpoTen(mpo_ten, id_op_, lvb_dim-1, rvb_dim-1);
  // Working for other edges.
  long ldim_idx, rdim_idx;
  for (auto &edge : edges_set_[site_idx]) {
    if (edge->last_node->is_ready) {
      ldim_idx = 0;
    } else {
      ldim_idx = edge->last_node->mid_state_idx;
    }
    if (edge->next_node->is_final) {
      rdim_idx = rvb_dim - 1;
    } else {
      rdim_idx = edge->next_node->mid_state_idx;
    }
    AddOpToCentMpoTen(mpo_ten, edge->op, ldim_idx, rdim_idx);
  }
  return mpo_ten;
}


void AddOpToHeadMpoTen(
    GQTensor *pmpo_ten, const GQTensor &rop, const long rvb_coor) {
  for (long bpb_coor = 0; bpb_coor < rop.indexes[0].dim; ++bpb_coor) {
    for (long tpb_coor = 0; tpb_coor < rop.indexes[1].dim; ++tpb_coor) {
      auto elem = rop.Elem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)({bpb_coor, rvb_coor, tpb_coor}) = elem;
      }
    }
  }
}


void AddOpToTailMpoTen(
    GQTensor *pmpo_ten, const GQTensor &rop, const long lvb_coor) {
  for (long bpb_coor = 0; bpb_coor < rop.indexes[0].dim; ++bpb_coor) {
    for (long tpb_coor = 0; tpb_coor < rop.indexes[1].dim; ++tpb_coor) {
      auto elem = rop.Elem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)({bpb_coor, lvb_coor, tpb_coor}) = elem;
      }
    }
  }
}


void AddOpToCentMpoTen(
    GQTensor *pmpo_ten, const GQTensor &rop,
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
