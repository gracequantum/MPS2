/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-13 15:13
* 
* Description: GraceQ/mps2 project. Private objects used by MPO generation, implementation.
*/
#include "mpogen.h"
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>


namespace gqmps2 {
using namespace gqten;


void MPOGenerator::AddOneSiteTerm(const double coef, const OpIdx &opidx) {
  edges_set_[opidx.idx].push_back(FSMEdge(coef*opidx.op, 0, -1));
}


void MPOGenerator::AddTwoSiteTerm(
    const double coef,
    const OpIdx &opidx1, const OpIdx &opidx2,
    const GQTensor &inter_op) {
  assert(opidx1.idx < opidx2.idx);
  //if (opidx2.idx-opidx1.idx == 1) {
    //// Nearest neighbor interaction term.
    //auto target_op1 = coef * opidx1.op;
    //auto target_op2 = opidx2.op;
    //long next_state1, last_state2;
    //auto has_edge1 = false;
    //auto has_edge2 = false;
    //for (auto &edge : edges_set_[opidx1.idx]) {
      //if (edge.lstate == 0 && edge.op == target_op1 && edge.nstate != -1) {
        //next_state1 = edge.nstate;
        //has_edge1 = true;
        //break;
      //}
    //}
    //if (!has_edge1) {
      //next_state1 = mid_state_nums_[opidx1.idx+1] + 1;
    //}
    //for (auto &edge : edges_set_[opidx2.idx]) {
      //if (edge.lstate != 0 && edge.op == target_op2 && edge.nstate == -1) {
        //last_state2 = edge.lstate;
        //has_edge2 = true;
        //break;
      //}
    //}
    //if (!has_edge2) {
      //last_state2 = mid_state_nums_[opidx2.idx] + 1;
    //}
    //if (has_edge1 && has_edge2) {
      //if (next_state1 != last_state2) {
        //mid_state_nums_[opidx1.idx+1] += 1;
        //next_state1 = mid_state_nums_[opidx1.idx+1];
        //last_state2 = mid_state_nums_[opidx1.idx+1];
        //edges_set_[opidx1.idx].push_back(
            //FSMEdge(target_op1, 0, next_state1));
        //edges_set_[opidx2.idx].push_back(
            //FSMEdge(target_op2, last_state2, -1));
      //} else {
        //std::cout << "Warning: term duplication, ignored!" << std::endl;
      //}
    //} else if (has_edge1 && !has_edge2) {
      //last_state2 = next_state1;
      //edges_set_[opidx2.idx].push_back(
          //FSMEdge(target_op2, last_state2, -1));
    //} else if (!has_edge1 && has_edge2) {
      //next_state1 = last_state2;
      //edges_set_[opidx1.idx].push_back(
          //FSMEdge(target_op1, 0, next_state1));
    //} else {
      //mid_state_nums_[opidx1.idx+1] += 1;
      //next_state1 = mid_state_nums_[opidx1.idx+1];
      //last_state2 = mid_state_nums_[opidx1.idx+1];
      //edges_set_[opidx1.idx].push_back(
          //FSMEdge(target_op1, 0, next_state1));
      //edges_set_[opidx2.idx].push_back(
          //FSMEdge(target_op2, last_state2, -1));
    //}
  //} else {
    // No nearest neighbor interaction term.
    GQTensor itrop;   // Inter operator.
    if (inter_op == kNullOperator) {
      itrop  = id_op_;
    } else {
      itrop = inter_op;
    }
    // Trace from left to right to find the begin site.
    long beg_lstate = 0;
    long beg_idx = opidx1.idx;
    long last_state = 0;
    long next_state;
    GQTensor target_op;
    for (long i = opidx1.idx; i < opidx2.idx-1; ++i) {
      auto has_edge = false;
      if (i == opidx1.idx) {
        target_op = coef * opidx1.op;
        for (auto &edge : edges_set_[i]) {
          if (edge.lstate == last_state &&
              edge.op == target_op &&
              edge.nstate != -1) {
            next_state = edge.nstate;
            has_edge = true;
            break;
          }
        }
      } else if (i == opidx2.idx) {
        target_op = opidx2.op;
        for (auto &edge : edges_set_[i]) {
          if (edge.lstate == last_state &&
              edge.op == target_op &&
              edge.nstate == -1) {
            next_state = edge.nstate;
            has_edge = true;
            break;
          }
        }
      } else {
        target_op = itrop;
        for (auto &edge : edges_set_[i]) {
          if (edge.lstate == last_state &&
              edge.op == target_op &&
              edge.nstate != -1) {
            next_state = edge.nstate;
            has_edge = true;
            break;
          }
        }
      }
      if (!has_edge) {
        beg_lstate = last_state;
        beg_idx = i;
        break;
      } else {
        last_state = next_state;
      }
    }
    // Trace from right to left to find the end site.
    long end_nstate = -1;
    long end_idx = opidx2.idx;
    next_state = -1;
    for (long i = opidx2.idx; i > opidx1.idx+1; --i) {
      auto has_edge = false;
      if (i == opidx2.idx) {
        target_op = opidx2.op;
        for (auto &edge : edges_set_[i]) {
          if (edge.nstate == next_state &&
              edge.op == target_op &&
              edge.lstate != 0) {
            last_state = edge.lstate;
            has_edge = true;
            break;
          }
        }
      } else if (i == opidx1.idx) {
        target_op = coef * opidx1.op;
        for (auto &edge : edges_set_[i]) {
          if (edge.nstate == next_state &&
              edge.op == target_op &&
              edge.lstate == 0) {
            last_state = edge.lstate;
            has_edge = true;
            break;
          } 
        }
      } else {
        target_op = itrop;
        for (auto &edge : edges_set_[i]) {
          if (edge.nstate == next_state &&
              edge.op == target_op &&
              edge.lstate != 0) {
            last_state = edge.lstate;
            has_edge = true;
            break;
          }
        }
      }
      if (!has_edge) {
        end_nstate = next_state;
        end_idx = i;
        break;
      } else {
        next_state = last_state;
      }
    }
    
    // Generate mid states.
    for (long i = beg_idx; i < end_idx+1; ++i) {
      if (i == opidx1.idx) {
        target_op = coef * opidx1.op;
      } else if (i == opidx2.idx) {
        target_op = opidx2.op;
      } else {
        target_op = itrop;
      }
      if (i == beg_idx) {
        last_state = beg_lstate;
      } else {
        last_state = next_state;
      }
      if (i == end_idx) {
        next_state = end_nstate;
      } else {
        mid_state_nums_[i+1] += 1;
        next_state = mid_state_nums_[i+1];
      }
      edges_set_[i].push_back(
          FSMEdge(target_op, last_state, next_state));
    }
  //}
}


GQTensor *MPOGenerator::GenHeadMpo(
    const std::vector<FSMEdge> &links, const long r_mid_state_num) {
  auto rvb_dim = r_mid_state_num + 2;
  auto rvb = Index({QNSector(QN(), rvb_dim)});
  auto mpo_ten = new GQTensor({pb_in_, rvb, pb_out_});
  // Add the R --> R identity edge.
  AddOpToHeadMpoTen(mpo_ten, id_op_, 0);
  // Working for other links.
  for (auto &edge : links) {
    if (edge.nstate == -1) {
      AddOpToHeadMpoTen(mpo_ten, edge.op, rvb_dim-1);
    } else {
      AddOpToHeadMpoTen(mpo_ten, edge.op, edge.nstate);
    }
  }
  return mpo_ten;
}


GQTensor *MPOGenerator::GenTailMpo(
    const std::vector<FSMEdge> &links, const long l_mid_state_num) {
  auto lvb_dim = l_mid_state_num + 2;
  auto lvb = Index({QNSector(QN(), lvb_dim)});
  auto mpo_ten = new GQTensor({pb_in_, lvb, pb_out_});
  // Add the F --> F idensity edge.
  AddOpToTailMpoTen(mpo_ten, id_op_, lvb_dim-1);
  // Working for other links.
  for (auto &edge : links) {
    AddOpToTailMpoTen(mpo_ten, edge.op, edge.lstate);
  }
  return mpo_ten;
}


GQTensor *MPOGenerator::GenCentMpo(
    const std::vector<FSMEdge> &links,
    const long l_mid_state_num,
    const long r_mid_state_num) {
  auto lvb_dim = l_mid_state_num + 2;
  auto lvb = Index({QNSector(QN(), lvb_dim)});
  auto rvb_dim = r_mid_state_num + 2;
  auto rvb = Index({QNSector(QN(), rvb_dim)});
  auto mpo_ten = new GQTensor({lvb, pb_in_, pb_out_, rvb});
  // Add the R --> R identity edge.
  AddOpToCentMpoTen(mpo_ten, id_op_, 0, 0);
  // Add the F --> F identity edge.
  AddOpToCentMpoTen(mpo_ten, id_op_, lvb_dim-1, rvb_dim-1);
  for (auto &edge : links) {
    if (edge.nstate == -1) {
      AddOpToCentMpoTen(mpo_ten, edge.op, edge.lstate, rvb_dim-1);
    } else {
      AddOpToCentMpoTen(mpo_ten, edge.op, edge.lstate, edge.nstate);
    }
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
