/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-16 21:15
* 
* Description: GraceQ/mps2 project. MPS operations.
*/
#include "mps_ops.h"
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"


namespace  gqmps2 {
using  namespace gqten;


void DumpMps(const std::vector<GQTensor *> &mps) {
  if (!IsPathExist(kMpsPath)) { CreatPath(kMpsPath); }
  auto N = mps.size();
  std::string file;
  for (std::size_t i = 0; i < N; ++i) {
    file = kMpsPath + "/" +
           kMpsTenBaseName + std::to_string(i) + "." + kGQTenFileSuffix;
    std::ofstream ofs(file, std::ofstream::binary);
    bfwrite(ofs, *mps[i]);
    ofs.close();
  }
}


void LoadMps(std::vector<GQTensor *> &mps) {
  auto N = mps.size();
  std::string file;
  for (std::size_t i = 0; i < N; ++i) {
    file = kMpsPath + "/" +
           kMpsTenBaseName + std::to_string(i) + "." + kGQTenFileSuffix;
    std::ifstream ifs(file, std::ifstream::binary);
    mps[i] = new GQTensor();
    bfread(ifs, *mps[i]);
    ifs.close();
  }
}


void RandomInitMps(
    std::vector<GQTensor *> &mps,
    const Index &pb_out,
    const QN &div, const QN &zero_div) {
  Index lvb, rvb;
  // Left to center.
  rvb = GenHeadRightVirtBond(pb_out, div);
  mps[0] = new GQTensor({pb_out, rvb});
  auto N = mps.size();
  for (std::size_t i = 1; i < N/2; ++i) {
    lvb = InverseIndex(rvb); 
    rvb = GenBodyRightVirtBond(lvb, pb_out, zero_div);
    mps[i] = new GQTensor({lvb, pb_out, rvb});
  }
  auto cent_bond = rvb;

  // Right to center.
  lvb = GenTailLeftVirtBond(pb_out, zero_div);
  mps[N-1] = new GQTensor({lvb, pb_out});
  for (std::size_t i = N-2; i > N/2; --i) {
    rvb = InverseIndex(lvb);
    lvb = GenBodyLeftVirtBond(rvb, pb_out, zero_div);
    mps[i] = new GQTensor({lvb, pb_out, rvb});
  }
  rvb = InverseIndex(lvb);
  lvb = InverseIndex(cent_bond);
  mps[N/2] = new GQTensor({lvb, pb_out, rvb});
  
  // Initialize elements randomly.
  mps[0]->Random(div);
  for (std::size_t i = 1; i < N; ++i) {
    mps[i]->Random(zero_div);
  }
}


Index GenHeadRightVirtBond(const Index &pb, const QN &div) {
  std::vector<QNSector> new_qnscts;
  for (auto &qnsct : pb.qnscts) {
    new_qnscts.push_back(QNSector(div - qnsct.qn, 1));
  }
  return Index(new_qnscts, OUT);
}


Index GenTailLeftVirtBond(const Index &pb, const QN &zero_div) {
  std::vector<QNSector> new_qnscts;
  for (auto &qnsct : pb.qnscts) {
    new_qnscts.push_back(QNSector(qnsct.qn - zero_div, 1));
  }
  return Index(new_qnscts, IN);
}


Index GenBodyRightVirtBond(const Index &lvb, const Index &pb, const QN &div) {
  std::vector<QNSector> new_qnscts;
  for (auto &lvqnsct : lvb.qnscts) {
    for (auto &pqnsct : pb.qnscts) {
      auto poss_rvb_qn = div + lvqnsct.qn - pqnsct.qn;
      auto has_qn = false;
      for (auto &new_qnsct : new_qnscts) {
        if (poss_rvb_qn == new_qnsct.qn) {
          has_qn = true;
          break;
        }
      }
      if (!has_qn) {
        new_qnscts.push_back(QNSector(poss_rvb_qn, 1));
      }
    }
  }
  return Index(new_qnscts, OUT);
}


Index GenBodyLeftVirtBond(
    const Index &rvb, const Index &pb, const QN &zero_div) {
  std::vector<QNSector> new_qnscts;
  for (auto &rvqnsct : rvb.qnscts) {
    for (auto &pqnsct : pb.qnscts) {
      auto poss_lvb_qn = pqnsct.qn - zero_div + rvqnsct.qn;
      auto has_qn = false;
      for (auto &new_qnsct : new_qnscts) {
        if (poss_lvb_qn == new_qnsct.qn) {
          has_qn = true;
          break;
        }
      }
      if (!has_qn) {
        new_qnscts.push_back(QNSector(poss_lvb_qn, 1));
      }
    }
  }
  return Index(new_qnscts, IN);
}
} /* gqmps2 */ 
