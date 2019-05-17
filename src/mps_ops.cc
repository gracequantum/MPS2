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
  auto rvb = GenHeadVirtBond(pb_out, div);
  auto lend_mps_ten = new GQTensor({pb_out, rvb});
  lend_mps_ten->Random(div);
  mps[0] = lend_mps_ten;
  auto N = mps.size();
  for (std::size_t i = 1; i < N; ++i) {
    auto lvb = InverseIndex(rvb); 
    rvb = GenBodyVirtBond(lvb, pb_out, zero_div);
    auto mps_teni = new GQTensor({lvb, pb_out, rvb});
    mps_teni->Random(zero_div);
    mps[i] = mps_teni;
  }
  auto lvb = InverseIndex(rvb);
  auto rend_mps_ten = new GQTensor({lvb, pb_out});
  rend_mps_ten->Random(zero_div);
  mps[N-1] = rend_mps_ten;
}


Index GenHeadVirtBond(const Index &pb, const QN &div) {
  std::vector<QNSector> new_qnscts;
  for (auto &qnsct : pb.qnscts) {
    new_qnscts.push_back(QNSector(div - qnsct.qn, 1));
  }
  return Index(new_qnscts, OUT);
}


Index GenBodyVirtBond(const Index &lvb, const Index &pb, const QN &div) {
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
} /* gqmps2 */ 
