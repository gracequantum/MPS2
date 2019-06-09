/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-16 21:15
* 
* Description: GraceQ/mps2 project. MPS operations.
*/
#include "mps_ops.h"
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <cmath>
#include <algorithm>
#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>

namespace  gqmps2 {
using  namespace gqten;


// MPS I/O.
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


// MPS initialization.
void RandomInitMps(
    std::vector<GQTensor *> &mps,
    const Index &pb,
    const QN &tot_div,
    const QN &zero_div,
    const long dmax) {
  Index lvb, rvb;

  // Left to center.
  rvb = GenHeadRightVirtBond(pb, tot_div, dmax);
  mps[0] = new GQTensor({pb, rvb});
  mps[0]->Random(tot_div);
  assert(Div(*mps[0]) == tot_div);
  auto N = mps.size();
  for (std::size_t i = 1; i < N/2; ++i) {
    lvb = InverseIndex(rvb);
    rvb = GenBodyRightVirtBond(lvb, pb, zero_div, dmax);
    mps[i] = new GQTensor({lvb, pb, rvb});
    mps[i]->Random(zero_div);
    assert(Div(*mps[i]) == zero_div);
  }
  auto cent_bond = rvb;

  // Right to center.
  lvb = GenTailLeftVirtBond(pb, zero_div, dmax);
  mps[N-1] = new GQTensor({lvb, pb});
  mps[N-1]->Random(zero_div);
  assert(Div(*mps[N-1]) == zero_div);
  for (std::size_t i = N-2; i > N/2; --i) {
    rvb = InverseIndex(lvb);
    lvb = GenBodyLeftVirtBond(rvb, pb, zero_div, dmax);
    mps[i] = new GQTensor({lvb, pb, rvb});
    mps[i]->Random(zero_div);
    assert(Div(*mps[i]) == zero_div);
  }

  rvb = InverseIndex(lvb);
  lvb = InverseIndex(cent_bond);
  mps[N/2] = new GQTensor({lvb, pb, rvb});
  mps[N/2]->Random(zero_div);
  assert(Div(*mps[N/2]) == zero_div);
}


Index GenHeadRightVirtBond(
    const Index &pb, const QN &tot_div, const long dmax) {
  std::vector<QNSector> new_qnscts;
  for (auto &qnsct : pb.qnscts) {
    auto new_qn = tot_div - qnsct.qn;
    auto has_qn = false;
    for (auto &new_qnsct : new_qnscts) {
      if (new_qnsct.qn == new_qn) {
        new_qnsct.dim += qnsct.dim;
        has_qn = true;
        break;
      }
    }
    if (!has_qn) {
      new_qnscts.push_back(QNSector(new_qn, qnsct.dim));
    }
  }
  DimCut(new_qnscts, dmax, pb.dim);
  return Index(new_qnscts, OUT);
}


Index GenBodyRightVirtBond(
    const Index &lvb, const Index &pb, const QN &zero_div, const long dmax) {
  std::vector<QNSector> new_qnscts;
  for (auto &lvqnsct : lvb.qnscts) {
    for (auto &pqnsct : pb.qnscts) {
      auto poss_rvb_qn = zero_div + lvqnsct.qn - pqnsct.qn;
      auto has_qn = false;
      for (auto &new_qnsct : new_qnscts) {
        if (poss_rvb_qn == new_qnsct.qn) {
          new_qnsct.dim += lvqnsct.dim;
          has_qn = true;
          break;
        }
      }
      if (!has_qn) {
        new_qnscts.push_back(QNSector(poss_rvb_qn, lvqnsct.dim));
      }
    }
  }
  DimCut(new_qnscts, dmax, pb.dim);
  return Index(new_qnscts, OUT);
}


Index GenTailLeftVirtBond(
    const Index &pb, const QN &zero_div, const long dmax) {
  std::vector<QNSector> new_qnscts;
  for (auto &qnsct : pb.qnscts) {
    auto new_qn = qnsct.qn - zero_div;
    auto has_qn = false;
    for (auto &new_qnsct : new_qnscts) {
      if (new_qnsct.qn == new_qn) {
        new_qnsct.dim += qnsct.dim;
        has_qn = true;
        break;
      }
    }
    if (!has_qn) {
      new_qnscts.push_back(QNSector(new_qn, qnsct.dim));
    }
  }
  DimCut(new_qnscts, dmax, pb.dim);
  return Index(new_qnscts, IN);
}


Index GenBodyLeftVirtBond(
    const Index &rvb, const Index &pb, const QN &zero_div, const long dmax) {
  std::vector<QNSector> new_qnscts;
  for (auto &rvqnsct : rvb.qnscts) {
    for (auto &pqnsct : pb.qnscts) {
      auto poss_lvb_qn = pqnsct.qn - zero_div + rvqnsct.qn;
      auto has_qn = false;
      for (auto &new_qnsct : new_qnscts) {
        if (poss_lvb_qn == new_qnsct.qn) {
          new_qnsct.dim += rvqnsct.dim;
          has_qn = true;
          break;
        }
      }
      if (!has_qn) {
        new_qnscts.push_back(QNSector(poss_lvb_qn, rvqnsct.dim));
      }
    }
  }
  DimCut(new_qnscts, dmax, pb.dim);
  return Index(new_qnscts, IN);
}


void DimCut(std::vector<QNSector> &qnscts, const long dmax, const long pdim) {
  std::sort(qnscts.begin(), qnscts.end(), GreaterQNSectorDim);
  auto kept_qn_cnt = 0;
  auto dim = 0;
  for (auto &qnsct : qnscts) {
    if (dim + qnsct.dim < dmax) {
      dim += qnsct.dim;
      kept_qn_cnt++;
    } else if (dim + qnsct.dim == dmax) {
      dim += qnsct.dim;
      kept_qn_cnt++;
      break;
    } else {
      qnsct.dim -= (dim + qnsct.dim - dmax);
      kept_qn_cnt++;
      break;
    }
  }
  qnscts.resize(kept_qn_cnt);
}


void DirectStateInitMps(
    std::vector<GQTensor *> &mps, const std::vector<long> &stat_labs,
    const Index &pb_out, const QN &zero_div) {
  auto N = mps.size();
  assert(N == stat_labs.size());
  Index lvb, rvb;

  // Calculate total quantum number.
  auto div = pb_out.CoorOffsetAndQnsct(stat_labs[0]).qnsct.qn;
  for (std::size_t i = 1; i < N; ++i) {
    div += pb_out.CoorOffsetAndQnsct(stat_labs[i]).qnsct.qn;
  }

  auto stat_lab = stat_labs[0];
  auto rvb_qn = div - pb_out.CoorOffsetAndQnsct(stat_lab).qnsct.qn;
  rvb = Index({QNSector(rvb_qn, 1)}, OUT);
  mps[0] = new GQTensor({pb_out, rvb});
  (*mps[0])({stat_lab, 0}) = 1;

  for (std::size_t i = 1; i < N-1; ++i) {
    lvb = InverseIndex(rvb); 
    stat_lab = stat_labs[i];
    rvb_qn = zero_div - 
             pb_out.CoorOffsetAndQnsct(stat_lab).qnsct.qn +
             lvb.CoorOffsetAndQnsct(0).qnsct.qn;
    rvb = Index({QNSector(rvb_qn, 1)}, OUT);
    mps[i] = new GQTensor({lvb, pb_out, rvb});
    (*mps[i])({0, stat_lab, 0}) = 1;
  }

  lvb = InverseIndex(rvb);
  mps[N-1] = new GQTensor({lvb, pb_out});
  stat_lab = stat_labs[N-1];
  (*mps[N-1])({0, stat_lab}) = 1;
}


void ExtendDirectRandomInitMps(
    std::vector<GQTensor *> &mps,
    const std::vector<std::vector<long>> &stat_labs_set,
    const Index &pb, const QN &zero_div, const long enlarged_dim) {
  auto fusion_stats_num = stat_labs_set.size();
  assert(fusion_stats_num >= 1);
  auto N = mps.size();
  assert(N == stat_labs_set[0].size());
  Index lvb, rvb;
  std::vector<QNSector> rvb_qnscts;

  // Calculate total quantum number.
  auto div = pb.CoorOffsetAndQnsct(stat_labs_set[0][0]).qnsct.qn;
  for (std::size_t i = 1; i < N; ++i) {
    div += pb.CoorOffsetAndQnsct(stat_labs_set[0][i]).qnsct.qn;
  }

  // Deal with MPS head local tensor.
  for (std::size_t i = 0; i < fusion_stats_num; ++i) {
    auto stat_lab = stat_labs_set[i][0];
    auto rvb_qn = div - pb.CoorOffsetAndQnsct(stat_lab).qnsct.qn;
    rvb_qnscts.push_back(QNSector(rvb_qn, enlarged_dim));
  }
  rvb = Index(rvb_qnscts, OUT);
  rvb_qnscts.clear();
  mps[0] = new GQTensor({pb, rvb});
  mps[0]->Random(div);

  // Deal with MPS middle local tensors.
  for (std::size_t i = 1; i < N-1; ++i) {
    lvb = InverseIndex(rvb);
    for (std::size_t j = 0; j < fusion_stats_num; ++j) {
      auto stat_lab = stat_labs_set[j][i];
      auto rvb_qn = zero_div -
                    pb.CoorOffsetAndQnsct(stat_lab).qnsct.qn +
                    lvb.CoorOffsetAndQnsct(j*enlarged_dim).qnsct.qn;
      rvb_qnscts.push_back(QNSector(rvb_qn, enlarged_dim));
    }
    rvb = Index(rvb_qnscts, OUT);
    mps[i] = new GQTensor({lvb, pb, rvb});
    rvb_qnscts.clear();
    mps[i]->Random(zero_div);
  }

  // Deal with MPS tail local tensor.
  lvb = InverseIndex(rvb);
  mps[N-1] = new GQTensor({lvb, pb});
  mps[N-1]->Random(zero_div);

  // Centralize MPS.
  auto temp_mps = MPS(mps, -1);
  RightNormalizeMps(temp_mps, temp_mps.N-1, 1);
}



// MPS centralization.
void CentralizeMps(MPS &mps, const long target_center) {
  auto origin_center = mps.center;
  if (target_center > origin_center) {
    LeftNormalizeMps(mps, origin_center, target_center-1);
    mps.center = target_center;
  } else if (target_center < origin_center) {
    RightNormalizeMps(mps, origin_center, target_center+1);
    mps.center = target_center;
  }
}


void LeftNormalizeMps(MPS &mps, const long from, const long to) {
  assert(to >= from);
  for (long i = from; i <= to; ++i) {
    LeftNormalizeMpsTen(mps, i);
  }
}


void RightNormalizeMps(MPS &mps, const long from, const long to) {
  assert(to <= from);
  for (long i = from; i >= to; --i) {
    RightNormalizeMpsTen(mps, i);
  }
}


void LeftNormalizeMpsTen(MPS &mps, const long site) {
  assert(site < mps.N-1);
  long ldims, rdims;
  if (site == 0) {
    ldims = 1;
    rdims = 1;
  } else {
    ldims = 2;
    rdims = 1;
  }
  auto svd_res = Svd(
      *mps.tens[site],
      ldims, rdims,
      Div(*mps.tens[site]), Div(*mps.tens[site+1]));
  delete mps.tens[site];  
  mps.tens[site] = svd_res.u;
  auto temp_ten = Contract(*svd_res.s, *svd_res.v, {{1}, {0}});
  delete svd_res.s;
  delete svd_res.v;
  auto next_ten = Contract(*temp_ten, *mps.tens[site+1], {{1}, {0}});
  delete temp_ten;
  delete mps.tens[site+1];
  mps.tens[site+1] = next_ten;
}


void RightNormalizeMpsTen(MPS &mps, const long site) {
  assert(site > 0);
  long ldims, rdims;
  if (site == mps.N-1) {
    ldims = 1;
    rdims = 1;
  } else {
    ldims = 1;
    rdims = 2;
  }
  auto svd_res = Svd(
      *mps.tens[site],
      ldims, rdims,
      Div(*mps.tens[site-1]), Div(*mps.tens[site]));
  delete mps.tens[site];
  mps.tens[site] = svd_res.v;
  auto temp_ten = Contract(*svd_res.u, *svd_res.s, {{1}, {0}});
  delete svd_res.u;
  delete svd_res.s;
  std::vector<long> ta_ctrct_axes;
  if ((site-1) == 0) {
    ta_ctrct_axes = {1};
  } else {
    ta_ctrct_axes = {2};
  }
  auto prev_ten = Contract(*mps.tens[site-1], *temp_ten, {ta_ctrct_axes, {0}});
  delete temp_ten;
  delete mps.tens[site-1];
  mps.tens[site-1] = prev_ten;
}
} /* gqmps2 */ 
