/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 14:37
* 
* Description: GraceQ/mps2 project. The main header file.
*/
#ifndef GQMPS2_GQMPS2_H
#define GQMPS2_GQMPS2_H


#include "gqten/gqten.h"

#include <string>
#include <vector>


namespace gqmps2 {
using namespace gqten;


const std::string kMpsPath = "mps";
const std::string kRuntimeTempPath = ".temp";
const std::string kBlockFileBaseName = "block";
const std::string kMpsTenBaseName = "mps_ten";


// MPO generator.
const GQTensor kNullOperator = GQTensor();


struct OpIdx {
  OpIdx(const GQTensor &op, const long idx) : op(op), idx(idx) {}

  GQTensor op;
  long idx;
};


struct FSMEdge {
  FSMEdge(const GQTensor &op, const long lstate, const long nstate) :
      op(op), lstate(lstate), nstate(nstate) {}
  FSMEdge(void) : FSMEdge(GQTensor(), 0, 0) {}

  GQTensor op;
  long lstate;
  long nstate;
};


class MPOGenerator {
public:
  MPOGenerator(const long, const Index &);

  void AddTerm(
      const double,
      const std::vector<OpIdx> &,
      const GQTensor &inter_op=kNullOperator);
  std::vector<GQTensor *> Gen(void);

private:
  long N_;
  Index pb_out_;
  Index pb_in_;
  std::vector<std::vector<FSMEdge>> edges_set_;
  std::vector<long> mid_state_nums_;
  GQTensor id_op_;
  
  void AddOneSiteTerm(const double, const OpIdx &);
  void AddTwoSiteTerm(
      const double, const OpIdx &, const OpIdx &, const GQTensor &);
  GQTensor *GenHeadMpo(const std::vector<FSMEdge> &, const long);
  GQTensor *GenCentMpo(const std::vector<FSMEdge> &, const long, const long);
  GQTensor *GenTailMpo(const std::vector<FSMEdge> &, const long);
};


// Lanczos Ground state search algorithm.
struct LanczosParams {
  LanczosParams(double err, long max_iter) :
      error(err), max_iterations(max_iter) {}
  LanczosParams(double err) : LanczosParams(err, 200) {}
  LanczosParams(void) : LanczosParams(1.0E-7, 200) {}
  LanczosParams(const LanczosParams &lancz_params) :
      LanczosParams(lancz_params.error, lancz_params.max_iterations) {}

  double error;
  long max_iterations;
};

struct LanczosRes {
  double gs_eng;
  GQTensor *gs_vec;
};

LanczosRes LanczosSolver(
    const std::vector<GQTensor *> &, GQTensor *,
    const LanczosParams &,
    const std::string &);


// Two sites update algorithm.
struct SweepParams {
  SweepParams(
      const long sweeps,
      const long dmin, const long dmax, const double cutoff, const bool fileio,
      const LanczosParams &lancz_params) :
      Sweeps(sweeps), Dmin(dmin), Dmax(dmax), Cutoff(cutoff), FileIO(fileio),
      LanczParams(lancz_params) {}
  SweepParams(
      const long sweeps,
      const long dmin, const long dmax, const double cutoff,
      const LanczosParams &lancz_params) :
      SweepParams(sweeps, dmin, dmax, cutoff, false, lancz_params) {}

  long Sweeps;
  long Dmin;
  long Dmax;
  double Cutoff;
  bool FileIO;
  LanczosParams LanczParams;
};

double TwoSiteAlgorithm(
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    const SweepParams &);

void DumpMps(const std::vector<GQTensor *> &);

void LoadMps(std::vector<GQTensor *> &);
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_GQMPS2_H */
