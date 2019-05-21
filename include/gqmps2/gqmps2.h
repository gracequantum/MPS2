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
#include <iostream>
#include <fstream>

#include <sys/stat.h>


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
  long iters;
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
      const bool restart,
      const LanczosParams &lancz_params) :
      Sweeps(sweeps), Dmin(dmin), Dmax(dmax), Cutoff(cutoff), FileIO(fileio),
      Restart(restart),
      LanczParams(lancz_params) {}
  SweepParams(
      const long sweeps,
      const long dmin, const long dmax, const double cutoff,
      const LanczosParams &lancz_params) :
      SweepParams(sweeps, dmin, dmax, cutoff, false, false, lancz_params) {}

  long Sweeps;
  long Dmin;
  long Dmax;
  double Cutoff;
  bool FileIO;
  bool Restart;
  LanczosParams LanczParams;
};

double TwoSiteAlgorithm(
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    const SweepParams &);


// MPS operations.
void DumpMps(const std::vector<GQTensor *> &);

void LoadMps(std::vector<GQTensor *> &);

void RandomInitMps(
    std::vector<GQTensor *> &, const Index &, const QN &, const QN &);

void DirectStateInitMps(
    std::vector<GQTensor *> &, const std::vector<long> &,
    const Index &, const QN &, const QN &);


// System I/O functions.
inline void WriteGQTensorTOFile(const GQTensor &t, const std::string &file) {
  std::ofstream ofs(file, std::ofstream::binary);  
  bfwrite(ofs, t);
  ofs.close();
}


inline void ReadGQTensorFromFile(GQTensor * &rpt, const std::string &file) {
  std::ifstream ifs(file, std::ifstream::binary);
  rpt = new GQTensor();
  bfread(ifs, *rpt);
  ifs.close();
}


inline bool IsPathExist(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}


inline void CreatPath(const std::string &path) {
  const int dir_err = mkdir(
                          path.c_str(),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  if (dir_err == -1) {
    std::cout << "error creating directory!" << std::endl;
    exit(1);
  }
}


// Timer.
class Timer {
public:
  Timer(const std::string &);

  void Restart(void);
  double Elapsed(void);
  void PrintElapsed(void);

private:
  double start_;
  std::string notes_;

  double GetWallTime(void);
};
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_GQMPS2_H */
