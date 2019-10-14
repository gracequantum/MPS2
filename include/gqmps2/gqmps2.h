// SPDX-License-Identifier: LGPL-3.0-only
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

#include "third_party/nlohmann/json.hpp"


namespace gqmps2 {
using namespace gqten;
using json = nlohmann::json;


const std::string kCaseParamsJsonObjName = "CaseParams";

const std::string kMpsPath = "mps";
const std::string kRuntimeTempPath = ".temp";
const std::string kBlockFileBaseName = "block";
const std::string kMpsTenBaseName = "mps_ten";

const char kTwoSiteAlgoWorkflowInitial = 'i';
const char kTwoSiteAlgoWorkflowRestart = 'r';
const char kTwoSiteAlgoWorkflowContinue = 'c';

const int kLanczEnergyOutputPrecision = 16;

template <typename TenElemType>
const GQTensor<TenElemType> kNullOperator = GQTensor<TenElemType>();    // C++14


// Simulation case parameter parser basic class.
class CaseParamsParserBasic {
public:
  CaseParamsParserBasic(const char *file) {
    std::ifstream ifs(file);
    ifs >> raw_json_;
    ifs.close();
    if (raw_json_.find(kCaseParamsJsonObjName) != raw_json_.end()) {
      case_params = raw_json_[kCaseParamsJsonObjName];
    } else {
      std::cout << "CaseParams object not found, exit!" << std::endl;
      exit(1);
    }
  }

  int ParseInt(const std::string &item) {
    return case_params[item].get<int>();
  }

  double ParseDouble(const std::string &item) {
    return case_params[item].get<double>();
  }

  char ParseChar(const std::string &item) {
    auto char_str = case_params[item].get<std::string>();
    return char_str.at(0);
  }

  std::string ParseStr(const std::string &item) {
    return case_params[item].get<std::string>();
  }

  bool ParseBool(const std::string &item) {
    return case_params[item].get<bool>();
  }

  json case_params;

private:
  json raw_json_;
};


// MPO generator.
template <typename>
struct FSMEdge;


template <typename TenElemType>
struct FSMNode {
  FSMNode(const long loc) :
    is_ready(false), is_final(false), mid_state_idx(0), loc(loc) {}
  FSMNode(void) : FSMNode(-1) {}

  bool is_ready;
  bool is_final;
  long mid_state_idx;
  long loc;
  std::vector<FSMEdge<TenElemType> *> ledges;
  std::vector<FSMEdge<TenElemType> *> redges;
};


template <typename TenElemType>
struct FSMEdge {
  FSMEdge(
      const GQTensor<TenElemType> &op,
      FSMNode<TenElemType> *l_node, FSMNode<TenElemType> *n_node,
      const long loc) :
      op(op), last_node(l_node), next_node(n_node), loc(loc) {}
  FSMEdge(void) : FSMEdge(GQTensor<TenElemType>(), nullptr, nullptr, -1) {}

  const GQTensor<TenElemType> op;
  FSMNode<TenElemType> *last_node;
  FSMNode<TenElemType> *next_node;
  long loc;
};


template <typename TenElemType>
class MPOGenerator {
//[> TODO: Merge terms only with different coefficients. <]
public:
  MPOGenerator(const long, const Index &, const QN &);

  void AddTerm(
      const TenElemType,
      const std::vector<GQTensor<TenElemType>> &,
      const std::vector<long> &,
      const GQTensor<TenElemType> &inter_op=kNullOperator<TenElemType>);
  std::vector<GQTensor<TenElemType> *> Gen(void);

private:
  long N_;
  Index pb_out_;
  Index pb_in_;
  GQTensor<TenElemType> id_op_;
  std::vector<FSMNode<TenElemType> *> ready_nodes_;
  std::vector<FSMNode<TenElemType> *> final_nodes_;
  std::vector<std::vector<FSMNode<TenElemType> *>> middle_nodes_set_;
  std::vector<std::vector<FSMEdge<TenElemType> *>> edges_set_;
  // For nodes merge.
  bool fsm_graph_merged_;
  bool relable_to_end_;
  // For generation process.
  QN zero_div_;
  std::vector<Index> rvbs_;
  bool fsm_graph_sorted_;
  
  // Add terms.
  void AddOneSiteTerm(
      const TenElemType,
      const GQTensor<TenElemType> &,
      const long);
  void AddTwoSiteTerm(
      const TenElemType,
      const GQTensor<TenElemType> &, const GQTensor<TenElemType> &,
      const long, const long,
      const GQTensor<TenElemType> &);

  // Merge finite state machine graph.
  void FSMGraphMerge(void);
  void FSMGraphMergeAt(const long);   // At given middle nodes list.
  bool FSMGraphMergeNodesTo(const long, std::vector<FSMNode<TenElemType> *> &);
  bool FSMGraphMergeTwoNodes(FSMNode<TenElemType> *&, FSMNode<TenElemType> *&);

  bool CheckMergeableLeftEdgePair(
      const FSMEdge<TenElemType> *,
      const FSMEdge<TenElemType> *);
  bool CheckMergeableRightEdgePair(
      const FSMEdge<TenElemType> *,
      const FSMEdge<TenElemType> *);

  void DeletePathToRightEnd(FSMEdge<TenElemType> *);
  void RelabelMidNodesIdx(const long);
  void RemoveNullEdges(void);

  // Generation process.
  void FSMGraphSort(void);
  Index FSMGraphSortAt(const long);    // At given site.
  GQTensor<TenElemType> *GenHeadMpo(void);
  GQTensor<TenElemType> *GenCentMpo(const long);
  GQTensor<TenElemType> *GenTailMpo(void);
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

template <typename TenElemType>
struct LanczosRes {
  long iters;
  double gs_eng;
  GQTensor<TenElemType> *gs_vec;
};

template <typename TenElemType>
LanczosRes<TenElemType> LanczosSolver(
    const std::vector<GQTensor<TenElemType> *> &, GQTensor<TenElemType> *,
    const LanczosParams &,
    const std::string &);


// Two sites update algorithm.
struct SweepParams {
  SweepParams(
      const long sweeps,
      const long dmin, const long dmax, const double cutoff,
      const bool fileio,
      const char workflow,
      const LanczosParams &lancz_params) :
      Sweeps(sweeps), Dmin(dmin), Dmax(dmax), Cutoff(cutoff), FileIO(fileio),
      Workflow(workflow),
      LanczParams(lancz_params) {}

  long Sweeps;

  long Dmin;
  long Dmax;
  double Cutoff;

  bool FileIO;
  char Workflow;

  LanczosParams LanczParams;
};

template <typename TenType>
double TwoSiteAlgorithm(
    std::vector<TenType *> &,
    const std::vector<TenType *> &,
    const SweepParams &);


// MPS operations.
template <typename TenType>
void DumpMps(const std::vector<TenType *> &);

template <typename TenType>
void LoadMps(std::vector<TenType *> &);

template <typename TenType>
void RandomInitMps(
    std::vector<TenType> &,
    const Index &,
    const QN &,
    const QN &,
    const long);

template <typename TenType>
void DirectStateInitMps(
    std::vector<TenType *> &, const std::vector<long> &,
    const Index &, const QN &);

template <typename TenType>
void ExtendDirectRandomInitMps(
    std::vector<TenType *> &, const std::vector<std::vector<long>> &,
    const Index &, const QN &, const long);


// Observation measurements.
template <typename TenType>
struct MPS {
  MPS(std::vector<TenType *> &tens, const long center) :
      tens(tens), center(center), N(tens.size()) {}
  
  std::vector<TenType *> &tens; 
  long center;
  std::size_t N;
};

template <typename AvgType>
struct MeasuResElem {
  MeasuResElem(void) = default;
  MeasuResElem(const std::vector<long> &sites, const AvgType avg) :
    sites(sites), avg(avg) {}

  std::vector<long> sites;
  AvgType avg;
};

template <typename AvgType>
using MeasuRes = std::vector<MeasuResElem<AvgType>>;

template <typename AvgType>
using MeasuResSet = std::vector<MeasuRes<AvgType>>;


// Single site operator.
template <typename TenElemType>
MeasuRes<TenElemType> MeasureOneSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const GQTensor<TenElemType> &, const std::string &);

template <typename TenElemType>
MeasuResSet<TenElemType> MeasureOneSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const std::vector<std::string> &);

template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const GQTensor<TenElemType> &,
    const GQTensor<TenElemType> &,
    const std::vector<std::vector<long>> &,
    const std::string &);

template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<std::vector<GQTensor<TenElemType>>> &,
    const std::vector<std::vector<GQTensor<TenElemType>>> &,
    const GQTensor<TenElemType> &,
    const std::vector<std::vector<long>> &,
    const std::string &);


// System I/O functions.
template <typename TenType>
inline void WriteGQTensorTOFile(const TenType &t, const std::string &file) {
  std::ofstream ofs(file, std::ofstream::binary);
  bfwrite(ofs, t);
  ofs.close();
}


template <typename TenType>
inline void ReadGQTensorFromFile(TenType * &rpt, const std::string &file) {
  std::ifstream ifs(file, std::ifstream::binary);
  rpt = new TenType();
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
} /* gqmps2 */ 


// Implementation details
#include "gqmps2/detail/lanczos_impl.h"
#include "gqmps2/detail/mpogen_impl.h"
#include "gqmps2/detail/two_site_algo_impl.h"
#include "gqmps2/detail/mps_ops_impl.h"
#include "gqmps2/detail/mps_measu_impl.h"


#endif /* ifndef GQMPS2_GQMPS2_H */
