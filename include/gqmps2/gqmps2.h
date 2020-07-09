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
#include "gqmps2/detail/mpogen/fsm.h"
#include "gqmps2/detail/mpogen/coef_op_alg.h"

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
template <typename TenElemType>
class MPOGenerator {
public:
  MPOGenerator(const long, const Index &, const QN &);
  /** MPOGenerator Generator for non-uniform local hilbert space
    Input: - vector<Index>& pb_out_vector: the sets collecting the indices of all sites
           - const QN& zero_div: The leftmost index of MPO
   */
  MPOGenerator(const std::vector<Index> &, const QN& );

  using TenElemVec = std::vector<TenElemType>;
  using GQTensorT = GQTensor<TenElemType>;
  using GQTensorVec = std::vector<GQTensorT>;
  using PGQTensorVec = std::vector<GQTensorT *>;

  void AddTerm(
      const TenElemType,
      const GQTensorVec &,
      const std::vector<long> &,
      const GQTensorVec &);

  void AddTerm(
    const TenElemType coef,
    GQTensorVec phys_ops,
    std::vector<long> idxs,
    const GQTensorVec &inst_ops,
    const std::vector<long> &inst_idxs);

  void AddTerm(
      const TenElemType,
      const GQTensorVec &,
      const std::vector<long> &,
      const GQTensorT &inst_op=kNullOperator<TenElemType>);

  void AddTerm(
      const TenElemType,
      const GQTensorT &,
      const long);

  FSM GetFSM(void) { return fsm_; }

  PGQTensorVec Gen(void);

private:
  long N_;
  std::vector<Index> pb_in_vector_;
  std::vector<Index> pb_out_vector_;
  QN zero_div_;
  std::vector<GQTensorT> id_op_vector_;
  FSM fsm_;
  LabelConvertor<TenElemType> coef_label_convertor_;
  LabelConvertor<GQTensorT> op_label_convertor_;

  GQTensorT GenIdOpTen_(const Index &);

  std::vector<size_t> SortSparOpReprMatColsByQN_(
      SparOpReprMat &, Index &, const GQTensorVec &);

  QN CalcTgtRvbQN_(
    const size_t, const size_t, const OpRepr &,
    const GQTensorVec &, const Index &);

  GQTensorT *HeadMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const Index &,
      const TenElemVec &, const GQTensorVec &);

  GQTensorT *TailMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const Index &,
      const TenElemVec &, const GQTensorVec &);

  GQTensorT *CentMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const Index &,
      const Index &,
      const TenElemVec &,
      const GQTensorVec &, const long);
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


template <typename TenType>
double TwoSiteAlgorithm( //Add by wanghx in June 21, 2020.
  std::vector<TenType *> &,
  const std::vector<TenType *> &,
  const SweepParams &,
  std::vector<double>);// Add a parameter noise

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
void DirectStateInitMps(
        std::vector<TenType *> &, const std::vector<long> &,
        const std::vector<Index> &, const QN &);

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
MeasuRes<TenElemType> MeasureOneSiteOp(//Add by wanghx June 29
  MPS<GQTensor<TenElemType>> &,
  const GQTensor<TenElemType> &,
  const std::vector<long> &site_set,//For nonuniform hilbert space we must
    const std::string &);//specify which sites are be measured

template <typename TenElemType>
MeasuResSet<TenElemType> MeasureOneSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &, //physical operator
    const std::vector<std::string> &);

template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &, //physical operator
    const GQTensor<TenElemType> &, //insertion opeartor
    const std::vector<std::vector<long>> &, // physical operator sites
    const std::string &);

template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
  MPS<GQTensor<TenElemType>> &mps,
  const std::vector<GQTensor<TenElemType>> &phys_ops, //physical operator
  const std::vector<std::vector<long>> &sites_set, //physical operator sites
  const std::string &res_file_basename); //< no insertion operator

template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
  MPS<GQTensor<TenElemType>> & mps,
  const std::vector<GQTensor<TenElemType>> & op_set,
  const GQTensor<TenElemType> & insertop,
  const GQTensor<TenElemType> & id,
  const std::vector<std::vector<long>> &site_set,
  const std::string & filename){
  return MeasureTwoSiteOp(mps, op_set,insertop,site_set,filename);
}

template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<std::vector<GQTensor<TenElemType>>> &,
    const std::vector<std::vector<GQTensor<TenElemType>>> &,
    const std::vector<std::vector<long>> &,
    const std::string &);

template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(
  MPS<GQTensor<TenElemType>> & mps,
  const std::vector<std::vector<GQTensor<TenElemType>>> & phy_op,
  const std::vector<std::vector<GQTensor<TenElemType>>> & ins_op,
  const GQTensor<TenElemType> & id,
  const std::vector<std::vector<long>> & site_set,
  const std::string &filename){
  return MeasureMultiSiteOp(mps, phy_op,ins_op, site_set,filename);
}

template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(
  MPS<GQTensor<TenElemType>> &mps,
  const std::vector<std::vector<GQTensor<TenElemType>>> &phys_ops_set,
  const std::vector<std::vector<GQTensor<TenElemType>>> &inst_ops_set,
  const std::vector<std::vector<long>> &sites_set,
  const std::vector<std::vector<long>> &insertsites_set,
  const std::string &res_file_basename);

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
#include "gqmps2/detail/mpogen/mpogen_impl.h"
#include "gqmps2/detail/two_site_algo_impl.h"
#include "gqmps2/detail/mps_ops_impl.h"
#include "gqmps2/detail/mps_measu_impl.h"


#endif /* ifndef GQMPS2_GQMPS2_H */
