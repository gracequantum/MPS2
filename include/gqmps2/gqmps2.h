// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 14:37
* 
* Description: GraceQ/MPS2 project. The main header file.
*/
#ifndef GQMPS2_GQMPS2_H
#define GQMPS2_GQMPS2_H


#include "gqmps2/case_params_parser.h"          // CaseParamsParserBasic
#include "gqmps2/site_vec.h"                    // SiteVec
#include "gqmps2/mpogen/mpogen.h"               // MPOGenerator
#include "gqmps2/algorithm/lanczos_solver.h"    // LanczosParams
#include "gqten/gqten.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <sys/stat.h>


namespace gqmps2 {
using namespace gqten;


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
double TwoSiteAlgorithm(
  std::vector<TenType *> &,
  const std::vector<TenType *> &,
  const SweepParams &,
  std::vector<double>);

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


/// Single site operator. Uniform indices version
template <typename TenElemType>
MeasuRes<TenElemType> MeasureOneSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const GQTensor<TenElemType> &, const std::string &);

/// For non-uniform indices
template <typename TenElemType>
MeasuRes<TenElemType> MeasureOneSiteOp(
  MPS<GQTensor<TenElemType>> &,
  const GQTensor<TenElemType> &,
  const std::vector<long> &site_set, //specify which site are be measured
  const std::string &);

template <typename TenElemType>
MeasuResSet<TenElemType> MeasureOneSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &,
    const std::vector<std::string> &);

template <typename TenElemType>
MeasuRes<TenElemType> MeasureOneSiteOp(
  MPS<GQTensor<TenElemType>> &,
  const GQTensor<TenElemType> &,
  const std::vector<long> &site_set,//For nonuniform hilbert space we must
  const std::string &);//specify which sites are be measured

  /// The insertion operators are inserted in all the sites between physical opeartors
  /// usually for uniform indices
template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<GQTensor<TenElemType>> &, //physical operator
    const GQTensor<TenElemType> &, //insertion opeartor
    const std::vector<std::vector<long>> &, // physical operator sites
    const std::string &);

/// No insertion operators, can be used for uniform or non-uniform indices
template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
  MPS<GQTensor<TenElemType>> &mps,
  const std::vector<GQTensor<TenElemType>> &phys_ops, //physical operator
  const std::vector<std::vector<long>> &sites_set, //physical operator sites
  const std::string &res_file_basename);

  /// For compatibility of old version, where we should input an indentity operator.
  /// For uniform indices
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

/// Specify which sites are be inserted, can be used for non-uniform indices
template <typename TenElemType>
MeasuRes<TenElemType> MeasureTwoSiteOp(
  MPS<GQTensor<TenElemType>> &mps,
  const std::vector<GQTensor<TenElemType>> &phys_ops,
  const std::vector<std::vector<long>> &sites_set,
  const std::vector<std::vector<long>> &insertsite_set,
  const std::string &res_file_basename);

/// The insertion operators are inserted in all the sites between physical opeartors
/// usually for uniform indices
template <typename TenElemType>
MeasuRes<TenElemType> MeasureMultiSiteOp(
    MPS<GQTensor<TenElemType>> &,
    const std::vector<std::vector<GQTensor<TenElemType>>> &,
    const std::vector<std::vector<GQTensor<TenElemType>>> &,
    const std::vector<std::vector<long>> &,
    const std::string &);


/// For compatibility of old version, where we should input an indentity operator.
/// For uniform indices
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


/// Specify which sites are be inserted, can be used for non-uniform indices
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
#include "gqmps2/two_site_algo_impl.h"
#include "gqmps2/two_site_algo_impl_with_noise.h"
#include "gqmps2/mps_ops_impl.h"
#include "gqmps2/mps_measu_impl.h"


#endif /* ifndef GQMPS2_GQMPS2_H */
