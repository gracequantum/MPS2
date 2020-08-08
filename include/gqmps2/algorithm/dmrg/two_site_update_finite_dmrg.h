// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-08 16:45
* 
* Description: GraceQ/MPS2 project. Two-site update finite size DMRG.
*/

/**
@file two_site_update_finite_dmrg.h
@brief Two-site update finite size DMRG.
*/
#ifndef GQMPS2_ALGORITHM_DMRG_TWO_SITE_UPDATE_FINITE_DMRG_H
#define GQMPS2_ALGORITHM_DMRG_TWO_SITE_UPDATE_FINITE_DMRG_H


#include "gqmps2/algorithm/lanczos_solver.h"    // LanczParams

#include <vector>     // vector


namespace gqmps2 {


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
    const SweepParams &
);

template <typename TenType>
double TwoSiteAlgorithm(
    std::vector<TenType *> &,
    const std::vector<TenType *> &,
    const SweepParams &,
    std::vector<double>
);
} /* gqmps2 */ 


// Implementation details
#include "gqmps2/algorithm/dmrg/two_site_update_finite_dmrg_impl.h"
#include "gqmps2/algorithm/dmrg/two_site_update_finite_dmrg_with_noise_impl.h"


#endif /* ifndef GQMPS2_ALGORITHM_DMRG_TWO_SITE_UPDATE_FINITE_DMRG_H */
