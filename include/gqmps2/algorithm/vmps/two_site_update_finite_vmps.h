// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-08 16:45
* 
* Description: GraceQ/MPS2 project. Two-site update finite size vMPS.
*/

/**
@file two_site_update_finite_vmps.h
@brief Two-site update finite size vMPS.
*/
#ifndef GQMPS2_ALGORITHM_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_H
#define GQMPS2_ALGORITHM_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_H


#include "gqmps2/algorithm/lanczos_solver.h"    // LanczParams
#include "gqmps2/one_dim_tn/mpo.h"    // MPO
#include "gqmps2/one_dim_tn/mps/mps.h"    // MPS

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
    MPS<TenType> &,
    const MPO<TenType> &,
    const SweepParams &
);

template <typename TenType>
double TwoSiteAlgorithm(
    MPS<TenType> &,
    const std::vector<TenType *> &,
    const SweepParams &,
    std::vector<double>
);
} /* gqmps2 */ 


// Implementation details
#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps_impl.h"
//#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps_with_noise_impl.h"


#endif /* ifndef GQMPS2_ALGORITHM_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_H */
