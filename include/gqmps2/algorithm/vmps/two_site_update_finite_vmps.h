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


#include "gqmps2/consts.h"                      // kMpsPath, kRuntimeTempPath
#include "gqmps2/algorithm/lanczos_solver.h"    // LanczParams

#include <string>     // string


namespace gqmps2 {


struct SweepParams {
  SweepParams(
      const size_t sweeps,
      const size_t dmin, const size_t dmax, const double trunc_err,
      const LanczosParams &lancz_params,
      const std::string mps_path = kMpsPath,
      const std::string temp_path = kRuntimeTempPath
  ) :
      sweeps(sweeps),
      Dmin(dmin), Dmax(dmax), trunc_err(trunc_err),
      lancz_params(lancz_params),
      mps_path(mps_path),
      temp_path(temp_path) {}

  size_t sweeps;

  size_t Dmin;
  size_t Dmax;
  double trunc_err;

  LanczosParams lancz_params;

  // Advanced parameters
  /// MPS directory path
  std::string mps_path;

  /// Runtime temporary files directory path
  std::string temp_path;
};
} /* gqmps2 */


// Implementation details
#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps_impl.h"


#endif /* ifndef GQMPS2_ALGORITHM_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_H */
