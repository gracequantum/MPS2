// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
*         Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-7-9
*
* Description: GraceQ/MPS2 project. Single-site vMPS algorithm header.
*/

/**
@file single_site_update_finite_vmps.h
@brief Single-site update finite size vMPS.
*/
#ifndef GQMPS2_ALGORITHM_VMPS_SINGLE_SITE_UPDATE_FINITE_VMPS_H
#define GQMPS2_ALGORITHM_VMPS_SINGLE_SITE_UPDATE_FINITE_VMPS_H


#include "gqmps2/consts.h"                      // kMpsPath, kRuntimeTempPath
#include "gqmps2/algorithm/lanczos_solver.h"    // LanczParams

#include <string>     // string


namespace gqmps2 {
const double kSingleVMPSMaxNoise = 1.0; //maximal noise
const double kSingleVMPSNoiseIncrease = 1.02;
const double kSingleVMPSNoiseDecrease = 0.95;
const double kSingleVMPSAlpha = 0.3;

struct SingleVMPSSweepParams {
  SingleVMPSSweepParams(
      const size_t sweeps,
      const size_t dmin, const size_t dmax, const double trunc_err,
      const LanczosParams &lancz_params,
      const std::vector<double> noises = std::vector<double>(1, 0.0),
      const double max_noise = kSingleVMPSMaxNoise,
      const double noise_increase = kSingleVMPSNoiseIncrease,
      const double noise_decrease = kSingleVMPSNoiseDecrease,
      const double alpha = kSingleVMPSAlpha,
      const std::string mps_path = kMpsPath,
      const std::string temp_path = kRuntimeTempPath
  ) :
      sweeps(sweeps),
      Dmin(dmin), Dmax(dmax), trunc_err(trunc_err),
      lancz_params(lancz_params),
      noises(noises),
      max_noise(max_noise),
      noise_increase(noise_increase),
      noise_decrease(noise_decrease),
      alpha(alpha),
      mps_path(mps_path),
      temp_path(temp_path) {}

  size_t sweeps;

  size_t Dmin;
  size_t Dmax;
  double trunc_err;

  LanczosParams lancz_params;


  /// Noise magnitude each sweep
  std::vector<double> noises;
  double max_noise;
  double noise_increase;
  double noise_decrease;
  double alpha;

  // Advanced parameters
  /// MPS directory path
  std::string mps_path;

  /// Runtime temporary files directory path
  std::string temp_path;

};
} /* gqmps2 */


// Implementation details
#include "gqmps2/algorithm/vmps/single_site_update_finite_vmps_impl.h"


#endif /* ifndef GQMPS2_ALGORITHM_VMPS_SINGLE_SITE_UPDATE_FINITE_VMPS_H */
