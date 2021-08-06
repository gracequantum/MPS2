// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 14:37
*
* Description: GraceQ/MPS2 project. The main header file.
*/

/**
@file gqmps2.h
@brief The main header file.
*/
#ifndef GQMPS2_GQMPS2_H
#define GQMPS2_GQMPS2_H


#define GQMPS2_VERSION_MAJOR "0"
#define GQMPS2_VERSION_MINOR "2-alpha"
#define GQMPS2_VERSION_PATCH "1"
// GQMPS2_VERSION_DEVSTR to describe the development status, for example the git branch
#define GQMPS2_VERSION_DEVSTR


#include "gqmps2/case_params_parser.h"                              // CaseParamsParserBasic
#include "gqmps2/site_vec.h"                                        // SiteVec
// MPS class and its initializations and measurements
#include "gqmps2/one_dim_tn/mps_all.h"                              // MPS, ...
// MPO and its generator
#include "gqmps2/one_dim_tn/mpo/mpo.h"                              // MPO
#include "gqmps2/one_dim_tn/mpo/mpogen/mpogen.h"                    // MPOGenerator
// Algorithms
#include "gqmps2/algorithm/lanczos_solver.h"                        // LanczosParams
#include "gqmps2/algorithm/vmps/two_site_update_finite_vmps.h"      // TwoSiteFiniteVMPS, SweepParams
#include "gqmps2/algorithm/vmps/single_site_update_finite_vmps.h"   // SingleSiteFiniteVMPS


#endif /* ifndef GQMPS2_GQMPS2_H */
