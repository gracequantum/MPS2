// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-08 14:22
*
* Description: GraceQ/MPS2 project. Lanczos solver.
*/

/**
@file lanczos_solver.h
@brief A Lanczos solver for the effective Hamiltonian in MPS-MPO based algorithms.
*/
#ifndef GQMPS2_ALGORITHM_LANCZOS_SOLVER_H
#define GQMPS2_ALGORITHM_LANCZOS_SOLVER_H


#include <stdlib.h>     // size_t


namespace gqmps2 {


/**
Parameters used by the Lanczos solver.
*/
struct LanczosParams {
  /**
  Setup Lanczos solver parameters.

  @param error The Lanczos tolerated error.
  @param max_iterations The maximal iteration times.
  */
  LanczosParams(double err, size_t max_iter) :
      error(err), max_iterations(max_iter) {}
  LanczosParams(double err) : LanczosParams(err, 200) {}
  LanczosParams(void) : LanczosParams(1.0E-7, 200) {}
  LanczosParams(const LanczosParams &lancz_params) :
      LanczosParams(lancz_params.error, lancz_params.max_iterations) {}

  double error;             ///< The Lanczos tolerated error.
  size_t max_iterations;    ///< The maximal iteration times.
};
} /* gqmps2 */


// Implementation details
#include "gqmps2/algorithm/lanczos_solver_impl.h"


#endif /* ifndef GQMPS2_ALGORITHM_LANCZOS_SOLVER_H */
