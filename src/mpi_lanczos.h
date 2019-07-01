/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-07-01 10:24
* 
* Description: GraceQ/mps2 project. Private objects for MPI parallel Lanczos solver.
*/
#ifndef GQMPS2_MPI_LANCZOS_H
#define GQMPS2_MPI_LANCZOS_H


#include "lanczos.h"
#include "gqten/gqten.h"

#include "mpi.h"


namespace gqmps2 {
using namespace gqten;

GQTensor *gqmps2_mpi_eff_ham_mul_state_cent(
    const std::vector<GQTensor *> &, GQTensor *,
    MPI_Comm, const int);

GQTensor *gqmps2_mpi_eff_ham_mul_state_lend(
    const std::vector<GQTensor *> &, GQTensor *,
    MPI_Comm, const int);

GQTensor *gqmps2_mpi_eff_ham_mul_state_rend(
    const std::vector<GQTensor *> &, GQTensor *,
    MPI_Comm, const int);


// Helpers.
inline void GQMPS2_MPI_InplaceContract(
    GQTensor * &lhs, const GQTensor &rhs,
    const std::vector<std::vector<long>> &axes,
    MPI_Comm comm, const int workers) {
  auto res = GQTEN_MPI_Contract(*lhs, rhs, axes, comm, workers);
  delete lhs;
  lhs = res;
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_MPI_LANCZOS_H */
