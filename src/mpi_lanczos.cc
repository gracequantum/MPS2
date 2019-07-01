/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-07-01 10:48
* 
* Description: GraceQ/mps2 project. Private objects for MPI parallel Lanczos solver,
*              implementation.
*/
#include "mpi_lanczos.h"
#include "gqten/gqten.h"

#include "mpi.h"


namespace gqmps2 {
using namespace gqten;


GQTensor *gqmps2_mpi_eff_ham_mul_state_cent(
    const std::vector<GQTensor *> &eff_ham, GQTensor *state,
    MPI_Comm comm, const int workers) {
  auto res = GQTEN_MPI_Contract(*eff_ham[0], *state, {{0}, {0}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[1], {{0, 2}, {0, 1}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[2], {{4, 1}, {0, 1}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[3], {{4, 1}, {1, 0}}, comm, workers);
  return res;
}


GQTensor *gqmps2_mpi_eff_ham_mul_state_lend(
    const std::vector<GQTensor *> &eff_ham, GQTensor *state,
    MPI_Comm comm, const int workers) {
  auto res = GQTEN_MPI_Contract(*state, *eff_ham[1], {{0}, {0}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[2], {{0, 2}, {1, 0}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[3], {{0, 3}, {0, 1}}, comm, workers);
  return res;
}


GQTensor *gqmps2_mpi_eff_ham_mul_state_rend(
    const std::vector<GQTensor *> &eff_ham, GQTensor *state,
    MPI_Comm comm, const int workers) {
  auto res = GQTEN_MPI_Contract(*state, *eff_ham[0], {{0}, {0}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[1], {{2, 0}, {0, 1}}, comm, workers);
  GQMPS2_MPI_InplaceContract(res, *eff_ham[2], {{3, 0}, {1, 0}}, comm, workers);
  return res;
}
} /* gqmps2 */ 
