/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-07-01 09:54
* 
* Description: GraceQ/mps2 project. Private objects for MPI parallel two sites algorithm.
*/
#ifndef GQMPS2_MPI_TWO_SITE_ALGO_H
#define GQMPS2_MPI_TWO_SITE_ALGO_H


#include "two_site_algo.h"
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include "mpi.h"


namespace gqmps2 {
using namespace gqten;


double GQMPS2_MPI_TwoSiteSweep(
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    std::vector<GQTensor *> &, std::vector<GQTensor *> &,
    const SweepParams &,
    MPI_Comm, const int);

double GQMPS2_MPI_TwoSiteUpdate(
    const long,
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    std::vector<GQTensor *> &, std::vector<GQTensor *> &,
    const SweepParams &, const char,
    MPI_Comm, const int);
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_MPI_TWO_SITE_ALGO_H */
