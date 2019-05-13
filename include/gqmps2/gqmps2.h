/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 14:37
* 
* Description: GraceQ/mps2 project. The main header file.
*/
#ifndef GQMPS2_GQMPS2_H
#define GQMPS2_GQMPS2_H


#include "gqten/gqten.h"

#include <string>
#include <vector>


namespace gqmps2 {
using namespace gqten;


// Lanczos Ground state search algorithm.
struct LanczosParams {
  LanczosParams(double err, long max_iter) :
      error(err), max_iterations(max_iter) {}
  LanczosParams(double err) : LanczosParams(err, 200) {}
  LanczosParams(void) : LanczosParams(1.0E-7, 200) {}
  double error;
  long max_iterations;
};

struct LanczosRes {
  double gs_eng;
  GQTensor *gs_vec;
};

LanczosRes LanczosSolver(
    const std::vector<GQTensor *> &, GQTensor *,
    const LanczosParams &,
    const std::string &);



} /* gqmps2 */ 
#endif /* ifndef GQMPS2_GQMPS2_H */
