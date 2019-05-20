/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-20 12:25
* 
* Description: GraceQ/mps2 project. Timer.
*/
#ifndef GQMPS2_TIMER_H
#define GQMPS2_TIMER_H

#include "gqmps2/gqmps2.h"

#include <iostream>
#include <string>

#include <time.h>
#include <sys/time.h>

namespace gqmps2 {


Timer::Timer(const std::string &notes) : notes_(notes), start_(0) {}


void Timer::Restart(void) { start_ = GetWallTime(); }


double Timer::Elapsed(void) { return GetWallTime() - start_; }


void Timer::PrintElapsed(void) {
  auto elapsed_time = Elapsed(); 
  std::cout << "[timing] " << notes_ << "\t" << elapsed_time << std::endl;
}


double Timer::GetWallTime(void) {
  struct timeval time;
  if (gettimeofday(&time, NULL)) { return 0; }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_TIMER_H */
