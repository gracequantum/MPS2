/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-14 12:15
* 
* Description: GraceQ/mps2 project. Private objects for two sites algorithm.
*/
#ifndef GQMPS2_TWO_SITE_ALGO_H
#define GQMPS2_TWO_SITE_ALGO_H


#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

namespace gqmps2 {
using namespace gqten;


std::pair<std::vector<GQTensor *>, std::vector<GQTensor *>> InitBlocks(
    const std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    const SweepParams &);

double TwoSiteSweep(
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    std::vector<GQTensor *> &, std::vector<GQTensor *> &,
    const SweepParams &);

double TwoSiteUpdate(
    const long,
    std::vector<GQTensor *> &, const std::vector<GQTensor *> &,
    std::vector<GQTensor *> &, std::vector<GQTensor *> &,
    const SweepParams &, const char);


// System I/O functions.
inline void WriteGQTensorTOFile(const GQTensor &t, const std::string &file) {
  std::ofstream ofs(file, std::ofstream::binary);  
  bfwrite(ofs, t);
  ofs.close();
}


inline void ReadGQTensorFromFile(GQTensor * &rpt, const std::string &file) {
  std::ifstream ifs(file, std::ifstream::binary);
  rpt = new GQTensor();
  bfread(ifs, *rpt);
  ifs.close();
}


inline bool IsPathExist(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}


inline void CreatPath(const std::string &path) {
  const int dir_err = mkdir(
                          path.c_str(),
                          S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  if (dir_err == -1) {
    std::cout << "error creating directory!" << std::endl;
    exit(1);
  }
}
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_TWO_SITE_ALGO_H */
