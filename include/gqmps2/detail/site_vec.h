// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-29 14:29
* 
* Description: GraceQ/MPS2 project. One-dimensional vector representation of the local Hilbert spaces of the system.
*/

/**
@file site_vec.h
@brief One-dimensional vector representation of the local Hilbert spaces of the
       system.
*/
#ifndef GQMPS2_DETAIL_SITE_VEC_H
#define GQMPS2_DETAIL_SITE_VEC_H


#include "gqten/gqten.h"  // IndexVec

#include <assert.h>       // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqmps2 {
using namespace gqten;


// Helpers
inline
Index SetIndexDirOut(const Index &idx) {
  if (idx.dir == OUT) {
    return idx;
  } else {
    return InverseIndex(idx);
  }
}


/**
Vector of the local Hilbert spaces of the system.

@since version 0.2.0
*/
class SiteVec {
public:
  /**
  Create a system with N identical sites.

  @param N Total number of sites (> 0) of the system.
  @param local_hilbert_space Local (on-site) Hilbert space represented by a
         Index.

  @since version 0.2.0
  */
  SiteVec(
      const int N,
      const Index &local_hilbert_space
  ) {
    assert(N > 0);
    size = N;
    sites = IndexVec(N, SetIndexDirOut(local_hilbert_space));
  }

  /**
  Create a sites vector using a vector of arbitrary Indexes.

  @param local_hilbert_spaces Local (on-site) Hilbert spaces on the each site.

  @since version 0.2.0
  */
  SiteVec(
      const IndexVec &local_hilbert_spaces
  ) {
    size = local_hilbert_spaces.size();
    assert(size > 0);
    sites.reserve(size);
    for (int i = 0; i < size; ++i) {
      sites.emplace_back(SetIndexDirOut(local_hilbert_spaces[i]));
    }
  }

  int size;        ///< The size of the SiteVec, i.e. the size of the system.
  IndexVec sites;     ///< Local Hilbert spaces represented by a vector of Index with OUT direction.
};
} /* gqmps2 */ 
#endif /* ifndef GQMPS2_DETAIL_SITE_VEC_H */
