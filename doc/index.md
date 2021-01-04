# GraceQ/MPS2
_Easily push your bond dimension to 10k_


## Features
- Memory efficiency and speed for large-scale MPS manipulations and optimizations.
- Exact matrix product operator (MPO) generation for a Hamiltonian with any n-body term.
- Header-only library.


## Design Goals
- High-performance MPS algorithms implementation based on the power of [GraceQ/tensor](https://github.com/gracequantum/tensor).
- Performing MPS related algorithms on kinds of HPC hardware architectures.
- Flexible API design to cope with complex research tasks. We do not offer something like t-J model, but you can easily define it.


## Newest version
- version [0.2-alpha.0](https://github.com/gracequantum/MPS2/releases/latest)


## Current developers and maintainers
- Rong-Yang Sun <sun-rongyang@outlook.com>

> Note: For a complete list of the contributors, see CONTRIBUTORS.txt


## Development homepage
- On GitHub: [gracequantum/MPS2](https://github.com/gracequantum/MPS2)


## User guide
> In the following code blocks, if it is available, you can click the class or function name to jump to the detail documentation.

### Dependence
To use GraceQ/MPS2, you should install [GraceQ/tensor](https://github.com/gracequantum/tensor) framework first. The latest GraceQ/MPS2 always depends on the latest GraceQ/tensor.

### Download GraceQ/MPS2
You can always get the latest usable version using `git` from the default branch on [GitHub](https://github.com/gracequantum/MPS2).
```
git clone https://github.com/gracequantum/MPS2.git gqmps2
```
And you can also download the release version from [release page](https://github.com/gracequantum/MPS2/releases) and uncompress the file to get the source code.

### Installation
From now on, we suppose the root directory of the GraceQ/MPS2 source code is `gqmps2`.

#### Using CMake
You can use [CMake](https://cmake.org/) 3.12 or higher version to install it.
```
cd gqmps2
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<your_gqmps2_installation_root>
make install
```
Where `<your_gqmps2_installation_root>` is the installation root directory which you have write access. For example, `~/.local` is a good choice.

#### Install manually
GraceQ/MPS2 is a header-only library. In theory, you can just copy these header files to anywhere you like as the installation procedure.
```
cd gqmps2
cp -r include/gqmps2 <your_gqmps2_installation_root>/include/
```

### Using GraceQ/MPS2
It is easy to use GraceQ/MPS2.
```cpp
#include "gqmps2/gqmps2.h"

// Because GraceQ/MPS2 highly depends on the GraceQ/tensor, you may also need to include it
#include "gqten/gqten.h"
```

When you compile your program, you can use following compile flags to gain the best performance.
```
-std=c++14 -g -O3 -DNDEBUG
```

GraceQ/MPS2 needs hptt (needed by GraceQ/tensor) and MKL during linking process. So you should use the following flags when you link the library.
```
-lhptt <your_mkl_linking_flags>
```

We highly recommend that you use [MKL Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/) to set `<your_mkl_linking_flags>`. A possible complete compiling command may looks like
```
g++ \
  -std=c++14 -g -O3 -DNDEBUG \
  -I<your_gqmps2_installation_root>/include \
  -I<your_gqten_installation_root>/include -L<your_gqten_installation_root>/lib \
  -lhptt <your_mkl_linking_flags> \
  -o <your_main_program_name> <your_main_program_file_name>
```

In the following sections, we will introduce basic features of GraceQ/MPS2.


### Abstract a quantum system using gqmps2::SiteVec
In quantum matter physics, a system we interest usually lives on a set of sites. Each site can be described by a local Hilbert space. The total Hilbert space of a quantum many-body system is the direct product of these local Hilbert spaces. Specially, to suit the one-dimensional tensor networks, like matrix product state (MPS), matrix product operator (MPO), we always need to specify an one-dimensional path to connect these local Hilbert spaces. So, in the language of one-dimensional tensor network, a quantum system can be abstracted a vector of local Hilbert spaces. In GraceQ/MPS2, this vector is the gqmps2::SiteVec.

For system with identical local Hilbert spaces, we can define the corresponding site vector using the total number of sites and the local Hilbert space, which can be described by a `gqten::Index` with `OUT` direction.
```cpp
// Create a system with 10 spin-1/2 local sites
using TenElemT = gqten::GQTEN_Double;
using QNT = gqten::QN<gqten::U1QNVal>;
using QNSctT = gqten::QNSector<QNT>;
using IndexT = gqten::Index<QNT>;
auto qnup = QNT({gqten::QNCard("Sz", gqten::U1QNVal( 1))});
auto qndn = QNT({gqten::QNCard("Sz", gqten::U1QNVal(-1))});
auto pb_out = IndexT(
                  {QNSctT(qnup, 1), QNSctT(qndn, 1)},
                  gqten::GQTenIndexDirType::OUT
              );
size_t N = 10;
gqmps2::SiteVec<TenElemT, QNT> len10_spin_onehalf_sites(N, pb_out);
```

For system is constituted by a set of different local Hilbert spaces, we can define the corresponding site vector using a vector of local Hilbert space.
```cpp
auto pb_out1 = IndexT(...);
auto pb_out2 = IndexT(...);
gqmps2::SiteVec<TenElemT, QNT> nonuniform_sites({pb_out1, pb_out2, pb_out1, pb_out2, ...});
```

After define a site vector, you can access its size, each of local Hilbert space, and identity operators for each site.
```cpp
// Get the size of the site vector
sites.size;

// Get local Hilbert spaces described by a vector of gqten::Index
sites.sites;

// Get identity operators for each site which is described by a vector of gqten::GQTensor
sites.id_ops;
```

### Matrix product state (MPS)
#### Generic MPS: gqmps2::MPS
GraceQ/MPS2 offers generic MPS class, gqmps2::MPS, which can be seen as a fix size one-dimensional container initialized by a gqmps2::SiteVec.
```cpp
gqmps2::SiteVec<TenElemT, QNT> sites(...);
gqmps2::MPS<TenElemT, QNT> mps(sites);

// Get the size of the MPS
mps.size();

// Get sites information
auto sites_info = mps.GetSitesInfo();
sites_info == sites;    // Will be true

// Access to MPS local tensor
mps[idx] = mps_ten;
auto ten = mps[idx];

// Access to pointer of MPS local tensor
mps(idx) = pmps_ten;    // Once feeding to pointer to MPS, the corresponding memory will be managed by MPS class
auto pten = mps(idx);

// Access to first and last local tensors
mps.front();
mps.back();

// Check whether the container is empty
mps.empty();

// Allocate memory of a specific local tensor
mps.alloc(idx);
// Deallocate memory of a specific local tensor
mps.dealloc(idx);
// Deallocate all local tensors
mps.clear();
mps.empty() == true;    // Will be true

// Load tensor from a file to a specific place
mps.LoadTen(idx, "gqten_file_path");
// Dump local tensor to a file and keep it in memory
mps.DumpTen(idx, "gqten_file_paht");
// Dump local tensor to a file and release the memory
mps.DumpTen(idx, "gqten_file_paht", true);

// Load the whole MPS from a MPS folder
mps.Load("mps_folder_path");
// Dump the whole MPS to a folder and keep contents in memory
mps.Dump("mps_folder_path");
// Dump the whole MPS to a folder and release the memory
mps.Dump("mps_folder_path", true);
mps.empty() == true;    // Will be true
```
#### MPS for finite size system: gqmps2::FiniteMPS
As you can see, the gqmps2::MPS is very loose. It does not even check whether the tensors managed is rank-3. For a MPS of a finite size system, GraceQ/MPS2 offers a much more rigorous class, gqmps2::FiniteMPS. Except above features in gqmps2::MPS, gqmps2::FiniteMPS obtains more functions which are unique for a finite size MPS.

Each local tensor in gqmps2::FiniteMPS has a property to trace its canonical type which is defined by gqmps2::MPSTenCanoType. When you use `finite_mps[]` or `finite_mps()` to access local tensor with modify permission, gqmps2::FiniteMPS will set its canonical type to gqmps2::MPSTenCanoType::NONE. gqmps2::FiniteMPS also trace the canonical center of the MPS and offers centralization function.
```cpp
gqmps2::FiniteMPS finite_mps(sites);

finite_mps[0] = local_ten0;
finite_mps[1] = local_ten1;
...

auto ten_cano_type = finite_mps.GetTenCanoType(idx);
ten_cano_type == gqmps2::MPSTenCanoType::NONE;    // Will be true
finite_mps.GetCenter() == gqmps2::kUncentralizedCenterIdx;    // Will be true

// Centralize a MPS
finite_mps.Centralize(1);     // Centralize the MPS to 1th site
finite_mps.GetCenter() == 1;    // Will be true
finite_mps.GetTenCanoType(0) == gqmps2::MPSTenCanoType::LEFT;     // Will be true
finite_mps.GetTenCanoType(2) == gqmps2::MPSTenCanoType::RIGHT;     // Will be true
finite_mps.GetTenCanoType(1) == gqmps2::MPSTenCanoType::NONE;     // Will be true
```

The convention of the order and the direction of indexes of MPS local tensors in gqmps2::FiniteMPS is
```
0                1                1                       1
^                ^                ^                       ^
|                |                |                       |
A-->--1    0-->--A-->--2    0-->--A-->--2    ...    0-->--A
```

##### Initialize gqmps2::FiniteMPS as a direct product state
In practice, we may need to initialize a MPS as a direct product state to continue future algorithm step. GraceQ/MPS2 offers gqmp2::DirectStateInitMps to do this task. In most cases, symmetry is kept during algorithm steps, so the quantum number of the initial MPS also labels the sector you are working in the whole Hilbert space. Notice that the MPS is centralized to site 0.
```cpp
gqmps2::DirectStateInitMps(
    mps,                                            // The to-be initialized MPS
    {stat_lab_on_site0, stat_lab_on_site1, ..},     // The label of state on each of the local Hilbert space
    related_zero_div_qn                             // The corresponding quantum number which describes zero divergence
);

// Initialize MPS as a spin-1/2 AFM Ising state
FiniteMPST mps(sites);
std::vector<size_t> stat_labs;
for (size_t i = 0; i < params.N; ++i) { stat_labs.push_back(i % 2); }
gqmps2::DirectStateInitMps(mps, stat_labs, qn0);
```

### Generating Hamiltonian's MPO by gqmps2::MPOGenerator
You can use gqmps2::MPOGenerator to generate Hamiltonian's exact MPO contains any one-site, two-site and multi-site terms. First, you initialize a MPO generator using gqmps2::SiteVec of the system and related zero divergence quantum number.
```cpp
MPOGenerator<TenElemT, QNT> mpo_gen(sites, related_zero_div_qn);
```

Then you can generator a n-body term using `AddTerm()`.
```cpp
mpo_gen.AddTerm(
    coef,                           // Coefficient of the term
    {local_op1, local_op2, ...},    // All the local (on-site) operators in the term
    {site_idx1, site_idx2, ...},    // The site indexes of these local operators. Notice that the indexes of the operators have to be ascending sorted
);

// Add a three spins interaction term
mpo_gen.AddTerm(CoefJ, {sz, sp, sm}, {i, j, k});
```

You can also add a many-body term defined by physical operators and insertion operators. This API is very convenient to deal with Fermion system where non-local Jordan-Wigner string exists. Notice that the indexes of the operators have to be ascending sorted.
```cpp
mpo_gen.AddTerm(
    coef,                           // The coefficient of the term
    {phys_op1, phys_op2, ...},      // Operators with physical meaning in this term. Its size must be larger than 1
    {phys_idx1, phys_idx2, ...},    // The corresponding site indexes of the physical operators
    {inst_op1, inst_op2, ...},      // Operators which will be inserted between physical operators and also behind the last physical operator as a tail string. For example the Jordan-Wigner string operator
    {
        {inst_op1_idx1, inst_op1_idx2, ...},
        {inst_op2_idx1, inst_op2_idx2, ...},
        ...
    }     // Each element defines the explicit site indexes of the corresponding insertion operator. If it is left to empty (default value), every site between the corresponding physical operators will be inserted a same insertion operator
);

// Add a hopping term in t-J type model with J-W string
mpo_gen.AddTerm(-t, {adagup, aup}, {i, j}, {f});    // f is the J-W string operator
```

If you want to generate two-body term you can use a simplified API.
```cpp
mpo_gen.AddTerm(
    coef,                               // The coefficient of the term
    op1, op1_idx,                       // The first physical operator and its site index
    op2, op2_idx,                       // The second physical operator and its site index
    inst_op,                            // The insertion operator for the two-body term. If it is left to empty (default value), identity operator will be inserted
    {inst_op_idx1, inst_op_idx2, ...}   // The explicit site indexes of the insertion operator. If it is left to empty (default value), every sites between two physical operators will be inserted the same insertion operator
);

// Add a two-body spin interaction term
mpo_gen.AddTerm(Jz, sz, i, sz, j);
// Add a hopping term in t-J type model with J-W string
mpo_gen.AddTerm(-t, adagup, i, aup, j, f);
```

If you want to generate one-body term, you can use a more simplified API.
```cpp
mpo_gen.AddTerm(coef, phys_op, site_idx);

// Add an on-site Hubbard interaction term
mpo_gen.AddTerm(U, nupndn, i);
```

Finally, call member function `Gen()` to generate the MPO.
```cpp
auto mpo = mpo_gen.Gen();
```
The type of the result MPO `mpo` is gqmps2::MPO which is an alias of gqmps2::TenVec at current stage.

### MPS optimization algorithms
#### Two-site variational MPS algorithm
GraceQ/MPS2 offer gqmps2::TwoSiteFiniteVMPS function to perform two-site DMRG algorithm in the language of one-dimensional tensor network (MPS and MPO). To call this function, we should define the runtime parameters first using gqmps2::SweepParams.
```cpp
gqmps2::SweepParams sweep_params(
    Sweeps,                   // Number of DMRG sweeps
    Dmin, Dmax, TruncErr,     // The minimal kept bond dimension, the maximal kept bond dimension, and the target truncation error
    gqmps2::LanczosParams(    // Parameters used by Lanczos matrix diagonalization algorithm
        LanczErr,             // The target Lanczos error
        MaxLanczIter          // The maximal Lanczos iterations
    ),
    MpsPath,                  // The path of the MPS folder. The default value is "mps"
    TempPath                  // The path of the runtime temporary folder. The default value is ".temp"
);
```

Then you can call the algorithm as the follow
```cpp
// Set the number of threads for tensor transpose calculation
// If skip this procedure, 4 threads will be used
size_t TenTransNumThreads = 6;
gqten::hp_numeric::SetTensorTransposeNumThreads(TenTransNumThreads);

// Perform two-site DMRG and return the finial energy of the system
auto e0 = gqmps2::TwoSiteFiniteVMPS(
              mps,            // MPS container
              mpo,            // Hamiltonian's MPO
              sweep_params    // Sweep parameters
          );
```

Users should always note: to obatin the maximal calculation ability, the implementation of gqmps2::TwoSiteFiniteVMPS follows the _minimal-memory-usage_ policy. That means the tensors (MPS local tensors, left envorinment tensors, and right envorinemt tensors) are loaded into main memory only when they will be used in the next step, and dumped to hard disk after they are used as soon as possible. The `mps` in gqmps2::TwoSiteFiniteVMPS arguments is only a **runtime container**. After the function returned, the `mps` is empty.
```cpp
mps.empty() == true;    // Will be true
```
And the updated MPS is stored at `sweep_params.mps_path`. Also, **before** the function is called, the MPS stored at `sweep_params.mps_path` should be right canonical and tensors stored at the runtime temporary folder `sweep_params.temp_path` (if the path is existed) should match the MPS stored at `sweep_params.mps_path` and be ready to start a new DMRG sweep. GraceQ/MPS2 does **NOT** check these and users should **guarantee** these before calling gqmps2::TwoSiteFiniteVMPS.

The following code is a complete demo to use GraceQ/MPS2 to obtain the ground state of a finite one-dimensional spin-1/2 Heisenberg chain.
```cpp
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"


using TenElemT = gqten::GQTEN_Double;
using QNT = gqten::QN<gqten::U1QNVal>;
using QNSctT = gqten::QNSector<QNT>;
using IndexT = gqten::Index<QNT>;
using Tensor = gqten::GQTensor<TenElemT, QNT>;
using SiteVecT = gqmps2::SiteVec<TenElemT, QNT>;
using FiniteMPST = gqmps2::FiniteMPS<TenElemT, QNT>;


int main() {
  auto qn0 =  QNT({gqten::QNCard("Sz", gqten::U1QNVal( 0))});
  auto qnup = QNT({gqten::QNCard("Sz", gqten::U1QNVal( 1))});
  auto qndn = QNT({gqten::QNCard("Sz", gqten::U1QNVal(-1))});

  auto pb_out = IndexT(
                    {QNSctT(qnup, 1), QNSctT(qndn, 1)},
                    gqten::GQTenIndexDirType::OUT
                );
  auto pb_in = gqten::InverseIndex(pb_out);

  Tensor sz({pb_in, pb_out});
  Tensor sp({pb_in, pb_out});
  Tensor sm({pb_in, pb_out});
  sz(0, 0) =  0.5;
  sz(1, 1) = -0.5;
  sp(0, 1) = 1;
  sm(1, 0) = 1;

  size_t N = 20;
  SiteVecT sites(N, pb_out);

  gqmps2::MPOGenerator<TenElemT, QNT> mpo_gen(sites, qn0);
  for (size_t i = 0; i < N-1; ++i) {
    mpo_gen.AddTerm(1.0, sz, i, sz, i+1);
    mpo_gen.AddTerm(0.5, sp, i, sm, i+1);
    mpo_gen.AddTerm(0.5, sm, i, sp, i+1);
  }
  auto mpo = mpo_gen.Gen();

  size_t Sweeps = 10;
  size_t Dmin = 16;
  size_t Dmax = 24;
  double TruncErr = 1e-7;
  double LanczErr = 1e-8;
  size_t MaxLanczIter = 100;
  gqmps2::SweepParams sweep_params(
      Sweeps,
      Dmin, Dmax, TruncErr,
      gqmps2::LanczosParams(LanczErr, MaxLanczIter)
  );

  FiniteMPST mps(sites);
  std::vector<size_t> stat_labs;
  for (size_t i = 0; i < N; ++i) { stat_labs.push_back(i % 2); }
  gqmps2::DirectStateInitMps(mps, stat_labs, qn0);
  mps.Dump(sweep_params.mps_path, true);

  size_t TenTransNumThreads = 4;
  gqten::hp_numeric::SetTensorTransposeNumThreads(TenTransNumThreads);
  auto e0 = gqmps2::TwoSiteFiniteVMPS(mps, mpo, sweep_params);

  std::cout << "E0/site: " << e0 / N << std::endl;    // E0/site: -0.43412
  return 0;
}
```
> Note: before performing a new run, please remove the `.temp` folder.

### Input parameters parser: gqmps2::CaseParamsParserBasic
...


## Developer guide
...


## License
GraceQ/MPS2 is freely available under the [LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html) licence.


## How to cite
You can cite the GraceQ/MPS2 where you use it as a support to this project. Please cite GraceQ/MPS2 as
> GraceQuantum.org . GraceQ/MPS2: A high-performance matrix product state algorithms library based on GraceQ/tensor. Homepage: https://mps2.gracequantum.org . For a complete list of the contributors, see CONTRIBUTORS.txt .


## Acknowledgments
We highly acknowledge the following people, project(s) and organization(s) (sorted in alphabetical order):

ALPS project, Chunyu Sun, Donna Sheng, Grace Song, Hao-Kai Zhang, Hao-Xin Wang, Hong-Chen Jiang, Hong-Hao Tu, Hui-Ke Jin, itensor.org, Jisi Xu, Le Zhao, Shuai Chen, Shuo Yang, Thomas P. Devereaux, Wayne Zheng, Xiaoyu Dong, Yi Zhou, Yifan Jiang, Zheng-Yu Weng

You can not meet this project without anyone of them. And the basic part of this project (before version 0.1) was developed by Rong-Yang Sun and Cheng Peng, when Rong-Yang Sun was a visiting student at Stanford University. So R.-Y. Sun want to give special thanks to his co-advisors Hong-Chen Jiang, Prof. Thomas P. Devereaux and their postdoctors Yifan Jiang and Cheng Peng.
