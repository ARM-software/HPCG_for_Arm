# HPCG for Arm

## Introduction

The HPCG benchmark is a tool for ranking HPC systems that implements a preconditioned conjugate gradient solver. The benchmark generates a regular sparse linear system that is mathematically similar to a finite difference discretization of a three-dimensional heat diffusion equation. The problem is solved using domain decomposition, where each subdomain is preconditioned using a symmetric Gauss-Seidel sweep.

This repository contains an optimized version of HPCG for Arm that make use of optimized mathematical libraries such as the Arm Performance Libraries, NEON and SVE intrinsics. The main kernels have been modified to enable shared-memory parallelism. Further information about the code can be found in the following publications:

  * Arm Community blog
    * [Parallelizing HPCG's main kernels](https://community.arm.com/developer/tools-software/hpc/b/hpc-blog/posts/parallelizing-hpcg)
    * [Optimizing HPCG for Arm SVE](https://community.arm.com/developer/tools-software/hpc/b/hpc-blog/posts/optimizing-hpcg-for-arm-sve)
  * Presentations
    * [SC'18 - HPCG BoF](https://www.hpcg-benchmark.org/downloads/sc18/sc18_hpcg_bof.pdf)


## Optimizations

### Shared-memory parallelism

This optimized code implements different parallelization techniques in the symmetric Gauss-Seidel kernel depending on the subdomain. For the finest level, we implement a task dependency graph. For the coarser levels, we parallelize the Gauss-Seidel kernel by using the multi-block colouring technique.

In the task dependency graph, nodes of the grid are processed as soon as all their dependencies are fulfilled. The amount of parallelism increases as nodes are processed until reaching a peak, then it decreases again. Therefore, this technique benefits from the larger grids such as the one present in the finest level of the multigrid.

The multi-block colouring technique groups consecutive nodes in blocks. Colours are assigned to blocks in a way that, given 2 blocks, any node from the first block does not have dependencies to any node from the second block. Parallelism is achieved by processing blocks with the same colour at the same time. This technique enables more parallelism than the task dependency graph one, therefore, it is a better fit for the coarser levels of the multigrid.

### Vectorization

While processing blocks, consecutive nodes depend on each other. To break this dependency and thus create vectorization opportunities, blocks with the same colour are interleaved at the node level. This enables the Gauss-Seidel kernel to be easier vectorized.

The code currently supports both ``NEON`` and ``SVE`` vector extensions.

### Reordering

Throughout the benchmark, sparse matrices and vectors are reordered in order to improve data locality. The way this reordering is performed depends on the parallelization technique applied.

### Minor tweaks

Loop unrolling has been applied at the different kernels in order to reduce loop overheads. In the same line, nested loops present at the ``GenerateProblem`` routine have been flattened.

## How to build

### Dependencies

The code does not have any dependencies. However, in order to enable some features, external packages are required:

  * An MPI implementation when enabling the MPI build of HPCG
  * A compiler that supports OpenMP syntax when enabling the OpenMP build of HPCG
  * A BLAS implementation when enabling the BLAS-enabled build of HPCG
  * The [Arm Performance Libraries](https://developer.arm.com/tools-and-software/server-and-hpc/arm-architecture-tools/arm-performance-libraries) when enabling the use of sparse matrix routines during the SpMV kernel
  * A compiler that supports NEON intrinsics when enabling the NEON build of HPCG
  * A compiler that supports SVE intrinsics when enabling the SVE build of HPCG

Build configurations are provided in the ``setup`` folder. The naming convention for these configuration files is ``Make.${config}`` Those must be used in order to build the benchmark. It is highly recommended to modify the chosen configuration to better suite your platform.

Builds can be performed in-source or out-of-source. For in-source builds, just type the following:

```
make arch=${config}
```

If the build is successful, the binary ``bin/xhpcg`` will be generated.

For out-of-source builds, type the following commands:

```
mkdir build && cd build
../configure ${config}
# At this point, you can modify again the configuration by editing
# the build/setup/Make.${config} file
make
```

If the build is successful, the binary ``build/bin/xhpcg`` will be generated.

For more detailed information, refer to the ``INSTALL`` file.

### Enabling and disabling implemented features

Configuration files inside the ``setup`` folder set different variables. One of these variables is called ``HPCG_OPTS``. This variable can contain defines that will be used at compilation time. The different defines that can be set are:

```
# -DHPCG_NO_MPI               Define to disable MPI
# -DHPCG_NO_OPENMP	          Define to disable OPENMP
# -DHPCG_CONTIGUOUS_ARRAYS    Define to have sparse matrix arrays long and contiguous
# -DHPCG_USE_DDOT_ARMPL       Define to use Arm Performance Libraries calls in the ComputeDotProduct
# -DHPCG_USE_WAXPBY_ARMPL     Define to use Arm Performance Libraries calls in the ComputeWAXPBY
# -DHPCG_USE_ARMPL_SPMV       Define to use Arm Performance Libraries calls in the ComputeSPMV kernel. Requires ArmPL >= 19.0
# -DHPCG_USE_NEON             Define to use NEON intrinsics in the main kernels
# -DHPCG_USE_SVE              Define to use SVE intrinsics in the main kernels
# -DHPCG_USE_FUSED_SYMGS_SPMV Define to fuse SYMGS and SPMV when possible. This makes the run invalid for submission
# -DHPCG_DEBUG                Define to enable debugging output
# -DHPCG_DETAILED_DEBUG       Define to enable very detailed debugging output
```

## How to run the benchmark

As a quick summary, the benchmark is run with the following commands:

```
cd bin # or cd build/bin

# Set OpenMP threads when using an OpenMP-enabled build
export OMP_NUM_THREADS=4

# Use mpirun when using an MPI-enabled build
[mpirun -np 4 ...] ./xhpcg [optional in-line flags]
```

The ``xhpcg`` binary will read by default the ``hpcg.dat`` file, if present. This file contains 4 lines. The first 2 lines are ignored. The third line contains three numbers that are used to specify the local (to an MPI process) dimensions of the problem. The fourth and last line specify the number of seconds the timed portion of the benchmark should run for.

By default, the file contains the following:

```
HPCG benchmark input file
Sandia National Laboratories; University of Tennessee, Knoxville
104 104 104
60
```

Meaning that the local domain will have a dimension of ``104x104x104`` and that the timed portion of the benchmark will run for, at least, 60 seconds.

If the file is not present, you can use the following command line flags:

  *  ``--nx=<n>`` to specify the local dimension in the X-axis
  *  ``--ny=<m>`` to specify the local dimension in the Y-axis
  *  ``--nz=<l>`` to specify the local dimension in the Z-axis
  *  ``--rt=<t>`` to specify the number of seconds the timed portion of the benchmark should run.

If the minimum execution time of the timed portion is set to ``0``, ``QuickPath`` is enabled. This will minimize the number of steps executed throughout the benchmark and will reduce the number of conjugate gradients to be executed to 1.

Due to some of the optimizations performed in the code, there are some constraints when selecting the local domain dimensions that must considered. Therefore, the local subdomain must fulfill the following requirements:

  1. ``nx``, ``ny`` and ``nz`` must be even numbers of every level of the grid
  2. ``nx``, ``ny`` and ``nz`` must be greater or equal to 32

## Output

The benchmark generates two different files. One contains general information of some parts of the execution such as residual generated during the different tests or the residual obtained after each conjugate gradient iteration. This file is updated throughout the execution.

The second file is generated at the very end of the execution and provides detailed metrics such as the problem local and global dimensions, number of MPI processes and OpenMP threads used and number of floating-point operations.

This file also contains the obtained ``GFLOPS`` and an estimation of the memory throughput.

## Valid runs

Official runs must be at least 1800 seconds (30 minutes) as reported in the output file. A valid run must also execute a problem size that is large enough so that data arrays accessed in the conjugate gradient iteration loop do not fit in the cache of the device. This restriction means that the problem size should be large enough to occupy a significant fraction of main memory, at least 1/4 of the total.

## License

This project is licensed under Apache-2.0.

This project includes some third-party code under other open source licenses. For more information, see ``LICENSE``.

## Contributions / Pull Requests

Contributions are accepted under Apache-2.0. Only submit contributions where you have authored all of the code. If you do this on work time, make sure you have your employer's approval.
