/*
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Copyright (C) 2019, Arm Limited and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */


//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ReportResults.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#include <vector>
#include <unistd.h>
#include "ReportResults.hpp"
#include "OutputFile.hpp"
#include "OptimizeProblem.hpp"

#ifdef HPCG_DEBUG
#include <fstream>
using std::endl;

#include "hpcg.hpp"
#endif
#include <iostream>

/*!
 Creates a YAML file and writes the information about the HPCG run, its results, and validity.

  @param[in] geom The description of the problem's geometry.
  @param[in] A    The known system matrix
  @param[in] numberOfMgLevels Number of levels in multigrid V cycle
  @param[in] numberOfCgSets Number of CG runs performed
  @param[in] niters Number of preconditioned CG iterations performed to lower the residual below a threshold
  @param[in] times  Vector of cumulative timings for each of the phases of a preconditioned CG iteration
  @param[in] testcg_data    the data structure with the results of the CG-correctness test including pass/fail information
  @param[in] testsymmetry_data the data structure with the results of the CG symmetry test including pass/fail information
  @param[in] testnorms_data the data structure with the results of the CG norm test including pass/fail information
  @param[in] global_failure indicates whether a failure occurred during the correctness tests of CG

  @see YAML_Doc
*/
void ReportResults(const SparseMatrix & A, int numberOfMgLevels, int numberOfCgSets, int refMaxIters,int optMaxIters, double times[],
    const TestCGData & testcg_data, const TestSymmetryData & testsymmetry_data, const TestNormsData & testnorms_data, int global_failure, bool quickPath) {

  double minOfficialTime = 1800; // Any official benchmark result must run at least this many seconds

#ifndef HPCG_NO_MPI
  double t4 = times[4];
  double t4min = 0.0;
  double t4max = 0.0;
  double t4avg = 0.0;
  MPI_Allreduce(&t4, &t4min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&t4, &t4max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&t4, &t4avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  t4avg = t4avg/((double) A.geom->size);
#endif

  if (A.geom->rank==0) { // Only PE 0 needs to compute and report timing results

    // TODO: Put the FLOP count, Memory BW and Memory Usage models into separate functions

    // ======================== FLOP count model =======================================

    double fNumberOfCgSets = numberOfCgSets;
    double fniters = fNumberOfCgSets * (double) optMaxIters;
    double fnrow = A.totalNumberOfRows;
    double fnnz = A.totalNumberOfNonzeros;

    // Op counts come from implementation of CG in CG.cpp (include 1 extra for the CG preamble ops)
    double fnops_ddot = (3.0*fniters+fNumberOfCgSets)*2.0*fnrow; // 3 ddots with nrow adds and nrow mults
    double fnops_waxpby = (3.0*fniters+fNumberOfCgSets)*2.0*fnrow; // 3 WAXPBYs with nrow adds and nrow mults
    double fnops_sparsemv = (fniters+fNumberOfCgSets)*2.0*fnnz; // 1 SpMV with nnz adds and nnz mults
    // Op counts from the multigrid preconditioners
    double fnops_precond = 0.0;
    const SparseMatrix * Af = &A;
    for (int i=1; i<numberOfMgLevels; ++i) {
      double fnnz_Af = Af->totalNumberOfNonzeros;
      double fnumberOfPresmootherSteps = Af->mgData->numberOfPresmootherSteps;
      double fnumberOfPostsmootherSteps = Af->mgData->numberOfPostsmootherSteps;
      fnops_precond += fnumberOfPresmootherSteps*fniters*4.0*fnnz_Af; // number of presmoother flops
      fnops_precond += fniters*2.0*fnnz_Af; // cost of fine grid residual calculation
      fnops_precond += fnumberOfPostsmootherSteps*fniters*4.0*fnnz_Af;  // number of postsmoother flops
      Af = Af->Ac; // Go to next coarse level
    }

    fnops_precond += fniters*4.0*((double) Af->totalNumberOfNonzeros); // One symmetric GS sweep at the coarsest level
    double fnops = fnops_ddot+fnops_waxpby+fnops_sparsemv+fnops_precond;
    double frefnops = fnops * ((double) refMaxIters)/((double) optMaxIters);

    // ======================== Memory bandwidth model =======================================

    // Read/Write counts come from implementation of CG in CG.cpp (include 1 extra for the CG preamble ops)
    double fnreads_ddot = (3.0*fniters+fNumberOfCgSets)*2.0*fnrow*sizeof(double); // 3 ddots with 2 nrow reads
    double fnwrites_ddot = (3.0*fniters+fNumberOfCgSets)*sizeof(double); // 3 ddots with 1 write
    double fnreads_waxpby = (3.0*fniters+fNumberOfCgSets)*2.0*fnrow*sizeof(double); // 3 WAXPBYs with nrow adds and nrow mults
    double fnwrites_waxpby = (3.0*fniters+fNumberOfCgSets)*fnrow*sizeof(double); // 3 WAXPBYs with nrow adds and nrow mults
    double fnreads_sparsemv = (fniters+fNumberOfCgSets)*(fnnz*(sizeof(double)+sizeof(local_int_t)) + fnrow*sizeof(double));// 1 SpMV with nnz reads of values, nnz reads indices,
    // plus nrow reads of x
    double fnwrites_sparsemv = (fniters+fNumberOfCgSets)*fnrow*sizeof(double); // 1 SpMV nrow writes
    // Op counts from the multigrid preconditioners
    double fnreads_precond = 0.0;
    double fnwrites_precond = 0.0;
    Af = &A;
    for (int i=1; i<numberOfMgLevels; ++i) {
      double fnnz_Af = Af->totalNumberOfNonzeros;
      double fnrow_Af = Af->totalNumberOfRows;
      double fnumberOfPresmootherSteps = Af->mgData->numberOfPresmootherSteps;
      double fnumberOfPostsmootherSteps = Af->mgData->numberOfPostsmootherSteps;
      fnreads_precond += fnumberOfPresmootherSteps*fniters*(2.0*fnnz_Af*(sizeof(double)+sizeof(local_int_t)) + fnrow_Af*sizeof(double)); // number of presmoother reads
      fnwrites_precond += fnumberOfPresmootherSteps*fniters*fnrow_Af*sizeof(double); // number of presmoother writes
      fnreads_precond += fniters*(fnnz_Af*(sizeof(double)+sizeof(local_int_t)) + fnrow_Af*sizeof(double)); // Number of reads for fine grid residual calculation
      fnwrites_precond += fniters*fnnz_Af*sizeof(double); // Number of writes for fine grid residual calculation
      fnreads_precond += fnumberOfPostsmootherSteps*fniters*(2.0*fnnz_Af*(sizeof(double)+sizeof(local_int_t)) + fnrow_Af*sizeof(double));  // number of postsmoother reads
      fnwrites_precond += fnumberOfPostsmootherSteps*fniters*fnnz_Af*sizeof(double);  // number of postsmoother writes
      Af = Af->Ac; // Go to next coarse level
    }

    double fnnz_Af = Af->totalNumberOfNonzeros;
    double fnrow_Af = Af->totalNumberOfRows;
    fnreads_precond += fniters*(2.0*fnnz_Af*(sizeof(double)+sizeof(local_int_t)) + fnrow_Af*sizeof(double)); // One symmetric GS sweep at the coarsest level
    fnwrites_precond += fniters*fnrow_Af*sizeof(double); // One symmetric GS sweep at the coarsest level
    double fnreads = fnreads_ddot+fnreads_waxpby+fnreads_sparsemv+fnreads_precond;
    double fnwrites = fnwrites_ddot+fnwrites_waxpby+fnwrites_sparsemv+fnwrites_precond;
    double frefnreads = fnreads * ((double) refMaxIters)/((double) optMaxIters);
    double frefnwrites = fnwrites * ((double) refMaxIters)/((double) optMaxIters);


    // ======================== Memory usage model =======================================

    // Data in GenerateProblem_ref

    double numberOfNonzerosPerRow = 27.0; // We are approximating a 27-point finite element/volume/difference 3D stencil
    double size = ((double) A.geom->size); // Needed for estimating size of halo

    double fnbytes = ((double) sizeof(Geometry));      // Geometry struct in main.cpp
    fnbytes += ((double) sizeof(double)*fNumberOfCgSets); // testnorms_data in main.cpp

    // Model for GenerateProblem_ref.cpp
    fnbytes += fnrow*sizeof(char);      // array nonzerosInRow
    fnbytes += fnrow*((double) sizeof(global_int_t*)); // mtxIndG
    fnbytes += fnrow*((double) sizeof(local_int_t*));  // mtxIndL
    fnbytes += fnrow*((double) sizeof(double*));      // matrixValues
    fnbytes += fnrow*((double) sizeof(double*));      // matrixDiagonal
    fnbytes += fnrow*numberOfNonzerosPerRow*((double) sizeof(local_int_t));  // mtxIndL[1..nrows]
    fnbytes += fnrow*numberOfNonzerosPerRow*((double) sizeof(double));       // matrixValues[1..nrows]
    fnbytes += fnrow*numberOfNonzerosPerRow*((double) sizeof(global_int_t)); // mtxIndG[1..nrows]
    fnbytes += fnrow*((double) 3*sizeof(double)); // x, b, xexact

    // Model for CGData.hpp
    double fncol = ((global_int_t) A.localNumberOfColumns) * size; // Estimate of the global number of columns using the value from rank 0
    fnbytes += fnrow*((double) 2*sizeof(double)); // r, Ap
    fnbytes += fncol*((double) 2*sizeof(double)); // z, p

    std::vector<double> fnbytesPerLevel(numberOfMgLevels); // Count byte usage per level (level 0 is main CG level)
    fnbytesPerLevel[0] = fnbytes;

    // Benchmarker-provided model for OptimizeProblem.cpp
    double fnbytes_OptimizedProblem = OptimizeProblemMemoryUse(A);
    fnbytes += fnbytes_OptimizedProblem;

    Af = A.Ac;
    for (int i=1; i<numberOfMgLevels; ++i) {
      double fnrow_Af = Af->totalNumberOfRows;
      double fncol_Af = ((global_int_t) Af->localNumberOfColumns) * size; // Estimate of the global number of columns using the value from rank 0
      double fnbytes_Af = 0.0;
      // Model for GenerateCoarseProblem.cpp
      fnbytes_Af += fnrow_Af*((double) sizeof(local_int_t)); // f2cOperator
      fnbytes_Af += fnrow_Af*((double) sizeof(double)); // rc
      fnbytes_Af += 2.0*fncol_Af*((double) sizeof(double)); // xc, Axf are estimated based on the size of these arrays on rank 0
      fnbytes_Af += ((double) (sizeof(Geometry)+sizeof(SparseMatrix)+3*sizeof(Vector)+sizeof(MGData))); // Account for structs geomc, Ac, rc, xc, Axf - (minor)

      // Model for GenerateProblem.cpp (called within GenerateCoarseProblem.cpp)
      fnbytes_Af += fnrow_Af*sizeof(char);      // array nonzerosInRow
      fnbytes_Af += fnrow_Af*((double) sizeof(global_int_t*)); // mtxIndG
      fnbytes_Af += fnrow_Af*((double) sizeof(local_int_t*));  // mtxIndL
      fnbytes_Af += fnrow_Af*((double) sizeof(double*));      // matrixValues
      fnbytes_Af += fnrow_Af*((double) sizeof(double*));      // matrixDiagonal
      fnbytes_Af += fnrow_Af*numberOfNonzerosPerRow*((double) sizeof(local_int_t));  // mtxIndL[1..nrows]
      fnbytes_Af += fnrow_Af*numberOfNonzerosPerRow*((double) sizeof(double));       // matrixValues[1..nrows]
      fnbytes_Af += fnrow_Af*numberOfNonzerosPerRow*((double) sizeof(global_int_t)); // mtxIndG[1..nrows]

      // Model for SetupHalo_ref.cpp
#ifndef HPCG_NO_MPI
      fnbytes_Af += ((double) sizeof(double)*Af->totalToBeSent); //sendBuffer
      fnbytes_Af += ((double) sizeof(local_int_t)*Af->totalToBeSent); // elementsToSend
      fnbytes_Af += ((double) sizeof(int)*Af->numberOfSendNeighbors); // neighbors
      fnbytes_Af += ((double) sizeof(local_int_t)*Af->numberOfSendNeighbors); // receiveLength, sendLength
#endif
      fnbytesPerLevel[i] = fnbytes_Af;
      fnbytes += fnbytes_Af; // Running sum
      Af = Af->Ac; // Go to next coarse level
    }

    assert(Af==0); // Make sure we got to the lowest grid level

    // Count number of bytes used per equation
    double fnbytesPerEquation = fnbytes/fnrow;

    // Instantiate YAML document
	char execConf[128];
	char hostname[128];
    std::cout << "HPCG-Benchmark_3.0" << std::endl;
    std::cout << "Release date " << "November 11, 2015" << std::endl;

    std::cout << "Machine Summary" << std::endl;
    std::cout << "Machine Summary " << "Distributed Processes " << A.geom->size << std::endl;
    std::cout << "Machine Summary " << "Threads per processes " << A.geom->numThreads << std::endl;

    std::cout << "Global Problem Dimensions" << std::endl;
    std::cout << "Global Problem Dimensions " << "Global nx " << A.geom->gnx << std::endl;
    std::cout << "Global Problem Dimensions " << "Global ny " << A.geom->gny << std::endl;
    std::cout << "Global Problem Dimensions " << "Global nz " << A.geom->gnz << std::endl;

    std::cout << "Processor Dimensions" << std::endl;
    std::cout << "Processor Dimensions " << "npx " << A.geom->npx << std::endl;
    std::cout << "Processor Dimensions " << "npy " << A.geom->npy << std::endl;
    std::cout << "Processor Dimensions " << "npz " << A.geom->npz << std::endl;

    std::cout << "Local Domain Dimensions" << std::endl;
    std::cout << "Local Domain Dimensions " << "nx " << A.geom->nx << std::endl;
    std::cout << "Local Domain Dimensions " << "ny " << A.geom->ny << std::endl;

    int ipartz_ids = 0;
    for (int i=0; i< A.geom->npartz; ++i) {
        std::cout << "Local Domain Dimensions " << "Lower ipz " << ipartz_ids << std::endl;
        std::cout << "Local Domain Dimensions " << "Upper ipz " << A.geom->partz_ids[i]-1 << std::endl;
        std::cout << "Local Domain Dimensions " << "nz " << A.geom->partz_nz[i] << std::endl;
      ipartz_ids = A.geom->partz_ids[i];
    }


    std::cout << "########## Problem Summary  ##########" << std::endl;

    std::cout << "Setup Information" << std::endl;
    std::cout << "Setup Information " << "Setup Time " << times[9] << std::endl;

    std::cout << "Linear System Information" << std::endl;
    std::cout << "Linear System Information " << "Number of Equations " << A.totalNumberOfRows << std::endl;
    std::cout << "Linear System Information " << "Number of Nonzero Terms " << A.totalNumberOfNonzeros << std::endl;

    std::cout << "Multigrid Information" << std::endl;
    std::cout << "Multigrid Information " << "Number of coarse grid levels " << numberOfMgLevels-1 << std::endl;
    Af = &A;
    std::cout << "Multigrid Informatioan " << "Coarse Grids" << std::endl;
    for (int i=1; i<numberOfMgLevels; ++i) {
        std::cout << "Multigrid Information " << "Coarse Grids " << "Grid Level" << i << std::endl;
        std::cout << "Multigrid Information " << "Coarse Grids " << "Number of Equations " << Af->Ac->totalNumberOfRows << std::endl;
        std::cout << "Multigrid Information " << "Coarse Grids " << "Number of Nonzero Terms " << Af->Ac->totalNumberOfNonzeros << std::endl;
        std::cout << "Multigrid Information " << "Coarse Grids " << "Number of Presmoother Steps " << Af->mgData->numberOfPresmootherSteps << std::endl;
        std::cout << "Multigrid Information " << "Coarse Grids " << "Number of Postsmoother Steps " << Af->mgData->numberOfPostsmootherSteps << std::endl;
      Af = Af->Ac;
    }

    std::cout << "########## Memory Use Summary  ##########" << std::endl;

    std::cout << "Memory Use Information" << std::endl;
    std::cout << "Memory Use Information " << "Total memory used for data (Gbytes) " << fnbytes/1000000000.0 << std::endl;
    std::cout << "Memory Use Information " << "Memory used for OptimizeProblem data (Gbytes) " << fnbytes_OptimizedProblem/1000000000.0 << std::endl;
    std::cout << "Memory Use Information " << "Bytes per equation (Total memory / Number of Equations) " << fnbytesPerEquation << std::endl;

    std::cout << "Memory Use Information " << "Memory used for linear system and CG (Gbytes) " << fnbytesPerLevel[0]/1000000000.0 << std::endl;

    std::cout << "Memory Use Information " << "Coarse Grids" << std::endl;
    for (int i=1; i<numberOfMgLevels; ++i) {
        std::cout << "Memory Use Information " << "Coarse Grids " << "Grid Level " << i << std::endl;
        std::cout << "Memory Use Information " << "Coarse Grids " << "Memory used " << fnbytesPerLevel[i]/1000000000.0 << std::endl;
    }

    std::cout << "########## V&V Testing Summary  ##########" << std::endl;
    std::cout << "Spectral Convergence Tests" << std::endl;
    if (testcg_data.count_fail==0)
      std::cout << "Spectral Convergence Tests " << "Result " << "PASSED" << std::endl;
    else
      std::cout << "Spectral Convergence Tests " << "Result " << "FAILED" << std::endl;
    std::cout << "Spectral Convergence Tests " << "Unpreconditioned" << std::endl;
    std::cout << "Spectral Convergence Tests " << "Unpreconditioned " << "Maximum iteration count " << testcg_data.niters_max_no_prec << std::endl;
    std::cout << "Spectral Convergence Tests " << "Unpreconditioned " << "Expected iteration count " << testcg_data.expected_niters_no_prec << std::endl;
    std::cout << "Spectral Convergence Tests " << "Preconditioned" << std::endl;
    std::cout << "Spectral Convergence Tests " << "Preconditioned " << "Maximum iteration count " << testcg_data.niters_max_prec << std::endl;
    std::cout << "Spectral Convergence Tests " << "Preconditioned " << "Expected iteration count " << testcg_data.expected_niters_prec << std::endl;

    const char DepartureFromSymmetry[] = "Departure from Symmetry |x'Ay-y'Ax|/(2*||x||*||A||*||y||)/epsilon";
    std::cout << DepartureFromSymmetry << std::endl;
    if (testsymmetry_data.count_fail==0)
      std::cout << DepartureFromSymmetry << "Result " << "PASSED" << std::endl;
    else
      std::cout << DepartureFromSymmetry << "Result " << "FAILED" << std::endl;
    std::cout << DepartureFromSymmetry << "Departure for SpMV " << testsymmetry_data.depsym_spmv << std::endl;
    std::cout << DepartureFromSymmetry << "Departure for MG " << testsymmetry_data.depsym_mg << std::endl;

    std::cout << "########## Iterations Summary  ##########" << std::endl;
    std::cout << "Iteration Count Information" << std::endl;
    if (!global_failure)
      std::cout << "Iteration Count Information " << "Result " << "PASSED" << std::endl;
    else
      std::cout << "Iteration Count Information" << "Result " << "FAILED" << std::endl;
    std::cout << "Iteration Count Information" << "Reference CG iterations per set " << refMaxIters << std::endl;
    std::cout << "Iteration Count Information" << "Optimized CG iterations per set " << optMaxIters << std::endl;
    std::cout << "Iteration Count Information" << "Total number of reference iterations " << refMaxIters*numberOfCgSets << std::endl;
    std::cout << "Iteration Count Information" << "Total number of optimized iterations " << optMaxIters*numberOfCgSets << std::endl;

    std::cout << "########## Reproducibility Summary  ##########" << std::endl;
    std::cout << "Reproducibility Information" << std::endl;
    if (testnorms_data.pass)
      std::cout << "Reproducibility Information " << "Result " << "PASSED" << std::endl;
    else
      std::cout << "Reproducibility Information " << "Result " << "FAILED" << std::endl;
    std::cout << "Reproducibility Information " << "Scaled residual mean " << testnorms_data.mean << std::endl;
    std::cout << "Reproducibility Information " << "Scaled residual variance " << testnorms_data.variance << std::endl;

    std::cout << "########## Performance Summary (times in sec) ##########" << std::endl;

    std::cout << "Benchmark Time Summary" << std::endl;
    std::cout << "Benchmark Time Summary " << "Optimization phase " << times[7] << std::endl;
    std::cout << "Benchmark Time Summary " << "DDOT " << times[1] << std::endl;
    std::cout << "Benchmark Time Summary " << "WAXPBY " << times[2] << std::endl;
    std::cout << "Benchmark Time Summary " << "SpMV " << times[3] << std::endl;
    std::cout << "Benchmark Time Summary " << "MG " << times[5] << std::endl;
    std::cout << "Benchmark Time Summary " << "Total " << times[0] << std::endl;

    std::cout << "Floating Point Operations Summary" << std::endl;
    std::cout << "Floating Point Operations Summary " << "Raw DDOT " << fnops_ddot << std::endl;
    std::cout << "Floating Point Operations Summary " << "Raw WAXPBY " << fnops_waxpby << std::endl;
    std::cout << "Floating Point Operations Summary " << "Raw SpMV " << fnops_sparsemv << std::endl;
    std::cout << "Floating Point Operations Summary " << "Raw MG " << fnops_precond << std::endl;
    std::cout << "Floating Point Operations Summary " << "Total " << fnops << std::endl;
    std::cout << "Floating Point Operations Summary " << "Total with convergence overhead " << frefnops << std::endl;

    std::cout << "GB/s Summary" << std::endl;
    std::cout << "GB/s Summary " << "Raw Read B/W " << fnreads/times[0]/1.0E9 << std::endl;
    std::cout << "GB/s Summary " << "Raw Write B/W " << fnwrites/times[0]/1.0E9 << std::endl;
    std::cout << "GB/s Summary " << "Raw Total B/W " << (fnreads+fnwrites)/(times[0])/1.0E9 << std::endl;
    std::cout << "GB/s Summary " << "Total with convergence and optimization phase overhead " << (frefnreads+frefnwrites)/(times[0]+fNumberOfCgSets*(times[7]/10.0+times[9]/10.0))/1.0E9 << std::endl;


    std::cout << "GFLOP/s Summary" << std::endl;
    std::cout << "GFLOP/s Summary " << "Raw DDOT " << fnops_ddot/times[1]/1.0E9 << std::endl;
    std::cout << "GFLOP/s Summary " << "Raw WAXPBY " << fnops_waxpby/times[2]/1.0E9 << std::endl;
    std::cout << "GFLOP/s Summary " << "Raw SpMV " << fnops_sparsemv/(times[3])/1.0E9 << std::endl;
    std::cout << "GFLOP/s Summary " << "Raw MG " << fnops_precond/(times[5])/1.0E9 << std::endl;
    std::cout << "GFLOP/s Summary " << "Raw Total " << fnops/times[0]/1.0E9 << std::endl;
    std::cout << "GFLOP/s Summary " << "Total with convergence overhead " << frefnops/times[0]/1.0E9 << std::endl;
    // This final GFLOP/s rating includes the overhead of problem setup and optimizing the data structures vs ten sets of 50 iterations of CG
    double totalGflops = frefnops/(times[0]+fNumberOfCgSets*(times[7]/10.0+times[9]/10.0))/1.0E9;
    double totalGflops24 = frefnops/(times[0]+fNumberOfCgSets*times[7]/10.0)/1.0E9;
    std::cout << "GFLOP/s Summary " << "Total with convergence and optimization phase overhead " << totalGflops << std::endl;

    std::cout << "User Optimization Overheads" << std::endl;
    std::cout << "User Optimization Overheads " << "Optimization phase time (sec) " <<  (times[7]) << std::endl;
    std::cout << "User Optimization Overheads " << "Optimization phase time vs reference SpMV+MG time " <<  times[7]/times[8] << std::endl;

#ifndef HPCG_NO_MPI
    std::cout << "DDOT Timing Variations" << "" << std::endl;
    std::cout << "DDOT Timing Variations " << "Min DDOT MPI_Allreduce time " << t4min << std::endl;
    std::cout << "DDOT Timing Variations " << "Max DDOT MPI_Allreduce time " << t4max << std::endl;
    std::cout << "DDOT Timing Variations " << "Avg DDOT MPI_Allreduce time " << t4avg << std::endl;

    //doc.get("Sparse Operations Overheads")->add("Halo exchange time (sec)", (times[6]));
    //doc.get("Sparse Operations Overheads")->add("Halo exchange as percentage of SpMV time", (times[6])/totalSparseMVTime*100.0);
#endif
    std::cout << "Final Summary" << std::endl;
    bool isValidRun = (testcg_data.count_fail==0) && (testsymmetry_data.count_fail==0) && (testnorms_data.pass) && (!global_failure);
#ifdef HPCG_USE_FUSED_SYMGS_SPMV
	isValidRun = false;
#endif
    if (isValidRun) {
      std::cout << "Final Summary " << "HPCG result is VALID with a GFLOP/s rating of " <<  totalGflops << std::endl;
      std::cout << "Final Summary " << "HPCG 2.4 rating for historical reasons is " <<  totalGflops24 << std::endl;
      if (!A.isDotProductOptimized) {
        std::cout << "Final Summary " << "Reference version of ComputeDotProduct used " << "Performance results are most likely suboptimal " << std::endl;
      }
      if (!A.isSpmvOptimized) {
        std::cout << "Final Summary " << "Reference version of ComputeSPMV used " << "Performance results are most likely suboptimal " << std::endl;
      }
      if (!A.isMgOptimized) {
        if (A.geom->numThreads>1)
          std::cout << "Final Summary " << "Reference version of ComputeMG used and number of threads greater than 1 " << "Performance results are severely suboptimal " << std::endl;
        else // numThreads ==1
          std::cout << "Final Summary " << "Reference version of ComputeMG used " << "Performance results are most likely suboptimal " << std::endl;
      }
      if (!A.isWaxpbyOptimized) {
        std::cout << "Final Summary " << "Reference version of ComputeWAXPBY used " << "Performance results are most likely suboptimal " << std::endl;
      }
      if (times[0]>=minOfficialTime) {
        std::cout << "Final Summary " << "Please upload results from the YAML file contents to " << "http://hpcg-benchmark.org" << std::endl;
      }
      else {
        std::cout << "Final Summary " << "Results are valid but execution time (sec) is " << times[0] << std::endl;
        if (quickPath) {
          std::cout << "Final Summary " << "You have selected the QuickPath option " <<  "Results are official for legacy installed systems with confirmation from the HPCG Benchmark leaders. " << std::endl;
          std::cout << "Final Summary " << "After confirmation please upload results from the YAML file contents to " << "http://hpcg-benchmark.org" << std::endl;
        } else {
          std::cout << "Final Summary " << "Official results execution time (sec) must be at least " << minOfficialTime << std::endl;
        }
      }
    } else {
#ifdef HPCG_USE_FUSED_SYMGS_SPMV
      std::cout << "Final Summary " << "HPCG result is INVALID because SYMGS and SPMV were fused. GFLOP/s rating of " <<  totalGflops << std::endl;
#else
      std::cout << "Final Summary " << "HPCG result is " << "INVALID." << std::endl;
#endif
      std::cout << "Final Summary " << "Please review the YAML file contents " << "You may NOT submit these results for consideration." << std::endl;
    }
  }
  return;
}
