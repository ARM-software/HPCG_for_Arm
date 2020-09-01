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
 @file CG.cpp

 HPCG routine
 */

#include <fstream>
#include <iostream>

#include <cmath>

#include "hpcg.hpp"

#include "CG.hpp"
#include "mytimer.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeMG.hpp"
#include "ComputeDotProduct.hpp"
#include "ComputeWAXPBY.hpp"

/*!
  Routine to compute an approximate solution to Ax = b

  @param[in]    geom The description of the problem's geometry.
  @param[inout] A    The known system matrix
  @param[inout] data The data structure with all necessary CG vectors preallocated
  @param[in]    b    The known right hand side vector
  @param[inout] x    On entry: the initial guess; on exit: the new approximate solution
  @param[in]    max_iter  The maximum number of iterations to perform, even if tolerance is not met.
  @param[in]    tolerance The stopping criterion to assert convergence: if norm of residual is <= to tolerance.
  @param[out]   niters    The number of iterations actually performed.
  @param[out]   normr     The 2-norm of the residual vector after the last iteration.
  @param[out]   normr0    The 2-norm of the residual vector before the first iteration.
  @param[out]   times     The 7-element vector of the timing information accumulated during all of the iterations.
  @param[in]    doPreconditioning The flag to indicate whether the preconditioner should be invoked at each iteration.

  @return Returns zero on success and a non-zero value otherwise.

  @see CG_ref()
*/
int CG(const SparseMatrix & A, CGData & data, const Vector & b, Vector & x,
    const int max_iter, const double tolerance, int & niters, double & normr, double & normr0,
    double * times, bool doPreconditioning) {

  double t_begin = 0;  // Start timing right away
  normr = 0.0;
  double rtz = 0.0, oldrtz = 0.0, alpha = 0.0, beta = 0.0, pAp = 0.0;


  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0, t5 = 0.0;
//#ifndef HPCG_NO_MPI
//  double t6 = 0.0;
//#endif
  local_int_t nrow = A.localNumberOfRows;
  Vector & r = data.r; // Residual vector
  Vector & z = data.z; // Preconditioned residual vector
  Vector & p = data.p; // Direction vector (in MPI mode ncol>=nrow)
  Vector & Ap = data.Ap;

  // Reorder vectors
  Vector rOrdered;
  Vector zOrdered;
  Vector xOrdered;
  InitializeVector(rOrdered, r.localLength);
  rOrdered.values = new double[262144];
  InitializeVector(zOrdered, z.localLength);
  zOrdered.values = new double[262144];
  InitializeVector(xOrdered, x.localLength);
  xOrdered.values = new double[262144];
  CopyAndReorderVector(r, rOrdered, A.whichNewRowIsOldRow);
  CopyAndReorderVector(z, zOrdered, A.whichNewRowIsOldRow);
  CopyAndReorderVector(x, xOrdered, A.whichNewRowIsOldRow);

  if (!doPreconditioning && A.geom->rank==0) std::cout << "WARNING: PERFORMING UNPRECONDITIONED ITERATIONS" << std::endl;

#ifdef HPCG_DEBUG
  int print_freq = 1;
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
#endif
  // p is of length ncols, copy x to p for sparse MV operation
  CopyVector(xOrdered, p);
  ComputeSPMV(A, p, Ap); // Ap = A*p
  ComputeWAXPBY(nrow, 1.0, b, -1.0, Ap, rOrdered, A.isWaxpbyOptimized); // r = b - Ax (x stored in p)
  ComputeDotProduct(nrow, rOrdered, rOrdered, normr, t4, A.isDotProductOptimized);
  normr = sqrt(normr);
#ifdef HPCG_DEBUG
  if (A.geom->rank==0) std::cout << "Initial Residual = "<< normr << std::endl;
#endif

  // Record initial residual for convergence testing
  normr0 = normr;

  // Start iterations

  for (int k=1; k<=max_iter && normr/normr0 > tolerance; k++ ) {
    if (doPreconditioning)
      ComputeMG(A, rOrdered, zOrdered); // Apply preconditioner
    else
      CopyVector (rOrdered, zOrdered); // copy r to z (no preconditioning)

    if (k == 1) {
      ComputeWAXPBY(nrow, 1.0, zOrdered, 0.0, zOrdered, p, A.isWaxpbyOptimized); // Copy Mr to p
      ComputeDotProduct (nrow, rOrdered, zOrdered, rtz, t4, A.isDotProductOptimized); // rtz = r'*z
    } else {
      oldrtz = rtz;
      ComputeDotProduct (nrow, rOrdered, zOrdered, rtz, t4, A.isDotProductOptimized); // rtz = r'*z
      beta = rtz/oldrtz;
      ComputeWAXPBY (nrow, 1.0, zOrdered, beta, p, p, A.isWaxpbyOptimized); // p = beta*p + z
    }

    ComputeSPMV(A, p, Ap); // Ap = A*p
    ComputeDotProduct(nrow, p, Ap, pAp, t4, A.isDotProductOptimized); // alpha = p'*Ap
    alpha = rtz/pAp;
    ComputeWAXPBY(nrow, 1.0, xOrdered, alpha, p, xOrdered, A.isWaxpbyOptimized); // x = x + alpha*p
    ComputeWAXPBY(nrow, 1.0, rOrdered, -alpha, Ap, rOrdered, A.isWaxpbyOptimized); // r = r - alpha*Ap
    ComputeDotProduct(nrow, rOrdered, rOrdered, normr, t4, A.isDotProductOptimized);
    normr = sqrt(normr);
#ifdef HPCG_DEBUG
    if (A.geom->rank==0 && (k%print_freq == 0 || k == max_iter))
      std::cout << "Iteration = "<< k << "   Scaled Residual = "<< normr/normr0 << std::endl;
#endif
    niters = k;
  }

  // Reorder back vectors
  CopyAndReorderVector(rOrdered, r, A.whichOldRowIsNewRow);
  CopyAndReorderVector(zOrdered, z, A.whichOldRowIsNewRow);
  CopyAndReorderVector(xOrdered, x, A.whichOldRowIsNewRow);
  DeleteVector(rOrdered);
  DeleteVector(zOrdered);
  DeleteVector(xOrdered);

  // Store times
  times[1] += t1; // dot-product time
  times[2] += t2; // WAXPBY time
  times[3] += t3; // SPMV time
  times[4] += t4; // AllReduce time
  times[5] += t5; // preconditioner apply time
  times[0] += 0;  // Total time. All done...
  return 0;
}
