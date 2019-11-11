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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#include <cassert>
#ifdef HPCG_USE_DDOT_ARMPL
#include "armpl.h"
#endif
#ifdef HPCG_USE_SVE
#include "arm_sve.h"
#endif

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

	assert(x.localLength >= n);
	assert(y.localLength >= n);

	double *xv = x.values;
	double *yv = y.values;
	double local_result = 0.0;

#ifdef HPCG_USE_SVE
	if ( xv == yv ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += svcntd()) {
			svbool_t pg = svwhilelt_b64(i, n);
			svfloat64_t svx = svld1_f64(pg, &xv[i]);

            svfloat64_t svlr = svmul_f64_z(pg, svx, svx);

            local_result += svaddv_f64(svptrue_b64(), svlr);
		}
	} else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += svcntd()) {
			svbool_t pg = svwhilelt_b64_u64(i, n);
			svfloat64_t svx = svld1_f64(pg, &xv[i]);
			svfloat64_t svy = svld1_f64(pg, &yv[i]);

            svfloat64_t svlr = svmul_f64_z(pg, svx, svy);
            
            local_result += svaddv_f64(svptrue_b64(), svlr);
		}
	}
#elif defined HPCG_USE_DDOT_ARMPL
	local_result = cblas_ddot(n, xv, 1, yv, 1);
#else //HPCG_USE_DDOT_ARMPL
	if ( yv == xv ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i++ ) {
            local_result += xv[i] * xv[i];
        }
	} else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i++ ) local_result += xv[i] * yv[i];
	}
#endif //HPCG_USE_DDOT_ARMPL

#ifndef HPCG_NO_MPI
	// Use MPI's reduce function to collect all partial sums
	double t0 = mytimer();
	double global_result = 0.0;
	MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	result = global_result;
	time_allreduce += mytimer() - t0;
#else //HPCG_NO_MPI
	time_allreduce += 0.0;
	result = local_result;
#endif //HPCG_NO_MPI

	return 0;
}
