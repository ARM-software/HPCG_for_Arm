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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"
#include <cassert>

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#ifdef HPCG_USE_WAXPBY_ARMPL
#include "armpl.h"
#endif

#ifdef HPCG_USE_SVE
#include "arm_sve.h"
#endif


/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

	assert(x.localLength >= n);
	assert(y.localLength >= n);
	
	const double * const xv = x.values;
	const double * const yv = y.values;
	double * const wv = w.values;

#ifdef HPCG_USE_SVE
	if ( alpha == 1.0 && beta == 1.0 ) {
		// w[i] = xv[i] + yv[i]
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < n; i += svcntd() ) {
			svbool_t pg = svwhilelt_b64(i, n);

			svfloat64_t svx = svld1(pg, &xv[i]);
			svfloat64_t svy = svld1(pg, &yv[i]);

			svfloat64_t svw = svadd_f64_z(pg, svx, svy);

			svst1_f64(pg, &wv[i], svw);
		}
	} else if ( alpha == 1.0 ) {
		// w[i] = xv[i] + beta*yv[i]
		svfloat64_t svb = svdup_f64(beta);
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < n; i += svcntd() ) {
			svbool_t pg = svwhilelt_b64(i, n);

			svfloat64_t svx = svld1(pg, &xv[i]);
			svfloat64_t svy = svld1(pg, &yv[i]);

			svfloat64_t svw = svmla_f64_z(pg, svx, svb, svy);

			svst1_f64(pg, &wv[i], svw);
		}
	} else if ( beta == 1.0 ) {
		// w[i] = alpha*xv[i] + yv[i]
		svfloat64_t sva = svdup_f64(alpha);
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < n; i += svcntd() ) {
			svbool_t pg = svwhilelt_b64(i, n);

			svfloat64_t svx = svld1(pg, &xv[i]);
			svfloat64_t svy = svld1(pg, &yv[i]);

			svfloat64_t svw = svmad_f64_z(pg, svx, sva, svy);

			svst1_f64(pg, &wv[i], svw);
		}
	} else {
		// w[i] = alpha*xv[i] + beta*yv[i]
		svfloat64_t sva = svdup_f64(alpha);
		svfloat64_t svb = svdup_f64(beta);
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < n; i += svcntd() ) {
			svbool_t pg = svwhilelt_b64(i, n);

			svfloat64_t svx = svld1(pg, &xv[i]);
			svfloat64_t svy = svld1(pg, &yv[i]);

			svfloat64_t svax = svmul_f64_z(pg, svx, sva);

			svfloat64_t svw = svmad_f64_z(pg, svb, svy, svax);

			svst1_f64(pg, &wv[i], svw);
		}
	}


#else

#ifdef HPCG_USE_WAXPBY_ARMPL
#ifndef HPCG_NO_OPENMP
#pragma omp parallel default(shared)
	{
		local_int_t nthreads = omp_get_num_threads();
		local_int_t elemsPerThread = n / nthreads;
		local_int_t threadID = omp_get_thread_num();
		local_int_t firstElement = elemsPerThread * threadID;
		local_int_t lastElement = firstElement + elemsPerThread;
		if ( elemsPerThread * nthreads != n && threadID == nthreads-1 ) {
			lastElement = n;
		}

		BLAS_dwaxpby_x(lastElement-firstElement, alpha, &xv[firstElement], 1, beta, &yv[firstElement], 1, &wv[firstElement], 1, blas_prec_double);
	}
#else // HPCG_NO_OPENMP
	BLAS_dwaxpby_x(n, alpha, xv, 1, beta, yv, 1, wv, 1, blas_prec_double);
#endif // HPCG_NO_OPENMP

#else //HPCG_USE_WAXPBY_ARMPL
	if ( alpha == 1.0 && beta == 1.0 ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i++ ) {
			wv[i] = xv[i] + yv[i];
		}
	} else if ( alpha == 1.0 ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i++ ) {
			wv[i] = xv[i] + beta*yv[i];
		}
	} else if ( beta == 1.0 ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i++ ) {
			wv[i] = alpha*xv[i] + yv[i];
		}
	} else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i++ ) {
			wv[i] = alpha*xv[i] + beta*yv[i];
		}
	}
#endif // HPCG_USE_WAXPBY_ARMPL
#endif // HPCG_USE_SVE

	return 0;
}
