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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include "ComputeRestriction.hpp"
#include "ComputeProlongation.hpp"

int ComputeMG_TDG(const SparseMatrix  & A, const Vector & r, Vector & x) {
	int ierr = 0;
	if ( A.mgData != 0 ) {
		int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
		for (int i = 0; i < numberOfPresmootherSteps-1; i++ ) {
			ierr += ComputeSYMGS(A, r, x);
		}
#ifdef HPCG_USE_FUSED_SYMGS_SPMV

		// Fuse the last SYMGS iteration with the following SPMV
		// HPCG rules forbid that, so the result will be invalid
		// and therefore not submiteable

#ifdef HPCG_USE_SVE
		ierr += ComputeFusedSYMGS_SPMV_SVE(A, r, x, *A.mgData->Axf);
#elif defined HPCG_USE_NEON
		ierr += ComputeFusedSYMGS_SPMV_NEON(A, r, x, *A.mgData->Axf);
#else
		ierr += ComputeFusedSYMGS_SPMV(A, r, x, *A.mgData->Axf);
#endif
		if ( ierr != 0 ) return ierr;

#else // if !HPCG_USE_FUSED_SYMGS_SPMV
		ierr += ComputeSYMGS(A, r, x);
		if ( ierr != 0 ) return ierr;

		ierr = ComputeSPMV(A, x, *A.mgData->Axf);
		if ( ierr != 0 ) return ierr;
#endif // HPCG_USE_FUSED_SYMGS_SPMV

		ierr = ComputeRestriction(A, r);
		if ( ierr != 0 ) return ierr;
		
		ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc);
		if ( ierr != 0 ) return ierr;

		ierr = ComputeProlongation(A, x);
		if ( ierr != 0 ) return ierr;

		int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
		for ( int i = 0; i < numberOfPostsmootherSteps; i++ ) {
			ierr += ComputeSYMGS(A, r, x);
		}
		if ( ierr != 0 ) return ierr;

	} else {
		ierr = ComputeSYMGS(A, r, x);
		if ( ierr != 0 ) return ierr;
	}
	return 0;
}

int ComputeMG_BLOCK(const SparseMatrix  & A, const Vector & r, Vector & x) {
	int ierr = 0;
	if ( A.mgData != 0 ) {
		int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
		for (int i = 0; i < numberOfPresmootherSteps; i++ ) {
			ierr += ComputeSYMGS(A, r, x);
		}
		if ( ierr != 0 ) return ierr;

		ierr = ComputeSPMV(A, x, *A.mgData->Axf);
		if ( ierr != 0 ) return ierr;

		ierr = ComputeRestriction(A, r);
		if ( ierr != 0 ) return ierr;
		
		ierr = ComputeMG(*A.Ac, *A.mgData->rc, *A.mgData->xc);
		if ( ierr != 0 ) return ierr;

		ierr = ComputeProlongation(A, x);
		if ( ierr != 0 ) return ierr;

		int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
		for ( int i = 0; i < numberOfPostsmootherSteps; i++ ) {
			ierr += ComputeSYMGS(A, r, x);
		}
		if ( ierr != 0 ) return ierr;

	} else {
		ierr = ComputeSYMGS(A, r, x);
		if ( ierr != 0 ) return ierr;
	}
	return 0;
}


/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {
	assert(x.localLength == A.localNumberOfColumns);

	ZeroVector(x);

	if ( A.TDG ) {
		return ComputeMG_TDG(A, r, x);
	}
	return ComputeMG_BLOCK(A, r, x);

}
