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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
/* SVE IMPLEMENTATIONS                                                                            */
/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/

#ifdef HPCG_USE_SVE
#include "arm_sve.h"

/*
 * TDG VERSION
 */
int ComputeSYMGS_TDG_SVE(const SparseMatrix & A, const Vector & r, Vector & x) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	return 0;
}
/*
 * END OF TDG VERSION
 */

/*
 * TDG FUSED SYMGS-SPMV VERSION
 */
int ComputeFusedSYMGS_SPMV_SVE(const SparseMatrix & A, const Vector & r, Vector & x, Vector & y) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double **matrixDiagonal = A.matrixDiagonal;
	double * const yv = y.values;

	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = A.tdg[l].size(); i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			totalContribution -= xv[row] * currentDiagonal; // remove diagonal contribution
			double sum = rv[row] - totalContribution; // substract contributions from RHS
			xv[row] = sum / currentDiagonal; // update row

			// SPMV part
			totalContribution += xv[row] * currentDiagonal; // add updated diagonal contribution
			yv[row] = totalContribution; // update SPMV output vector
			
		}
	}

	return 0;
}
/*
 * END OF TDG FUSED SYMGS-SPMV VERSION
 */

/*
 * BLOCK COLORED VERSION
 */
int ComputeSYMGS_BLOCK_SVE(const SparseMatrix & A, const Vector & r, Vector & x ) {
	assert(x.localLength >= A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	double **matrixDiagonal = A.matrixDiagonal;
	const double * const rv = r.values;
	double * const xv = x.values;
	local_int_t firstBlock = 0;
	local_int_t lastBlock = firstBlock + A.numberOfBlocksInColor[0];
	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t color = 0; color < A.numberOfColors; color++ ) { // for each color
		if ( color > 0 ) {
			firstBlock += A.numberOfBlocksInColor[color-1];
			lastBlock = firstBlock + A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t block = firstBlock; block < lastBlock; block += A.chunkSize ) { // for each superblock with the same color
			local_int_t firstRow = block * A.blockSize;
			local_int_t firstChunk = firstRow / A.chunkSize;
			local_int_t lastChunk = (firstRow + A.blockSize * A.chunkSize) / A.chunkSize;

			for ( local_int_t chunk = firstChunk; chunk < lastChunk; chunk++ ) { // for each chunk of this super block
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;
				const int currentNumberOfNonzeros = A.nonzerosInChunk[chunk];
				local_int_t i = first;
				if ( A.chunkSize == 4 ) {
					const double * const currentValues0 = A.matrixValues[i  ];
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues2 = A.matrixValues[i+2];
					const double * const currentValues3 = A.matrixValues[i+3];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices2 = A.mtxIndL[i+2];
					const local_int_t * const currentColIndices3 = A.mtxIndL[i+3];

					const double currentDiagonal0 = matrixDiagonal[i  ][0];
					const double currentDiagonal1 = matrixDiagonal[i+1][0];
					const double currentDiagonal2 = matrixDiagonal[i+2][0];
					const double currentDiagonal3 = matrixDiagonal[i+3][0];

					svfloat64_t contribs0 = svdup_f64(0.0);
					svfloat64_t contribs1 = svdup_f64(0.0);
					svfloat64_t contribs2 = svdup_f64(0.0);
					svfloat64_t contribs3 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);
						svfloat64_t values1 = svld1_f64(pg, &currentValues1[j]);
						svfloat64_t values2 = svld1_f64(pg, &currentValues2[j]);
						svfloat64_t values3 = svld1_f64(pg, &currentValues3[j]);

						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);
						svuint64_t indices1 = svld1sw_u64(pg, &currentColIndices1[j]);
						svuint64_t indices2 = svld1sw_u64(pg, &currentColIndices2[j]);
						svuint64_t indices3 = svld1sw_u64(pg, &currentColIndices3[j]);

						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);
						svfloat64_t xvv1 = svld1_gather_u64index_f64(pg, xv, indices1);
						svfloat64_t xvv2 = svld1_gather_u64index_f64(pg, xv, indices2);
						svfloat64_t xvv3 = svld1_gather_u64index_f64(pg, xv, indices3);

						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0);
						contribs1 = svmla_f64_m(pg, contribs1, xvv1, values1);
						contribs2 = svmla_f64_m(pg, contribs2, xvv2, values2);
						contribs3 = svmla_f64_m(pg, contribs3, xvv3, values3);
					}

					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);
					double totalContribution1 = svaddv_f64(svptrue_b64(), contribs1);
					double totalContribution2 = svaddv_f64(svptrue_b64(), contribs2);
					double totalContribution3 = svaddv_f64(svptrue_b64(), contribs3);

					double sum0 = rv[i  ] - totalContribution0;
					double sum1 = rv[i+1] - totalContribution1;
					double sum2 = rv[i+2] - totalContribution2;
					double sum3 = rv[i+3] - totalContribution3;

					sum0 += xv[i  ] * currentDiagonal0;
					sum1 += xv[i+1] * currentDiagonal1;
					sum2 += xv[i+2] * currentDiagonal2;
					sum3 += xv[i+3] * currentDiagonal3;

					xv[i  ] = sum0 / currentDiagonal0;
					xv[i+1] = sum1 / currentDiagonal1;
					xv[i+2] = sum2 / currentDiagonal2;
					xv[i+3] = sum3 / currentDiagonal3;
				} else if ( A.chunkSize == 2 ) {
					const double * const currentValues0 = A.matrixValues[i  ];
					const double * const currentValues1 = A.matrixValues[i+1];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];

					const double currentDiagonal0 = matrixDiagonal[i  ][0];
					const double currentDiagonal1 = matrixDiagonal[i+1][0];

					svfloat64_t contribs0 = svdup_f64(0.0);
					svfloat64_t contribs1 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);
						svfloat64_t values1 = svld1_f64(pg, &currentValues1[j]);

						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);
						svuint64_t indices1 = svld1sw_u64(pg, &currentColIndices1[j]);

						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);
						svfloat64_t xvv1 = svld1_gather_u64index_f64(pg, xv, indices1);

						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0);
						contribs1 = svmla_f64_m(pg, contribs1, xvv1, values1);
					}

					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);
					double totalContribution1 = svaddv_f64(svptrue_b64(), contribs1);

					double sum0 = rv[i  ] - totalContribution0;
					double sum1 = rv[i+1] - totalContribution1;

					sum0 += xv[i  ] * currentDiagonal0;
					sum1 += xv[i+1] * currentDiagonal1;

					xv[i  ] = sum0 / currentDiagonal0;
					xv[i+1] = sum1 / currentDiagonal1;
				} else { //A.chunkSize == 1
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal0 = matrixDiagonal[i  ][0];

					svfloat64_t contribs0 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);

						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);

						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);

						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0);
					}

					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);

					double sum0 = rv[i  ] - totalContribution0;

					sum0 += xv[i  ] * currentDiagonal0;

					xv[i  ] = sum0 / currentDiagonal0;
				}
			}
		}
	}

	firstBlock = A.numberOfBlocks-1;
	lastBlock = firstBlock - A.numberOfBlocksInColor[A.numberOfColors-1];
	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t color = A.numberOfColors-1; color >= 0; color-- ) {
		if ( color < A.numberOfColors-1 ) {
			firstBlock -= A.numberOfBlocksInColor[color+1];
			lastBlock = firstBlock - A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t block = firstBlock; block > lastBlock; block -= A.chunkSize ) {
			local_int_t firstRow = ((block+1) * A.blockSize) - 1;
			local_int_t firstChunk = firstRow / A.chunkSize;
			local_int_t lastChunk = (firstRow - A.blockSize * A.chunkSize) / A.chunkSize;

			for ( local_int_t chunk = firstChunk; chunk > lastChunk; chunk-- ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				const int currentNumberOfNonzeros = A.nonzerosInChunk[chunk];
				local_int_t i = first;
				if ( A.chunkSize == 4 ) {
					const double * const currentValues3 = A.matrixValues[i+3];
					const double * const currentValues2 = A.matrixValues[i+2];
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices3 = A.mtxIndL[i+3];
					const local_int_t * const currentColIndices2 = A.mtxIndL[i+2];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal3 = matrixDiagonal[i+3][0];
					const double currentDiagonal2 = matrixDiagonal[i+2][0];
					const double currentDiagonal1 = matrixDiagonal[i+1][0];
					const double currentDiagonal0 = matrixDiagonal[i  ][0];

					svfloat64_t contribs3 = svdup_f64(0.0);
					svfloat64_t contribs2 = svdup_f64(0.0);
					svfloat64_t contribs1 = svdup_f64(0.0);
					svfloat64_t contribs0 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values3 = svld1_f64(pg, &currentValues3[j]);
						svfloat64_t values2 = svld1_f64(pg, &currentValues2[j]);
						svfloat64_t values1 = svld1_f64(pg, &currentValues1[j]);
						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);

						svuint64_t indices3 = svld1sw_u64(pg, &currentColIndices3[j]);
						svuint64_t indices2 = svld1sw_u64(pg, &currentColIndices2[j]);
						svuint64_t indices1 = svld1sw_u64(pg, &currentColIndices1[j]);
						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);

						svfloat64_t xvv3 = svld1_gather_u64index_f64(pg, xv, indices3);
						svfloat64_t xvv2 = svld1_gather_u64index_f64(pg, xv, indices2);
						svfloat64_t xvv1 = svld1_gather_u64index_f64(pg, xv, indices1);
						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);

						contribs3 = svmla_f64_m(pg, contribs3, xvv3, values3 );
						contribs2 = svmla_f64_m(pg, contribs2, xvv2, values2 );
						contribs1 = svmla_f64_m(pg, contribs1, xvv1, values1 );
						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0 );
					}

					double totalContribution3 = svaddv_f64(svptrue_b64(), contribs3);
					double totalContribution2 = svaddv_f64(svptrue_b64(), contribs2);
					double totalContribution1 = svaddv_f64(svptrue_b64(), contribs1);
					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);

					double sum3 = rv[i+3] - totalContribution3;
					double sum2 = rv[i+2] - totalContribution2;
					double sum1 = rv[i+1] - totalContribution1;
					double sum0 = rv[i  ] - totalContribution0;

					sum3 += xv[i+3] * currentDiagonal3;
					sum2 += xv[i+2] * currentDiagonal2;
					sum1 += xv[i+1] * currentDiagonal1;
					sum0 += xv[i  ] * currentDiagonal0;
					
					xv[i+3] = sum3 / currentDiagonal3;
					xv[i+2] = sum2 / currentDiagonal2;
					xv[i+1] = sum1 / currentDiagonal1;
					xv[i  ] = sum0 / currentDiagonal0;
				} else if ( A.chunkSize == 2 ) {
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal1 = matrixDiagonal[i+1][0];
					const double currentDiagonal0 = matrixDiagonal[i  ][0];

					svfloat64_t contribs1 = svdup_f64(0.0);
					svfloat64_t contribs0 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values1 = svld1_f64(pg, &currentValues1[j]);
						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);

						svuint64_t indices1 = svld1sw_u64(pg, &currentColIndices1[j]);
						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);

						svfloat64_t xvv1 = svld1_gather_u64index_f64(pg, xv, indices1);
						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);

						contribs1 = svmla_f64_m(pg, contribs1, xvv1, values1 );
						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0 );
					}

					double totalContribution1 = svaddv_f64(svptrue_b64(), contribs1);
					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);

					double sum1 = rv[i+1] - totalContribution1;
					double sum0 = rv[i  ] - totalContribution0;

					sum1 += xv[i+1] * currentDiagonal1;
					sum0 += xv[i  ] * currentDiagonal0;
					
					xv[i+1] = sum1 / currentDiagonal1;
					xv[i  ] = sum0 / currentDiagonal0;
				} else { // A.chunkSize == 1
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal0 = matrixDiagonal[i  ][0];

					svfloat64_t contribs0 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);

						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);

						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);

						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0 );
					}

					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);

					double sum0 = rv[i  ] - totalContribution0;

					sum0 += xv[i  ] * currentDiagonal0;
					
				}
			}
		}
	}

	return 0;
}
/*
 * END OF BLOCK COLORED VERSION
 */
#elif defined(HPCG_USE_NEON)

/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
/* NEON IMPLEMENTATIONS                                                                           */
/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/

#include "arm_neon.h"

/*
 * TDG VERSION
 */
int ComputeSYMGS_TDG_NEON(const SparseMatrix & A, const Vector & r, Vector & x) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			float64x2_t contribs = vdupq_n_f64(0.0);

			local_int_t j = 0;
			for ( j = 0; j < currentNumberOfNonzeros-1; j+=2 ) {
				// Load the needed j values
				float64x2_t mtxValues = vld1q_f64(&currentValues[j]);
				// Load the needed x values
				double aux[] = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };
				float64x2_t xvv = vld1q_f64(aux);
				// Add the contribution
				contribs = vfmaq_f64(contribs, mtxValues, xvv);
			}
			// reduce contributions
			double totalContribution = vgetq_lane_f64(contribs, 0) + vgetq_lane_f64(contribs, 1);
			double sum = rv[row] - totalContribution;
			// Add missing values from last loop
			if ( j < currentNumberOfNonzeros ) {
				sum -= currentValues[j] * xv[currentColIndices[j]];
			}
			sum += xv[row] * currentDiagonal; // remove diagonal contribution
			xv[row] = sum / currentDiagonal; // update row
		}
	}

	/*
	 * BACKWARD
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			float64x2_t contribs = vdupq_n_f64(0.0);

			local_int_t j = 0;
			for ( j = 0; j < currentNumberOfNonzeros-1; j+=2 ) {
				// Load the needed j values
				float64x2_t mtxValues = vld1q_f64(&currentValues[j]);
				// Load the needed x values
				double aux[] = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };
				float64x2_t xvv = vld1q_f64(aux);
				// Add the contribution
				contribs = vfmaq_f64(contribs, mtxValues, xvv);
			}
			// reduce contributions
			double totalContribution = vgetq_lane_f64(contribs, 0) + vgetq_lane_f64(contribs, 1);
			double sum = rv[row] - totalContribution;
			// Add missing values from last loop
			if ( j < currentNumberOfNonzeros ) {
				sum -= currentValues[j] * xv[currentColIndices[j]];
			}
			sum += xv[row] * currentDiagonal; // remove diagonal contribution
			xv[row] = sum / currentDiagonal; // update row
		}
	}

	return 0;
}
/*
 *
 */
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/*
 * TDG FUSED VERSION
 */
int ComputeFusedSYMGS_SPMV_NEON(const SparseMatrix & A, const Vector & r, Vector & x, Vector & y) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double * const yv = y.values;
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			float64x2_t contribs = vdupq_n_f64(0.0);

			local_int_t j = 0;
			for ( j = 0; j < currentNumberOfNonzeros-1; j+=2 ) {
				// Load the needed j values
				float64x2_t mtxValues = vld1q_f64(&currentValues[j]);
				// Load the needed x values
				double aux[] = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };
				float64x2_t xvv = vld1q_f64(aux);
				// Add the contribution
				contribs = vfmaq_f64(contribs, mtxValues, xvv);
			}
			// reduce contributions
			double totalContribution = vgetq_lane_f64(contribs, 0) + vgetq_lane_f64(contribs, 1);
			double sum = rv[row] - totalContribution;
			// Add missing values from last loop
			if ( j < currentNumberOfNonzeros ) {
				sum -= currentValues[j] * xv[currentColIndices[j]];
			}
			sum += xv[row] * currentDiagonal; // remove diagonal contribution
			xv[row] = sum / currentDiagonal; // update row
		}
	}

	/*
	 * BACKWARD (fusing SYMGS and SPMV)
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			float64x2_t contribs = vdupq_n_f64(0.0);

			local_int_t j = 0;
			for ( j = 0; j < currentNumberOfNonzeros-1; j+=2 ) {
				// Load the needed j values
				float64x2_t mtxValues = vld1q_f64(&currentValues[j]);
				// Load the needed x values
				double aux[] = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };
				float64x2_t xvv = vld1q_f64(aux);
				// Add the contribution
				contribs = vfmaq_f64(contribs, mtxValues, xvv);
			}
			// reduce contributions
			double totalContribution = vgetq_lane_f64(contribs, 0) + vgetq_lane_f64(contribs, 1);
			// Add missing values from last loop
			if ( j < currentNumberOfNonzeros ) {
				totalContribution += currentValues[j] * xv[currentColIndices[j]];
			}
			totalContribution -= xv[row] * currentDiagonal; // remove diagonal contribution
			double sum = rv[row] - totalContribution; // substract contributions from RHS
			xv[row] = sum / currentDiagonal; // update row
			// Fusion part
			totalContribution += xv[row] * currentDiagonal; // add updated diagonal contribution
			yv[row] = totalContribution; // update SPMV output vector
		}
	}

	return 0;
}
/*
 *
 */
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/*
 * BLOCK COLORED VERSION
 */
int ComputeSYMGS_BLOCK_NEON(const SparseMatrix & A, const Vector & r, Vector & x) {

	assert(x.localLength >= A.localNumberOfColumns);
	
#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	double **matrixDiagonal = A.matrixDiagonal;
	const double * const rv = r.values;
	double * const xv = x.values;

	local_int_t firstBlock = 0;
	local_int_t lastBlock = firstBlock + A.numberOfBlocksInColor[0];
	/*
	 * FORWARD
	 */
	for ( local_int_t color = 0; color < A.numberOfColors; color++ ) { // for each color
		if ( color > 0 ) {
			firstBlock += A.numberOfBlocksInColor[color-1];
			lastBlock = firstBlock + A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t block = firstBlock; block < lastBlock; block += A.chunkSize ) { // for each super block with the same color
			local_int_t firstRow = block * A.blockSize;
			local_int_t firstChunk = firstRow / A.chunkSize;
			local_int_t lastChunk = (firstRow + A.blockSize*A.chunkSize) / A.chunkSize;

			for ( local_int_t chunk = firstChunk; chunk < lastChunk; chunk++ ) { // for each chunk of this super block
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				const int currentNumberOfNonzeros = A.nonzerosInChunk[chunk];
				local_int_t i = first;
				if ( A.chunkSize == 4 ) {
					const double * const currentValues0 = A.matrixValues[i  ];
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues2 = A.matrixValues[i+2];
					const double * const currentValues3 = A.matrixValues[i+3];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices2 = A.mtxIndL[i+2];
					const local_int_t * const currentColIndices3 = A.mtxIndL[i+3];

					const double currentDiagonal[4] = { matrixDiagonal[i  ][0],\
														matrixDiagonal[i+1][0],\
														matrixDiagonal[i+2][0],\
														matrixDiagonal[i+3][0]};
					float64x2_t diagonal01 = vld1q_f64(&currentDiagonal[0]);
					float64x2_t diagonal23 = vld1q_f64(&currentDiagonal[2]);

					float64x2_t contribs0 = vdupq_n_f64(0.0);
					float64x2_t contribs1 = vdupq_n_f64(0.0);
					float64x2_t contribs2 = vdupq_n_f64(0.0);
					float64x2_t contribs3 = vdupq_n_f64(0.0);

					float64x2_t vrv01 = vld1q_f64(&rv[i]);
					float64x2_t vrv23 = vld1q_f64(&rv[i+2]);

					float64x2_t vxv01 = vld1q_f64(&xv[i]);
					float64x2_t vxv23 = vld1q_f64(&xv[i+2]);

					local_int_t j = 0;
					for ( j = 0; j < currentNumberOfNonzeros-1; j += 2 ) {
						// Load values
						float64x2_t values0 = vld1q_f64(&currentValues0[j]);
						float64x2_t values1 = vld1q_f64(&currentValues1[j]);
						float64x2_t values2 = vld1q_f64(&currentValues2[j]);
						float64x2_t values3 = vld1q_f64(&currentValues3[j]);

						// Load x
						float64x2_t vxv0 = { xv[currentColIndices0[j]], xv[currentColIndices0[j+1]] };
						float64x2_t vxv1 = { xv[currentColIndices1[j]], xv[currentColIndices1[j+1]] };
						float64x2_t vxv2 = { xv[currentColIndices2[j]], xv[currentColIndices2[j+1]] };
						float64x2_t vxv3 = { xv[currentColIndices3[j]], xv[currentColIndices3[j+1]] };

						// Add contribution
						contribs0 = vfmaq_f64(contribs0, values0, vxv0);
						contribs1 = vfmaq_f64(contribs1, values1, vxv1);
						contribs2 = vfmaq_f64(contribs2, values2, vxv2);
						contribs3 = vfmaq_f64(contribs3, values3, vxv3);
					}
					// Reduce contribution
					// First for i and i+1
					float64x2_t totalContribution01;
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs0), totalContribution01, 0);
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs1), totalContribution01, 1);

					// Then for i+2 and i+3
					float64x2_t totalContribution23;
					totalContribution23 = vsetq_lane_f64(vaddvq_f64(contribs2), totalContribution23, 0);
					totalContribution23 = vsetq_lane_f64(vaddvq_f64(contribs3), totalContribution23, 1);

					// Substract contributions from RHS
					float64x2_t sum01 = vsubq_f64(vrv01, totalContribution01);
					float64x2_t sum23 = vsubq_f64(vrv23, totalContribution23);

					// Add contributions from missing elements (if any)
					if ( j < currentNumberOfNonzeros ) {
						// Load current values
						float64x2_t values01 = { currentValues0[j], currentValues1[j] };
						float64x2_t values23 = { currentValues2[j], currentValues3[j] };

						// Load x
						float64x2_t vx01 = { xv[currentColIndices0[j]], xv[currentColIndices1[j]] };
						float64x2_t vx23 = { xv[currentColIndices2[j]], xv[currentColIndices3[j]] };

						// Add contributions
						sum01 = vfmsq_f64(sum01, values01, vx01);
						sum23 = vfmsq_f64(sum23, values23, vx23);
					}

					// Remove diagonal contribution and update rows i and i+1
					sum01 = vfmaq_f64(sum01, vxv01, diagonal01);
					xv[i  ] = vgetq_lane_f64(sum01, 0) / currentDiagonal[0];
					xv[i+1] = vgetq_lane_f64(sum01, 1) / currentDiagonal[1];

					// Remove diagonal contribution and update rows i+2 and i+3
					sum23 = vfmaq_f64(sum23, vxv23, diagonal23);
					xv[i+2] = vgetq_lane_f64(sum23, 0) / currentDiagonal[2];
					xv[i+3] = vgetq_lane_f64(sum23, 1) / currentDiagonal[3];
				} else if ( A.chunkSize == 2 ) {
					const double * const currentValues0 = A.matrixValues[i  ];
					const double * const currentValues1 = A.matrixValues[i+1];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];

					const double currentDiagonal[2] = { matrixDiagonal[i  ][0],\
														matrixDiagonal[i+1][0]};
					float64x2_t diagonal01 = vld1q_f64(&currentDiagonal[0]);

					float64x2_t contribs0 = vdupq_n_f64(0.0);
					float64x2_t contribs1 = vdupq_n_f64(0.0);

					float64x2_t vrv01 = vld1q_f64(&rv[i]);

					float64x2_t vxv01 = vld1q_f64(&xv[i]);

					local_int_t j = 0;
					for ( j = 0; j < currentNumberOfNonzeros-1; j += 2 ) {
						// Load values
						float64x2_t values0 = vld1q_f64(&currentValues0[j]);
						float64x2_t values1 = vld1q_f64(&currentValues1[j]);

						// Load x
						float64x2_t vxv0 = { xv[currentColIndices0[j]], xv[currentColIndices0[j+1]] };
						float64x2_t vxv1 = { xv[currentColIndices1[j]], xv[currentColIndices1[j+1]] };

						// Add contribution
						contribs0 = vfmaq_f64(contribs0, values0, vxv0);
						contribs1 = vfmaq_f64(contribs1, values1, vxv1);
					}
					// Reduce contribution
					// First for i and i+1
					float64x2_t totalContribution01;
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs0), totalContribution01, 0);
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs1), totalContribution01, 1);

					// Substract contributions from RHS
					float64x2_t sum01 = vsubq_f64(vrv01, totalContribution01);

					// Add contributions from missing elements (if any)
					if ( j < currentNumberOfNonzeros ) {
						// Load current values
						float64x2_t values01 = { currentValues0[j], currentValues1[j] };

						// Load x
						float64x2_t vx01 = { xv[currentColIndices0[j]], xv[currentColIndices1[j]] };

						// Add contributions
						sum01 = vfmsq_f64(sum01, values01, vx01);
					}

					// Remove diagonal contribution and update rows i and i+1
					sum01 = vfmaq_f64(sum01, vxv01, diagonal01);
					xv[i  ] = vgetq_lane_f64(sum01, 0) / currentDiagonal[0];
					xv[i+1] = vgetq_lane_f64(sum01, 1) / currentDiagonal[1];
				} else { // A.chunkSize == 1
					const double * const currentValues = A.matrixValues[i];
					const local_int_t * const currentColIndices = A.mtxIndL[i];
					const double currentDiagonal = matrixDiagonal[i][0];
					float64x2_t contribs = vdupq_n_f64(0.0);

					local_int_t j = 0;
					for ( j = 0; j < currentNumberOfNonzeros-1; j += 2 ) {
						// Load values
						float64x2_t values = vld1q_f64(&currentValues[j]);

						// Load x
						float64x2_t vxv = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };

						// Add contribution
						contribs = vfmaq_f64(contribs, values, vxv);
					}
					// Reduce contribution
					// First for i and i+1
					double totalContribution;
					totalContribution = vaddvq_f64(contribs);

					// Substract contributions from RHS
					double sum = rv[i] - totalContribution;

					// Add contributions from missing elements (if any)
					if ( j < currentNumberOfNonzeros ) {
						sum -= currentValues[j] * xv[currentColIndices[j]];
					}

					// Remove diagonal contribution and update rows i and i+1
					sum += xv[i] * currentDiagonal;
					xv[i] = sum / currentDiagonal;
				}
			}
		}
	}

	firstBlock = A.numberOfBlocks-1;
	lastBlock = firstBlock - A.numberOfBlocksInColor[A.numberOfColors-1];
	/*
	 * BACKWARD
	 */
	for ( local_int_t color = A.numberOfColors-1; color >= 0; color-- ) {
		if ( color < A.numberOfColors-1 ) {
			firstBlock -= A.numberOfBlocksInColor[color+1];
			lastBlock = firstBlock - A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t block = firstBlock; block > lastBlock; block -= A.chunkSize ) { // we skip a whole superblock on each iteration
			local_int_t firstRow = ((block+1) * A.blockSize) - 1; // this is the last row of the last block (i.e., next block first row - 1)
			local_int_t firstChunk = firstRow / A.chunkSize; // this is the  chunk of the row above
			local_int_t lastChunk = (firstRow - A.blockSize*A.chunkSize) / A.chunkSize; 

			for ( local_int_t chunk = firstChunk; chunk > lastChunk; chunk-- ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				const int currentNumberOfNonzeros = A.nonzerosInChunk[chunk];
				if ( A.chunkSize == 4 ) {
					local_int_t i = last-1-3;

					const double * const currentValues3 = A.matrixValues[i+3];
					const double * const currentValues2 = A.matrixValues[i+2];
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices3 = A.mtxIndL[i+3];
					const local_int_t * const currentColIndices2 = A.mtxIndL[i+2];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal[4] = {\
							matrixDiagonal[i  ][0],\
							matrixDiagonal[i+1][0],\
							matrixDiagonal[i+2][0],\
							matrixDiagonal[i+3][0]};

					float64x2_t diagonal01 = vld1q_f64(&currentDiagonal[0]);
					float64x2_t diagonal23 = vld1q_f64(&currentDiagonal[2]);

					float64x2_t contribs0 = vdupq_n_f64(0.0);
					float64x2_t contribs1 = vdupq_n_f64(0.0);
					float64x2_t contribs2 = vdupq_n_f64(0.0);
					float64x2_t contribs3 = vdupq_n_f64(0.0);

					float64x2_t vrv23 = vld1q_f64(&rv[i+2]);
					float64x2_t vrv01 = vld1q_f64(&rv[i  ]);

					float64x2_t vxv23 = vld1q_f64(&xv[i+2]);
					float64x2_t vxv01 = vld1q_f64(&xv[i  ]);

					local_int_t j = 0;
					for ( j = currentNumberOfNonzeros-2; j >= 0; j -= 2 ) {
						// Load values
						float64x2_t values0 = vld1q_f64(&currentValues0[j]);
						float64x2_t values1 = vld1q_f64(&currentValues1[j]);
						float64x2_t values2 = vld1q_f64(&currentValues2[j]);
						float64x2_t values3 = vld1q_f64(&currentValues3[j]);

						// Load x
						float64x2_t vxv0 = { xv[currentColIndices0[j]], xv[currentColIndices0[j+1]] };
						float64x2_t vxv1 = { xv[currentColIndices1[j]], xv[currentColIndices1[j+1]] };
						float64x2_t vxv2 = { xv[currentColIndices2[j]], xv[currentColIndices2[j+1]] };
						float64x2_t vxv3 = { xv[currentColIndices3[j]], xv[currentColIndices3[j+1]] };

						// Add contribution
						contribs0 = vfmaq_f64(contribs0, values0, vxv0);
						contribs1 = vfmaq_f64(contribs1, values1, vxv1);
						contribs2 = vfmaq_f64(contribs2, values2, vxv2);
						contribs3 = vfmaq_f64(contribs3, values3, vxv3);
					}
					// Reduce contribution
					// First for i and i-1
					float64x2_t totalContribution01;
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs0), totalContribution01, 0);
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs1), totalContribution01, 1);

					// Then for i-2 and i-3
					float64x2_t totalContribution23;
					totalContribution23 = vsetq_lane_f64(vaddvq_f64(contribs2), totalContribution23, 0);
					totalContribution23 = vsetq_lane_f64(vaddvq_f64(contribs3), totalContribution23, 1);

					// Substract contributions from RHS
					float64x2_t sum23 = vsubq_f64(vrv23, totalContribution23);
					float64x2_t sum01 = vsubq_f64(vrv01, totalContribution01);

					// Add contributions from missing elements (if any)
					if ( j == -1 ) {
						// Load current values
						float64x2_t values23 = { currentValues2[j+1], currentValues3[j+1] };
						float64x2_t values01 = { currentValues0[j+1], currentValues1[j+1] };

						// Load x
						float64x2_t vx23 = { xv[currentColIndices2[j+1]], xv[currentColIndices3[j+1]] };
						float64x2_t vx01 = { xv[currentColIndices0[j+1]], xv[currentColIndices1[j+1]] };

						// Add contributions
						sum23 = vfmsq_f64(sum23, values23, vx23);
						sum01 = vfmsq_f64(sum01, values01, vx01);
					}

					// Remove diagonal contribution and update rows i-2 and i-3
					sum23 = vfmaq_f64(sum23, vxv23, diagonal23);
					xv[i+3] = vgetq_lane_f64(sum23, 1) / currentDiagonal[3];
					xv[i+2] = vgetq_lane_f64(sum23, 0) / currentDiagonal[2];

					// Remove diagonal contribution and update rows i and i-1
					sum01 = vfmaq_f64(sum01, vxv01, diagonal01);
					xv[i+1] = vgetq_lane_f64(sum01, 1) / currentDiagonal[1];
					xv[i  ] = vgetq_lane_f64(sum01, 0) / currentDiagonal[0];
				} else if ( A.chunkSize == 2 ) {
					local_int_t i = last-1-1;

					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal[2] = {\
							matrixDiagonal[i  ][0],\
							matrixDiagonal[i+1][0]};

					float64x2_t diagonal01 = vld1q_f64(&currentDiagonal[0]);

					float64x2_t contribs0 = vdupq_n_f64(0.0);
					float64x2_t contribs1 = vdupq_n_f64(0.0);

					float64x2_t vrv01 = vld1q_f64(&rv[i  ]);

					float64x2_t vxv01 = vld1q_f64(&xv[i  ]);

					local_int_t j = 0;
					for ( j = currentNumberOfNonzeros-2; j >= 0; j -= 2 ) {
						// Load values
						float64x2_t values0 = vld1q_f64(&currentValues0[j]);
						float64x2_t values1 = vld1q_f64(&currentValues1[j]);

						// Load x
						float64x2_t vxv0 = { xv[currentColIndices0[j]], xv[currentColIndices0[j+1]] };
						float64x2_t vxv1 = { xv[currentColIndices1[j]], xv[currentColIndices1[j+1]] };

						// Add contribution
						contribs0 = vfmaq_f64(contribs0, values0, vxv0);
						contribs1 = vfmaq_f64(contribs1, values1, vxv1);
					}
					// Reduce contribution
					// First for i and i-1
					float64x2_t totalContribution01;
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs0), totalContribution01, 0);
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs1), totalContribution01, 1);

					// Substract contributions from RHS
					float64x2_t sum01 = vsubq_f64(vrv01, totalContribution01);

					// Add contributions from missing elements (if any)
					if ( j == -1 ) {
						// Load current values
						float64x2_t values01 = { currentValues0[j+1], currentValues1[j+1] };

						// Load x
						float64x2_t vx01 = { xv[currentColIndices0[j+1]], xv[currentColIndices1[j+1]] };

						// Add contributions
						sum01 = vfmsq_f64(sum01, values01, vx01);
					}

					// Remove diagonal contribution and update rows i and i-1
					sum01 = vfmaq_f64(sum01, vxv01, diagonal01);
					xv[i+1] = vgetq_lane_f64(sum01, 1) / currentDiagonal[1];
					xv[i  ] = vgetq_lane_f64(sum01, 0) / currentDiagonal[0];
				} else { // A.chunkSize == 1
					local_int_t i = last - 1; // == first
					const double * const currentValues = A.matrixValues[i];
					const local_int_t * const currentColIndices = A.mtxIndL[i];
					const double currentDiagonal = matrixDiagonal[i][0];

					float64x2_t contribs = vdupq_n_f64(0.0);

					local_int_t j = 0;
					for ( j = 0; j < currentNumberOfNonzeros-1; j += 2 ) {
						// Load values
						float64x2_t values = vld1q_f64(&currentValues[j]);

						// Load x
						float64x2_t vxv = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };

						// Add contribution
						contribs = vfmaq_f64(contribs, values, vxv);
					}
					// Reduce contribution
					double totalContribution = vaddvq_f64(contribs);

					// Substract contribution from RHS
					double sum = rv[i] - totalContribution;

					// Add contributions from missing elements (if any)
					if ( j < currentNumberOfNonzeros ) {
						sum -= currentValues[j] * xv[currentColIndices[j]];
					}

					// Remove diagonal contribution and updated row i
					sum += xv[i] * currentDiagonal;
					xv[i] = sum / currentDiagonal;
				}
			}
		}
	}

	return 0;
}
/*
 *
 */
#endif
//#else // !HPCG_USE_SVE ! HPCG_USE_NEON

int ComputeFusedSYMGS_SPMV ( const SparseMatrix & A, const Vector & r, Vector & x, Vector & y ) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double * const yv = y.values;
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = rv[row];

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j++ ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	/*
	 * BACKWARD (fusing SYMGS and SPMV)
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = 0.0;

			for ( local_int_t j = currentNumberOfNonzeros-1; j >= 0; j-- ) {
				local_int_t curCol = currentColIndices[j];
				sum += currentValues[j] * xv[curCol];
			}
			sum -= xv[row] * currentDiagonal;
			xv[row] = (rv[row] - sum) / currentDiagonal;
			sum += xv[row] * currentDiagonal;
			yv[row] = sum;
		}
	}

	return 0;
}

int ComputeSYMGS_TDG ( const SparseMatrix & A, const Vector & r, Vector & x ) {

	assert( x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A,x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = rv[row];

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j++ ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	/*
	 * BACKWARD
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = rv[row];

			for ( local_int_t j = currentNumberOfNonzeros-1; j >= 0; j-- ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	return 0;
}

int ComputeSYMGS_BLOCK( const SparseMatrix & A, const Vector & r, Vector & x ) {

	assert(x.localLength >= A.localNumberOfColumns);
	
#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const local_int_t nrow = A.localNumberOfRows;
	double **matrixDiagonal = A.matrixDiagonal;
	const double * const rv = r.values;
	double * const xv = x.values;

	local_int_t firstBlock = 0;
	local_int_t lastBlock = firstBlock + A.numberOfBlocksInColor[0];
	/*
	 * FORWARD
	 */
	for ( local_int_t color = 0; color < A.numberOfColors; color++ ) {
		if ( color > 0 ) {
			firstBlock += A.numberOfBlocksInColor[color-1];
			lastBlock = firstBlock + A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t block = firstBlock; block < lastBlock; block += A.chunkSize ) {
			local_int_t firstRow = block * A.blockSize;
			local_int_t firstChunk = firstRow / A.chunkSize;
			local_int_t lastChunk = (firstRow + A.blockSize*A.chunkSize) / A.chunkSize;

			for ( local_int_t chunk = firstChunk; chunk < lastChunk; chunk++ ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				//for ( local_int_t i = first; i < last; i+= (A.chunkSize/2)) {
				local_int_t i = first;
				if ( A.chunkSize == 4 ) {
					double sum0 = rv[i+0];
					double sum1 = rv[i+1];
					double sum2 = rv[i+2];
					double sum3 = rv[i+3];

					for ( local_int_t j = 0; j < A.nonzerosInChunk[chunk]; j++ ) {
						sum0 -= A.matrixValues[i+0][j] * xv[A.mtxIndL[i+0][j]];
						sum1 -= A.matrixValues[i+1][j] * xv[A.mtxIndL[i+1][j]];
						sum2 -= A.matrixValues[i+2][j] * xv[A.mtxIndL[i+2][j]];
						sum3 -= A.matrixValues[i+3][j] * xv[A.mtxIndL[i+3][j]];
					}
					sum0 += matrixDiagonal[i+0][0] * xv[i+0];
					xv[i+0] = sum0 / matrixDiagonal[i+0][0];
					sum1 += matrixDiagonal[i+1][0] * xv[i+1];
					xv[i+1] = sum1 / matrixDiagonal[i+1][0];
					sum2 += matrixDiagonal[i+2][0] * xv[i+2];
					xv[i+2] = sum2 / matrixDiagonal[i+2][0];
					sum3 += matrixDiagonal[i+3][0] * xv[i+3];
					xv[i+3] = sum3 / matrixDiagonal[i+3][0];
				} else if ( A.chunkSize == 2 ) {
					double sum0 = rv[i+0];
					double sum1 = rv[i+1];

					for ( local_int_t j = 0; j < A.nonzerosInChunk[chunk]; j++ ) {
						sum0 -= A.matrixValues[i+0][j] * xv[A.mtxIndL[i+0][j]];
						sum1 -= A.matrixValues[i+1][j] * xv[A.mtxIndL[i+1][j]];
					}
					sum0 += matrixDiagonal[i+0][0] * xv[i+0];
					xv[i+0] = sum0 / matrixDiagonal[i+0][0];
					sum1 += matrixDiagonal[i+1][0] * xv[i+1];
					xv[i+1] = sum1 / matrixDiagonal[i+1][0];
				} else { // A.chunkSize == 1
					double sum0 = rv[i+0];

					for ( local_int_t j = 0; j < A.nonzerosInChunk[chunk]; j++ ) {
						sum0 -= A.matrixValues[i+0][j] * xv[A.mtxIndL[i+0][j]];
					}
					sum0 += matrixDiagonal[i+0][0] * xv[i+0];
					xv[i+0] = sum0 / matrixDiagonal[i+0][0];
				}
			}
		}
	}

	firstBlock = A.numberOfBlocks-1;
	lastBlock = firstBlock - A.numberOfBlocksInColor[A.numberOfColors-1];
	/*
	 * BACKWARD
	 */
	for ( local_int_t color = A.numberOfColors-1; color >= 0; color-- ) {
		if ( color < A.numberOfColors-1 ) {
			firstBlock -= A.numberOfBlocksInColor[color+1];
			lastBlock = firstBlock - A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
		for ( local_int_t block = firstBlock; block > lastBlock; block -= A.chunkSize ) {
			local_int_t firstRow = ((block+1) * A.blockSize) - 1; // this is the last row of the last block
			local_int_t firstChunk = firstRow / A.chunkSize; // this is the  chunk of the row above
			local_int_t lastChunk = (firstRow - A.blockSize*A.chunkSize) / A.chunkSize; 

			for ( local_int_t chunk = firstChunk; chunk > lastChunk; chunk-- ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				//for ( local_int_t i = last-1; i >= first; i -= (A.chunkSize/2)) {
				local_int_t i = last-1;
				if ( A.chunkSize == 4 ) {
					double sum3 = rv[i-3];
					double sum2 = rv[i-2];
					double sum1 = rv[i-1];
					double sum0 = rv[i  ];

					for ( local_int_t j = A.nonzerosInChunk[chunk]-1; j >= 0; j-- ) {
						sum3 -= A.matrixValues[i-3][j] * xv[A.mtxIndL[i-3][j]];
						sum2 -= A.matrixValues[i-2][j] * xv[A.mtxIndL[i-2][j]];
						sum1 -= A.matrixValues[i-1][j] * xv[A.mtxIndL[i-1][j]];
						sum0 -= A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
					}
					sum3 += matrixDiagonal[i-3][0] * xv[i-3];
					xv[i-3] = sum3 / matrixDiagonal[i-3][0];

					sum2 += matrixDiagonal[i-2][0] * xv[i-2];
					xv[i-2] = sum2 / matrixDiagonal[i-2][0];

					sum1 += matrixDiagonal[i-1][0] * xv[i-1];
					xv[i-1] = sum1 / matrixDiagonal[i-1][0];

					sum0 += matrixDiagonal[i  ][0] * xv[i  ];
					xv[i  ] = sum0 / matrixDiagonal[i  ][0];
				} else if ( A.chunkSize == 2 ) {
					double sum1 = rv[i-1];
					double sum0 = rv[i  ];

					for ( local_int_t j = A.nonzerosInChunk[chunk]-1; j >= 0; j-- ) {
						sum1 -= A.matrixValues[i-1][j] * xv[A.mtxIndL[i-1][j]];
						sum0 -= A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
					}

					sum1 += matrixDiagonal[i-1][0] * xv[i-1];
					xv[i-1] = sum1 / matrixDiagonal[i-1][0];

					sum0 += matrixDiagonal[i  ][0] * xv[i  ];
					xv[i  ] = sum0 / matrixDiagonal[i  ][0];
				} else { // A.chunkSize == 1
					double sum0 = rv[i  ];

					for ( local_int_t j = A.nonzerosInChunk[chunk]-1; j >= 0; j-- ) {
						sum0 -= A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
					}

					sum0 += matrixDiagonal[i  ][0] * xv[i  ];
					xv[i  ] = sum0 / matrixDiagonal[i  ][0];
				}
			}
		}
	}

	return 0;
}
//#endif



/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

	// This function is just a stub right now which decides which implementation of the SYMGS will be executed (TDG or block coloring)
	if ( A.TDG ) {
#ifdef HPCG_USE_NEON
		return ComputeSYMGS_TDG_NEON(A, r, x);
#elif defined HPCG_USE_SVE
		return ComputeSYMGS_TDG_SVE(A, r, x);
#else
		return ComputeSYMGS_TDG(A, r, x);
#endif
	}
#ifdef HPCG_USE_NEON
	return ComputeSYMGS_BLOCK_NEON(A, r, x);
#elif defined HPCG_USE_SVE
	return ComputeSYMGS_BLOCK_SVE(A, r, x);
#else
	return ComputeSYMGS_BLOCK(A, r, x);
#endif
}
