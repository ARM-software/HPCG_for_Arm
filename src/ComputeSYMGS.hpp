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

#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeSYMGS( const SparseMatrix  & A, const Vector & r, Vector & x );
int ComputeFusedSYMGS_SPMV( const SparseMatrix  & A, const Vector & r, Vector & x, Vector & y );
#ifdef HPCG_USE_NEON
int ComputeFusedSYMGS_SPMV_NEON (const SparseMatrix & A, const Vector & r, Vector & x, Vector & y);
#endif
#ifdef HPCG_USE_SVE
int ComputeFusedSYMGS_SPMV_SVE ( const SparseMatrix & A, const Vector & r, Vector & x, Vector & y);
#endif

#endif // COMPUTESYMGS_HPP
