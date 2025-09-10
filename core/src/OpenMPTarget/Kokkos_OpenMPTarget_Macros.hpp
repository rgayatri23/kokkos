//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_OPENMPTARGET_MACROS_HPP
#define KOKKOS_OPENMPTARGET_MACROS_HPP

// Define a macro that can be used to separate Kernel Mode extensions in llvm
// compiler from OpenMP standard directives. The extensions are only available
// from llvm compiler version greater than version 20.
#if (KOKKOS_COMPILER_CLANG >= 2000)
#define KOKKOS_IMPL_OPENMPTARGET_KERNEL_MODE
#endif

#define KOKKOS_IMPL_OPENMPTARGET_PRAGMA_HELPER(x) _Pragma(#x)
#define KOKKOS_IMPL_OMPTARGET_PRAGMA(x) \
  KOKKOS_IMPL_OPENMPTARGET_PRAGMA_HELPER(omp target x)

// Use scratch memory extensions to request dynamic shared memory for the
// right compiler/architecture combination.
#define KOKKOS_IMPL_OMPX_DYN_CGROUP_MEM(N) ompx_dyn_cgroup_mem(N)

#endif  // KOKKOS_OPENMPTARGET_MACROS_HPP
