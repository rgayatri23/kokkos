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

#ifndef KOKKOS_OPENMPTARGET_PARALLEL_FOR_RANGE_HPP
#define KOKKOS_OPENMPTARGET_PARALLEL_FOR_RANGE_HPP

#include <omp.h>
#include <sstream>
#include <Kokkos_Parallel.hpp>
#include "Kokkos_OpenMPTarget_Instance.hpp"
#include "Kokkos_OpenMPTarget_FunctorAdapter.hpp"
#if defined(KOKKOS_IMPL_OPENMPTARGET_KERNEL_MODE)
#include <ompx.h>
#endif

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
                  Kokkos::Experimental::OpenMPTarget> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;
  using Member = typename Policy::member_type;

  Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> m_functor;
  const Policy m_policy;

 public:
  void execute() const { execute_impl(); }

  void execute_impl() const {
    Experimental::Impl::OpenMPTargetInternal::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    Experimental::Impl::OpenMPTargetInternal::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    const auto begin = m_policy.begin();
    const auto end   = m_policy.end();

    if (end <= begin) return;

    auto const a_functor(m_functor);

/*#if defined(KOKKOS_IMPL_OPENMPTARGET_KERNEL_MODE)*/
/*    const auto size      = end - begin;*/
/*    const auto team_size = 128;*/
/**/
/*    const int nTeams = size / team_size + !!(size % team_size);*/
/*#pragma omp target teams ompx_bare num_teams(nTeams, 1, 1) \*/
/*    thread_limit(team_size, 1, 1) firstprivate(a_functor)*/
/*    {*/
/*      const auto blockIdx  = ompx::block_id(ompx::dim_x);*/
/*      const auto blockDimx = ompx::block_dim(ompx::dim_x);*/
/*      const auto threadIdx = ompx::thread_id(ompx::dim_x);*/
/**/
/*      const auto i = blockIdx * blockDimx + threadIdx + begin;*/
/**/
/*      if (i < end) a_functor(i);*/
/*    }*/
/*#else*/
#pragma omp target teams distribute parallel for map(to : a_functor)
    for (auto i = begin; i < end; ++i) {
      a_functor(i);
    }
    /*#endif*/
  }

  ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif
