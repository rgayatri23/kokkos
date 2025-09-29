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

#ifndef KOKKOS_OPENMPTARGET_PARALLEL_FOR_TEAM_HPP
#define KOKKOS_OPENMPTARGET_PARALLEL_FOR_TEAM_HPP

#include <omp.h>
#include <ompx.h>
#include <sstream>
#include <OpenMPTarget/Kokkos_OpenMPTarget_Macros.hpp>
#include <Kokkos_Parallel.hpp>
#include <OpenMPTarget/Kokkos_OpenMPTarget_Parallel.hpp>
#include <OpenMPTarget/Kokkos_OpenMPTarget_FunctorAdapter.hpp>

namespace Kokkos {

/** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team.
 */
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_boundaries,
    const Lambda& lambda) {
#if defined(KOKKOS_IMPL_OPENMPTARGET_KERNEL_MODE)
  const int blockDimy = ompx::block_dim(ompx::dim_y);
  const int threadIdy = ompx::thread_id(ompx::dim_y);

  for (iType i = loop_boundaries.start + threadIdy; i < loop_boundaries.end;
       i += blockDimy)
    lambda(i);
#else
#pragma omp for nowait schedule(static, 1)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) lambda(i);
#endif
}

/** \brief  Intra-thread vector parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
 */
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_boundaries,
    const Lambda& lambda) {
#if defined(KOKKOS_IMPL_OPENMPTARGET_KERNEL_MODE)
  const int blockDimx = ompx::block_dim(ompx::dim_x);
  const int threadIdx = ompx::thread_id(ompx::dim_x);
  for (iType i = loop_boundaries.start + threadIdx; i < loop_boundaries.end;
       i += blockDimx)
    lambda(i);
#else
#pragma omp simd
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) lambda(i);
#endif
}

/** \brief  Intra-team vector parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling team.
 */
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::OpenMPTargetExecTeamMember>& loop_boundaries,
    const Lambda& lambda) {
#if defined(KOKKOS_IMPL_OPENMPTARGET_KERNEL_MODE)
  const int blockDimx = ompx::block_dim(ompx::dim_x);
  const int threadIdx = ompx::thread_id(ompx::dim_x);

  for (iType i = loop_boundaries.start + threadIdx; i < loop_boundaries.end;
       i += blockDimx)
    lambda(i);
#else
#pragma omp for simd nowait schedule(static, 1)
  for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) lambda(i);
#endif
}

namespace Impl {

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Experimental::OpenMPTarget> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::OpenMPTarget,
                                       Properties...>;
  using Member = typename Policy::member_type;

  Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> m_functor;

  const Policy m_policy;
  const size_t m_shmem_size;

 public:
  void execute() const {
    Experimental::Impl::OpenMPTargetInternal::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    Experimental::Impl::OpenMPTargetInternal::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    execute_impl();
  }

 private:
  void execute_impl() const {
    Experimental::Impl::OpenMPTargetInternal::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    Experimental::Impl::OpenMPTargetInternal::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    const auto league_size   = m_policy.league_size();
    const auto team_size     = m_policy.team_size();
    const auto vector_length = m_policy.impl_vector_length();

    size_t shmem_size_L0       = m_policy.scratch_size(0, team_size);
    const size_t shmem_size_L1 = m_policy.scratch_size(1, team_size);
    m_policy.space().impl_internal_space_instance()->resize_scratch(
        team_size, shmem_size_L0, shmem_size_L1, league_size);

    void* scratch_ptr =
        m_policy.space().impl_internal_space_instance()->get_scratch_ptr();
    auto const a_functor(m_functor);

    // Maximum active teams possible.
    int max_active_teams = omp_get_max_teams();

    // If the league size is <=0, do not launch the kernel.
    if (max_active_teams <= 0) return;

#if defined(KOKKOS_IMPL_OPENMPTARGET_KERNEL_MODE)
    shmem_size_L0 += Impl::OpenMPTargetExecTeamMember::TEAM_REDUCE_SIZE;
    KOKKOS_IMPL_OMPTARGET_PRAGMA(
        teams ompx_bare num_teams(max_active_teams, 1, 1) thread_limit(
            vector_length, team_size, 1) firstprivate(a_functor, scratch_ptr)
            KOKKOS_IMPL_OMPX_DYN_CGROUP_MEM(shmem_size_L0)) {
      const auto blockIdx  = ompx::block_id(ompx::dim_x);
      const auto gridDimx  = ompx::grid_dim(ompx::dim_x);
      const auto blockDimy = ompx::block_dim(ompx::dim_y);
      const auto blockDimx = ompx::block_dim(ompx::dim_x);

      for (int league_id = blockIdx; league_id < league_size;
           league_id += gridDimx) {
        typename Policy::member_type team(league_id, league_size, blockDimy,
                                          blockDimx, scratch_ptr, blockIdx, shmem_size_L0,
                                          shmem_size_L1);
        a_functor(team);
      }
    }
#else
    KOKKOS_IMPL_OMPTARGET_PRAGMA(
        teams thread_limit(team_size) firstprivate(a_functor)
            num_teams(max_active_teams) is_device_ptr(scratch_ptr)
                KOKKOS_IMPL_OMPX_DYN_CGROUP_MEM(shmem_size_L0))
#pragma omp parallel
    {
      if (omp_get_num_teams() > max_active_teams)
        Kokkos::abort("`omp_set_num_teams` call was not respected.\n");

      const int blockIdx = omp_get_team_num();
      const int gridDim  = omp_get_num_teams();

      // Iterate through the number of teams until league_size and assign the
      // league_id accordingly
      for (int league_id = blockIdx; league_id < league_size;
           league_id += gridDim) {
        typename Policy::member_type team(league_id, league_size, team_size,
                                          vector_length, scratch_ptr,
                                          shmem_size_L0, shmem_size_L1);
        a_functor(team);
      }
    }
#endif
  }

 public:
  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_shmem_size(m_policy.scratch_size(0) + m_policy.scratch_size(1) +
                     FunctorTeamShmemSize<FunctorType>::value(
                         arg_functor, m_policy.team_size())) {}
};

}  // namespace Impl
}  // namespace Kokkos

#endif
