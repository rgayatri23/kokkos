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

#ifndef KOKKOS_OPENMPTARGET_PARALLELFOR_MDRANGE_HPP
#define KOKKOS_OPENMPTARGET_PARALLELFOR_MDRANGE_HPP

#include <omp.h>
#include <Kokkos_Parallel.hpp>
#include "Kokkos_OpenMPTarget_MDRangePolicy.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Experimental::OpenMPTarget> {
 private:
  using Policy  = Kokkos::MDRangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  using Member  = typename Policy::member_type;
  using Index   = typename Policy::index_type;

  const FunctorType m_functor;
  const Policy m_policy;

 public:
  inline void execute() const {
    OpenMPTargetExec::verify_is_process(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    OpenMPTargetExec::verify_initialized(
        "Kokkos::Experimental::OpenMPTarget parallel_for");
    FunctorType functor(m_functor);
    Policy policy = m_policy;

    typename Policy::point_type unused;
    static_assert(1 < Policy::rank && Policy::rank < 7);
    static_assert(Policy::inner_direction == Iterate::Left ||
                  Policy::inner_direction == Iterate::Right);

    execute_tile<Policy::rank>(
        unused, functor, policy,
        std::integral_constant<Iterate, Policy::inner_direction>());
  }

  template <int Rank>
  inline std::enable_if_t<Rank == 2> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateRight) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];

#pragma omp target teams distribute parallel for collapse(2) map(to : functor)
    for (auto i0 = begin_0; i0 < end_0; ++i0)
      for (auto i1 = begin_1; i1 < end_1; ++i1) {
        if constexpr (std::is_void<typename Policy::work_tag>::value)
          functor(i0, i1);
        else
          functor(typename Policy::work_tag(), i0, i1);
      }
  }

  template <int Rank>
  inline std::enable_if_t<Rank == 3> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateRight) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];
    const Index begin_2 = policy.m_lower[2];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];
    const Index end_2 = policy.m_upper[2];

#pragma omp target teams distribute parallel for collapse(3) map(to : functor)
    for (auto i0 = begin_0; i0 < end_0; ++i0) {
      for (auto i1 = begin_1; i1 < end_1; ++i1) {
        for (auto i2 = begin_2; i2 < end_2; ++i2) {
          if constexpr (std::is_void<typename Policy::work_tag>::value)
            functor(i0, i1, i2);
          else
            functor(typename Policy::work_tag(), i0, i1, i2);
        }
      }
    }
  }

  template <int Rank>
  inline std::enable_if_t<Rank == 4> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateRight) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];
    const Index begin_2 = policy.m_lower[2];
    const Index begin_3 = policy.m_lower[3];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];
    const Index end_2 = policy.m_upper[2];
    const Index end_3 = policy.m_upper[3];

#pragma omp target teams distribute parallel for collapse(4) map(to : functor)
    for (auto i0 = begin_0; i0 < end_0; ++i0) {
      for (auto i1 = begin_1; i1 < end_1; ++i1) {
        for (auto i2 = begin_2; i2 < end_2; ++i2) {
          for (auto i3 = begin_3; i3 < end_3; ++i3) {
            if constexpr (std::is_void<typename Policy::work_tag>::value)
              functor(i0, i1, i2, i3);
            else
              functor(typename Policy::work_tag(), i0, i1, i2, i3);
          }
        }
      }
    }
  }

  template <int Rank>
  inline std::enable_if_t<Rank == 5> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateRight) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];
    const Index begin_2 = policy.m_lower[2];
    const Index begin_3 = policy.m_lower[3];
    const Index begin_4 = policy.m_lower[4];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];
    const Index end_2 = policy.m_upper[2];
    const Index end_3 = policy.m_upper[3];
    const Index end_4 = policy.m_upper[4];

#pragma omp target teams distribute parallel for collapse(5) map(to : functor)
    for (auto i0 = begin_0; i0 < end_0; ++i0) {
      for (auto i1 = begin_1; i1 < end_1; ++i1) {
        for (auto i2 = begin_2; i2 < end_2; ++i2) {
          for (auto i3 = begin_3; i3 < end_3; ++i3) {
            for (auto i4 = begin_4; i4 < end_4; ++i4) {
              if constexpr (std::is_same<typename Policy::work_tag,
                                         void>::value)
                functor(i0, i1, i2, i3, i4);
              else
                functor(typename Policy::work_tag(), i0, i1, i2, i3, i4);
            }
          }
        }
      }
    }
  }

  template <int Rank>
  inline std::enable_if_t<Rank == 6> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateRight) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];
    const Index begin_2 = policy.m_lower[2];
    const Index begin_3 = policy.m_lower[3];
    const Index begin_4 = policy.m_lower[4];
    const Index begin_5 = policy.m_lower[5];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];
    const Index end_2 = policy.m_upper[2];
    const Index end_3 = policy.m_upper[3];
    const Index end_4 = policy.m_upper[4];
    const Index end_5 = policy.m_upper[5];

#pragma omp target teams distribute parallel for collapse(6) map(to : functor)
    for (auto i0 = begin_0; i0 < end_0; ++i0) {
      for (auto i1 = begin_1; i1 < end_1; ++i1) {
        for (auto i2 = begin_2; i2 < end_2; ++i2) {
          for (auto i3 = begin_3; i3 < end_3; ++i3) {
            for (auto i4 = begin_4; i4 < end_4; ++i4) {
              for (auto i5 = begin_5; i5 < end_5; ++i5) {
                {
                  if constexpr (std::is_same<typename Policy::work_tag,
                                             void>::value)
                    functor(i0, i1, i2, i3, i4, i5);
                  else
                    functor(typename Policy::work_tag(), i0, i1, i2, i3, i4,
                            i5);
                }
              }
            }
          }
        }
      }
    }
  }

  template <int Rank>
  inline std::enable_if_t<Rank == 2> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateLeft) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];

#pragma omp target teams distribute parallel for collapse(2) map(to : functor)
    for (auto i1 = begin_1; i1 < end_1; ++i1)
      for (auto i0 = begin_0; i0 < end_0; ++i0) {
        if constexpr (std::is_void<typename Policy::work_tag>::value)
          functor(i0, i1);
        else
          functor(typename Policy::work_tag(), i0, i1);
      }
  }

  template <int Rank>
  inline std::enable_if_t<Rank == 3> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateLeft) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];
    const Index begin_2 = policy.m_lower[2];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];
    const Index end_2 = policy.m_upper[2];


#if defined(KOKKOS_IMPL_OPENMPTARGET_KERNEL_MODE)
    const Index tot = (end_2-begin_2) * (end_1-begin_1) * (end_0-begin_0);

    auto tot_inner = (end_1 - begin_1) * (end_0 - begin_0);
    auto tot_outer = (end_2 - begin_2) * tot_inner;

//#pragma omp target teams distribute parallel for map(to : functor)
    //for (auto iter2 = 0; iter2 < tot_outer; ++iter2) {
//      if(omp_get_team_num() == 0 && omp_get_num_threads() == 0)
//      printf("num_teams = %d, team_size = %d, thread)id = %d\n", omp_get_num_teams(), omp_get_num_threads(), omp_get_num_threads());

    constexpr const int team_size = 1;
    const int num_teams  = (tot_outer + team_size - 1) / team_size * team_size;

      printf("tot_outer = %d\n", tot_outer);

    for (auto tmp = 0; tmp < tot_outer; ++tmp) {
#pragma omp target teams ompx_bare thread_limit(1) num_teams(1) map(to:functor) firstprivate(tot_outer)
{
  const Index blockDimx = ompx::block_dim(ompx::dim_x);
  const Index blockIdx  = ompx::block_id(ompx::dim_x);
  const Index threadIdx  = ompx::thread_id(ompx::dim_x);

  auto iter2 = tmp; //+ blockDimx * blockIdx + threadIdx;
  if (iter2 < tot_outer) {
          auto i2 = iter2 / tot_inner;
          auto iter = iter2 % tot_inner;

          auto i1 = iter / (end_0 - begin_0);
          auto i0 = iter % (end_0 - begin_0);

//        printf("(i0,i1,i2) = (%d,%d,%d)\n", i0,i1,i2);
//        printf("blockIdx = %d, iter2 = %d\n", blockIdx, iter2);

          if constexpr (std::is_void<typename Policy::work_tag>::value)
            functor(i0, i1, i2);
          else
            functor(typename Policy::work_tag(), i0, i1, i2);
      }
      }
}
#else
#pragma omp target teams distribute parallel for collapse(3) map(to : functor)
    for (auto i2 = begin_2; i2 < end_2; ++i2) {
      for (auto i1 = begin_1; i1 < end_1; ++i1) {
        for (auto i0 = begin_0; i0 < end_0; ++i0) {
          if constexpr (std::is_void<typename Policy::work_tag>::value)
            functor(i0, i1, i2);
          else
            functor(typename Policy::work_tag(), i0, i1, i2);
        }
      }
    }
#endif

  }

  template <int Rank>
  inline std::enable_if_t<Rank == 4> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateLeft) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];
    const Index begin_2 = policy.m_lower[2];
    const Index begin_3 = policy.m_lower[3];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];
    const Index end_2 = policy.m_upper[2];
    const Index end_3 = policy.m_upper[3];

#pragma omp target teams distribute parallel for collapse(4) map(to : functor)
    for (auto i3 = begin_3; i3 < end_3; ++i3) {
      for (auto i2 = begin_2; i2 < end_2; ++i2) {
        for (auto i1 = begin_1; i1 < end_1; ++i1) {
          for (auto i0 = begin_0; i0 < end_0; ++i0) {
            if constexpr (std::is_void<typename Policy::work_tag>::value)
              functor(i0, i1, i2, i3);
            else
              functor(typename Policy::work_tag(), i0, i1, i2, i3);
          }
        }
      }
    }
  }

  template <int Rank>
  inline std::enable_if_t<Rank == 5> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateLeft) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];
    const Index begin_2 = policy.m_lower[2];
    const Index begin_3 = policy.m_lower[3];
    const Index begin_4 = policy.m_lower[4];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];
    const Index end_2 = policy.m_upper[2];
    const Index end_3 = policy.m_upper[3];
    const Index end_4 = policy.m_upper[4];

#pragma omp target teams distribute parallel for collapse(5) map(to : functor)
    for (auto i4 = begin_4; i4 < end_4; ++i4) {
      for (auto i3 = begin_3; i3 < end_3; ++i3) {
        for (auto i2 = begin_2; i2 < end_2; ++i2) {
          for (auto i1 = begin_1; i1 < end_1; ++i1) {
            for (auto i0 = begin_0; i0 < end_0; ++i0) {
              if constexpr (std::is_same<typename Policy::work_tag,
                                         void>::value)
                functor(i0, i1, i2, i3, i4);
              else
                functor(typename Policy::work_tag(), i0, i1, i2, i3, i4);
            }
          }
        }
      }
    }
  }

  template <int Rank>
  inline std::enable_if_t<Rank == 6> execute_tile(
      typename Policy::point_type offset, const FunctorType& functor,
      const Policy& policy, OpenMPTargetIterateLeft) const {
    (void)offset;
    const Index begin_0 = policy.m_lower[0];
    const Index begin_1 = policy.m_lower[1];
    const Index begin_2 = policy.m_lower[2];
    const Index begin_3 = policy.m_lower[3];
    const Index begin_4 = policy.m_lower[4];
    const Index begin_5 = policy.m_lower[5];

    const Index end_0 = policy.m_upper[0];
    const Index end_1 = policy.m_upper[1];
    const Index end_2 = policy.m_upper[2];
    const Index end_3 = policy.m_upper[3];
    const Index end_4 = policy.m_upper[4];
    const Index end_5 = policy.m_upper[5];

#pragma omp target teams distribute parallel for collapse(6) map(to : functor)
    for (auto i5 = begin_5; i5 < end_5; ++i5) {
      for (auto i4 = begin_4; i4 < end_4; ++i4) {
        for (auto i3 = begin_3; i3 < end_3; ++i3) {
          for (auto i2 = begin_2; i2 < end_2; ++i2) {
            for (auto i1 = begin_1; i1 < end_1; ++i1) {
              for (auto i0 = begin_0; i0 < end_0; ++i0) {
                {
                  if constexpr (std::is_same<typename Policy::work_tag,
                                             void>::value)
                    functor(i0, i1, i2, i3, i4, i5);
                  else
                    functor(typename Policy::work_tag(), i0, i1, i2, i3, i4,
                            i5);
                }
              }
            }
          }
        }
      }
    }
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
  // TODO DZP: based on a conversation with Christian, we're using 256 as a
  // heuristic here. We need something better once we can query these kinds of
  // properties
  template <typename Policy, typename Functor>
  static int max_tile_size_product(const Policy&, const Functor&) {
    return 256;
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif /* KOKKOS_OPENMPTARGET_PARALLELFOR_MDRANGE_HPP */
