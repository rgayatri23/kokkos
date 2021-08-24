
/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <TestDefaultDeviceType_Category.hpp>

namespace Test {

template <typename T, class DeviceType>
class TestViewAPI {
 public:
  using device = DeviceType;

  enum { N0 = 1000, N1 = 3, N2 = 5, N3 = 7 };

  using dView0       = Kokkos::View<T, device>;
  using dView1       = Kokkos::View<T *, device>;
  using dView2       = Kokkos::View<T * [N1], device>;
  using dView3       = Kokkos::View<T * [N1][N2], device>;
  using dView4       = Kokkos::View<T * [N1][N2][N3], device>;
  using const_dView4 = Kokkos::View<const T * [N1][N2][N3], device>;
  using dView4_unmanaged =
      Kokkos::View<T ****, device, Kokkos::MemoryUnmanaged>;
  using host = typename dView0::host_mirror_space;

  using DataType = T[2];


  static void run_test_error() {
    // Issue with giving a number to `alloc_size` using numeric_limits for size_t type.
    // The first one works. The second one fails the test and the third one seg faults. 
    auto alloc_size = std::numeric_limits<size_t>::max()/2 + 1;
//    auto alloc_size = std::numeric_limits<size_t>::max()/2 - 1;
//    auto alloc_size = std::numeric_limits<size_t>::max()/2 - 42;

    try {
      auto should_always_fail = dView1("hello_world_failure", alloc_size);
    } catch (std::runtime_error const &error) {
      // TODO once we remove the conversion to std::runtime_error, catch the
      //      appropriate Kokkos error here
      std::string msg = error.what();
      ASSERT_PRED_FORMAT2(::testing::IsSubstring, "hello_world_failure", msg);
      ASSERT_PRED_FORMAT2(::testing::IsSubstring,
                          typename device::memory_space{}.name(), msg);
      // Can't figure out how to make assertions either/or, so we'll just use
      // an if statement here for now.  Test failure message will be a bit
      // misleading, but developers should figure out what's going on pretty
      // quickly.
      if (msg.find("is not a valid size") != std::string::npos) {
        ASSERT_PRED_FORMAT2(::testing::IsSubstring, "is not a valid size", msg);
      } else
      {
        ASSERT_PRED_FORMAT2(::testing::IsSubstring, "insufficient memory", msg);
      }
      // SYCL cannot tell the reason why a memory allocation failed
    }
  }
};

TEST(TEST_CATEGORY, view_allocation_error) {
  TestViewAPI<double, TEST_EXECSPACE>::run_test_error();
}

}  // namespace Test
