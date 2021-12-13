#ifdef NVHPC_COMPILER
#include <cuda_profiler_api.h>
#endif
#include <iostream>
#include <memory>
#include <omp.h>

#define stringify(x) #x
#ifdef _OPENMP
#define omp_pragma(x) _Pragma(stringify(omp x))
#define acc_pragma(x)
#elif defined(_OPENACC)
#define omp_pragma(x)
#define acc_pragma(x) _Pragma(stringify(acc x))
#else
#define omp_pragma(x)
#define acc_pragma(x)
#endif

void init(int* data, std::size_t data_size) {
  acc_pragma(parallel loop present(data[0:data_size]))
  omp_pragma(target teams distribute parallel for simd depend(out:data[0:data_size]) nowait)
  for(auto i = 0; i < data_size; ++i) {
    data[i] = i;
  }
}

template<typename T>
void compute(int* data, std::size_t data_start, std::size_t data_size, std::array< T, 2 > streams, int index) {
  std::cout << "[compute] : data_start: " << data_start << " data_size: " << data_size << std::endl;
  acc_pragma(parallel loop present(data[data_start:data_size]) async(streams[index]))
  omp_pragma(target teams distribute parallel for simd depend(inout:streams[index]) nowait)
  for(auto i = data_start; i < data_start + data_size; ++i) {
    for(auto j = 0; j < i*i; ++j) {
      data[i] += j;
    }
    // data[i] *= i;
  }
}

int main() {
#ifdef NVHPC_COMPILER
  cudaProfilerStart();
#endif
  // Choose a size small enough for there to be space for the two compute kernels on the device at the same time.
  auto const data_size = 10000;
  auto* data = new int[data_size];
  std::array<int,2> streams;
  streams[0] = 0;
  streams[1] = 1;
  acc_pragma(data create(data[0:data_size]))
  omp_pragma(target data map(tofrom: data[0:data_size]))
  {
    init(data, data_size);
    // Launch 2 kernels that can be run in parallel
    compute(data, 0, data_size/2, streams, 0);
    compute(data, data_size/2, data_size/2, streams, 1);
    acc_pragma(wait async(0))
    acc_pragma(wait async(1))
    omp_pragma(taskwait)
  }
  for(int i = 0; i < data_size; i++) {
    std::cout << data[i] << std::endl;
  }
#ifdef NVHPC_COMPILER
  cudaProfilerStop();
#endif
  return 0;
}
