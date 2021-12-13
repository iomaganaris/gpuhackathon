#include <cuda_profiler_api.h>

#include <memory>
#include <vector>

#define stringify(x) #x
#ifdef _OPENMP
#define omp_pragma(x) _Pragma(stringify(omp x))
#define acc_pragma(x)
#elif defined(_OPENACC)
#define omp_pragma(x)
#define acc_pragma(x) _Pragma(stringify(acc x))
#else
#error "Expected OpenMP or OpenACC to be enabled."
#endif

template<typename T>
void init(int* data, std::size_t data_size, std::vector<T>& streams, int queue = 0) {
  acc_pragma(parallel loop present(data[0:data_size]))
  omp_pragma(target teams distribute parallel for simd depend(out: streams[queue]) nowait)
  for(auto i = 0; i < data_size; ++i) {
    data[i] = i;
  }
}

template<typename T>
void compute(int* data, std::size_t data_start, std::size_t data_size, std::vector<T>& streams, int queue) {
  acc_pragma(parallel loop present(data[data_start:data_size]) async(streams[queue]))
  omp_pragma(target teams distribute parallel for simd depend(inout: streams[queue]) nowait)
  for(auto i = data_start; i < data_start + data_size; ++i) {
    for(auto j = 0; j < i*i; ++j) {
      data[i] += j;
    }
  }
}

int main() {
  cudaProfilerStart();
  // Choose a size small enough for there to be space for the two compute kernels on the device at the same time.
  auto const data_size = 10000;
  auto* data = new int[data_size];
  auto const nstreams = 2;
  std::vector<int> streams;
  streams.reserve(nstreams);
  streams[0] = 0;
  streams[1] = 1;
  acc_pragma(data create(data[0:data_size]))
  omp_pragma(target data map(alloc: data[0:data_size]))
  {
    init(data, data_size, streams);
    // Launch 2 kernels that can be run in parallel
    compute(data, 0, data_size/2, streams, 0);
    compute(data, data_size/2, data_size/2, streams, 1);
    acc_pragma(wait async(0))
    acc_pragma(wait async(1))
    omp_pragma(taskwait)
  }
  cudaProfilerStop();
  return 0;
}
