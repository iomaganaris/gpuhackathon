all: bin/parallel_kernels_omp_cpu profiles/parallel_kernels_omp_cpu.qdrep bin/parallel_kernels_omp profiles/parallel_kernels_omp.qdrep #bin/parallel_cufft_example_omp profiles/parallel_cufft_example_omp.qdrep #bin/parallel_kernels_acc profiles/parallel_kernels_acc.qdrep #bin/openmp_example profiles/openmp_example.qdrep

bin: bin/parallel_kernels_cpu_gcc bin/parallel_kernels_omp bin/parallel_kernels_omp_cpu bin/parallel_kernels_omp_cpu_gcc
bin_nvhpc: bin/parallel_kernels_omp bin/parallel_kernels_omp_cpu
profile_nvhpc: profiles/parallel_kernels_omp.qdrep
bin/parallel_kernels_omp: src/parallel_kernels.cpp
	mkdir -p bin/
	nvc++ -DNVHPC_COMPILER -cuda -fast -mp=gpu -Minfo=accel,mp -gpu=lineinfo -o $@ $^

bin/parallel_kernels_omp_cpu: src/parallel_kernels.cpp
	mkdir -p bin/
	nvc++ -DNVHPC_COMPILER -cuda -mp=multicore -Minfo=mp -o $@ $^

bin/parallel_kernels_cpu_gcc: src/parallel_kernels.cpp
	mkdir -p bin/
	g++ -o $@ $^

bin/parallel_kernels_omp_cpu_gcc: src/parallel_kernels.cpp
	mkdir -p bin/
	g++ -fopenmp -o $@ $^

bin/parallel_cufft_example_omp: /gpfs/bbp.cscs.ch/apps/hpc/singularity/install/linux-rhel7-x86_64/gcc-9.3.0/nvhpc-21.11-qhk3q2/Linux_x86_64/21.11/examples/CUDA-Libraries/cuFFT/test_fft_omp_cpp/tcufft2dompc5.cpp
	mkdir -p bin/
	nvc++ -DNVHPC_COMPILER -cudalib=cufft -cuda -mp=multicore,gpu -Minfo=accel,mp -gpu=lineinfo -o $@ $^

bin/parallel_kernels_acc: src/parallel_kernels.cpp 
	mkdir -p bin/
	nvc++ -DNVHPC_COMPILER -cuda -acc -Minfo=accel -gpu=lineinfo -o $@ $^

# bin/openmp_example: src/openmp_example.cpp
# 	mkdir -p bin/
# 	nvc++ -DNVHPC_COMPILER -cuda -Minfo=accel,mp -acc -mp=gpu -o $@ $^

profiles/%.qdrep: bin/%
	mkdir -p profiles/
	nsys profile -f true -o $@ -c cudaProfilerApi --kill=none $^

.PHONY: clean
clean:
	rm -rf profiles bin
