all: bin/parallel_kernels_acc bin/parallel_kernels_omp bin/openmp_example profiles/parallel_kernels_acc.qdrep profiles/parallel_kernels_omp.qdrep profiles/openmp_example.qdrep

bin/parallel_kernels_omp: src/parallel_kernels.cpp
	mkdir -p bin/
	nvc++ -cuda -fast -mp=multicore,gpu -Minfo=accel,mp -gpu=lineinfo -o $@ $^ # without -fast this is crashing

bin/parallel_kernels_acc: src/parallel_kernels.cpp 
	mkdir -p bin/
	nvc++ -cuda -acc -Minfo=accel -gpu=lineinfo -o $@ $^

bin/openmp_example: src/openmp_example.cpp
	mkdir -p bin/
	nvc++ -cuda -Minfo=accel,mp -acc -mp=gpu -o $@ $^

profiles/%.qdrep: bin/%
	mkdir -p profiles/
	nsys profile -o $@ -c cudaProfilerApi --kill=none $^

.PHONY: clean
clean:
	rm -rf profiles bin
