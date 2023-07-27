hipsycl-omp:
	syclcc --hipsycl-targets=omp main.cpp
hipsycl-cuda:
	syclcc --hipsycl-targets=cuda-nvcxx main.cpp
dpcpp:
	dpcpp main.cpp
