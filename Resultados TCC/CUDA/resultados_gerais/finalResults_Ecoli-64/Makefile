NVCC = nvcc -ccbin g++-13
NVCCFLAGS = -O3 -rdc=true -w -Xcompiler -Wall

MAXOUTVALUE = 1
MAXNODESVALUE = 500
MAXEVAL = 150000

main:
	${NVCC} ${NVCCFLAGS} -DMAX_OUTPUTS=${MAXOUTVALUE} -DMAX_NODES=${MAXNODESVALUE} -DNUM_GENERATIONS=${MAXEVAL} -x cu source/CUDAConfig.cpp source/main.cpp source/cgp.cpp source/utils.cpp source/GPTime.cpp source/stack.cpp source/kernels/kernel.cu -o progCUDA
