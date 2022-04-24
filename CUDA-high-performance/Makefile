NVCOMP			=nvcc

ifdef CUDA_ARCH
CUDA_ARCH_FLAG=-arch=$(CUDA_ARCH)
else
CUDA_ARCH_FLAG=
endif

NVCOMP_FLAGS	=-std=c++11 $(CUDA_ARCH_FLAG) -g -G -use_fast_math
NVLINK_FLAGS	=$(CUDA_ARCH_FLAG)
NVCOMP_LIBS		=-lstdc++

BIN_DIR			=./bin

RM				=rm -rf

all: default

default: Test

%.o : %.cu
	$(NVCOMP) $(NVCOMP_FLAGS) -c $<

Test: Test.o
	$(NVCOMP) $(NVLINK_FLAGS) $^ -o $(BIN_DIR)/$@ -lm $(NVCOMP_LIBS)

clean:
	$(RM) $(BIN_DIR)/*
	$(RM) *.o
	$(RM) *.ppm
	$(RM) *.dat
	$(RM) *.log
	$(RM) *.out
