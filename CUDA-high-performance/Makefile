NVCOMP			=nvcc
NVCOMP_FLAGS	=-std=c++11 -arch=sm_35 -g -G -use_fast_math
NVLINK_FLAGS	=-arch=sm_35
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
