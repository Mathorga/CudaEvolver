CCOMP			=gcc
CCOMP_FLAGS		=-std=c++98 -Wall -Wpedantic
CCOMP_LIBS		=-lstdc++
NVCOMP			=nvcc
NVCOMP_FLAGS	=-std=c++11 -arch=sm_35 -dc -g -G
NVLINK_FLAGS	=-arch=sm_35 -rdc=true
NVCOMP_LIBS		=-lstdc++

GA_INC_DIR		=./lib/galib247/include
GA_LIB_DIR		=./lib/galib247

BIN_DIR			=./bin

RM				=rm -rf

all: default #deepcopy

default: Test

old: PathTest

deepcopy: PathTestDeepCopy

%.o : %.cu
	$(NVCOMP) $(NVCOMP_FLAGS) -c $<

Test: Test.o try.o CUDAPopulation.o CUDAPathGenome.o
	$(NVCOMP) $(NVLINK_FLAGS) $^ -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(NVCOMP_LIBS)

PathTest: PathTest.o PathGenome.o try.o CUDAPopulation.o CUDAPathGenome.o
	$(NVCOMP) $(NVLINK_FLAGS) $^ -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(NVCOMP_LIBS)

PathTestDeepCopy: PathTest.o PathGenomeDeepCopy.o
	$(NVCOMP) $(NVLINK_FLAGS) $^ -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(NVCOMP_LIBS)

clean:
	$(RM) $(BIN_DIR)/*
	$(RM) *.o
	$(RM) *.ppm
	$(RM) *.dat
