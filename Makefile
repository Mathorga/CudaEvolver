CCOMP			=gcc
CCOMP_FLAGS		=-std=c++98 -Wall -Wpedantic
CCOMP_LIBS		=-lstdc++
C=-dc -dlink -arch=sm_35 -g -G
NVCOMP			=nvcc
NVCOMP_FLAGS	=-std=c++11 -arch=sm_35 -g -G --compiler-options -Wall
NVCOMP_LIBS		=-lstdc++

GA_INC_DIR		=./lib/galib247/include
GA_LIB_DIR		=./lib/galib247

BIN_DIR			=./bin

RM				=rm -rf

all: default deepcopy

default: PathTest

deepcopy: PathTestDeepCopy

%.o : %.cu
	$(NVCOMP) $(NVCOMP_FLAGS) -c $<

PathTest: PathTest.o PathGenome.o
	$(NVCOMP) $^ -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(NVCOMP_LIBS)

PathTestDeepCopy: PathTest.o PathGenomeDeepCopy.o
	$(NVCOMP) $^ -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(NVCOMP_LIBS)

clean:
	$(RM) $(BIN_DIR)/*
	$(RM) *.o
	$(RM) *.ppm
	$(RM) *.dat

exec: test
	./bin/test
