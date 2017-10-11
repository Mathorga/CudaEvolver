CCOMP			=gcc
CCOMP_FLAGS		=-std=c++98 -Wall -Wpedantic
CCOMP_LIBS		=-lstdc++
C=-dc -dlink -arch=sm_35
NVCOMP			=nvcc
NVCOMP_FLAGS	=-std=c++11 --compiler-options -Wall
NVCOMP_LIBS		=-lstdc++

GA_INC_DIR		=./lib/galib247/include
GA_LIB_DIR		=./lib/galib247

BIN_DIR			=./bin

RM				=rm -rf

all: PathTest

%.o : %.cu
	$(NVCOMP) $(NVCOMP_FLAGS) -c $<

PathTest: PathTest.o PathGenome.o
	$(NVCOMP) $@.o PathGenome.o -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(NVCOMP_LIBS)

clean:
	$(RM) $(BIN_DIR)/*
	$(RM) *.o
	$(RM) *.ppm
	$(RM) *.dat

exec: test
	./bin/test
