CCOMP		=gcc
CCOMP_FLAGS	=-std=c++98 -Wall -Wpedantic
CCOMP_LIBS	=-lstdc++

NVCOMP		=nvcc
NVCOMP_FLAGS=
NVCOMP_LIBS	=-lstdc++

GA_INC_DIR	=./lib/galib247/include
GA_LIB_DIR	=./lib/galib247

BIN_DIR		=./bin

RM			=rm -rf

all: PathTest

.cpp.o:
	$(CCOMP) $(CCOMP_FLAGS) -c $<

CUDATest.o:
	$(NVCOMP) $(NVCOMP_FLAGS) -c CUDATest.cu

PathTest.o:
	$(NVCOMP) $(NVCOMP_FLAGS) -c PathTest.cu

test: test.o
	$(CCOMP) $@.o -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(CCOMP_LIBS)

CUDATest: CUDATest.o
	$(NVCOMP) $@.o -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(NVCOMP_LIBS)

PathTest: PathTest.o
	$(NVCOMP) $@.o -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(NVCOMP_LIBS)

clean:
	$(RM) $(BIN_DIR)/*
	$(RM) *.o
	$(RM) *.ppm
	$(RM) *.dat

exec: test
	./bin/test
