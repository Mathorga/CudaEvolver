CCOMP			=gcc
CCOMP_FLAGS		=-std=c++11 -g
CLINK_FLAGS		=
CCOMP_LIBS		=-lstdc++

BIN_DIR			=./bin

RM				=rm -rf

all: default

default: Test

%.o : %.cpp
	$(CCOMP) $(CCOMP_FLAGS) -c $<

Test: Test.o Population.o PathGenome.o
	$(CCOMP) $(CLINK_FLAGS) $^ -o $(BIN_DIR)/$@ -lm $(CCOMP_LIBS)

clean:
	$(RM) $(BIN_DIR)/*
	$(RM) *.o
	$(RM) *.ppm
	$(RM) *.dat
	$(RM) *.log
