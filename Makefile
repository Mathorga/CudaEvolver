CCOMP		=gcc
CCOMP_FLAGS	=-std=c++98 -Wall -Wpedantic
CCOMP_LIBS	=-lstdc++
GA_INC_DIR	=./lib/galib247/include
GA_LIB_DIR	=./lib/galib247

BIN_DIR		=./bin

RM			=rm -rf

.cpp.o:
	$(CCOMP) $(CCOMP_FLAGS) -c $<

all: test

test: test.o
	$(CCOMP) $@.o -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(CCOMP_LIBS)

clean:
	$(RM) $(BIN_DIR)/*
	$(RM) *.o
	$(RM) *.ppm
	$(RM) *.dat

exec: test
	./bin/test
