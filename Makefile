CCOMP		=gcc
CCOMP_FLAGS	=-Wall -Wpedantic
CCOMP_LIBS	=-lstdc++
GA_INC_DIR	=./lib/galib247/include
GA_LIB_DIR	=./lib/galib247

BIN_DIR		=./bin

.cpp.o:
	$(CCOMP) $(CCOMP_FLAGS) $(GA_INC_DIR) -c $<

all: test

test: test.o
	$(CCOMP) $@.o -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(CCOMP_LIBS)

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf *.o
	rm -rf *.dat
