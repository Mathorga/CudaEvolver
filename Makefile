CXX         =gcc
CXX_FLAGS	=-Wall -Wpedantic
CXX_LIBS	=-lstdc++
GA_INC_DIR	=./lib/galib247/include
GA_LIB_DIR	=./lib/galib247

BIN_DIR		=./bin

.cpp.o:
	gcc $(CXX_FLAGS) -c $<

all: test

test: test.o
	$(CXX) $@.o -o $(BIN_DIR)/$@ -L$(GA_LIB_DIR) -lga -lm $(CXX_LIBS)

clean:
	rm -rf $(BIN_DIR)/*
	rm -rf *.o
	rm -rf *.dat
