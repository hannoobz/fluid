CXX      = nvcc
CXXFLAGS = -std=c++17 -O3 --use_fast_math

GLFW_PREFIX := /usr
CXXFLAGS    += -I$(GLFW_PREFIX)/include
LDFLAGS     += -L$(GLFW_PREFIX)/lib

LIBS = -lGLEW -lglfw -lGL -lm

SRC_CU  := main.cu flip_fluid.cu
SRC_CPP := ui.cpp
OBJ     := $(SRC_CU:.cu=.o) $(SRC_CPP:.cpp=.o)
BIN     := flip-gpu

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cu flip_fluid.h ui.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.o: %.cpp ui.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all clean