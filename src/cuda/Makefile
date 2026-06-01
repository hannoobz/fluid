CXX      ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -ffast-math -Wall -Wextra

# GLFW (via Homebrew on macOS)
GLFW_PREFIX := $(shell brew --prefix glfw 2>/dev/null || echo /opt/homebrew/opt/glfw)
CXXFLAGS    += -I$(GLFW_PREFIX)/include
LDFLAGS     += -L$(GLFW_PREFIX)/lib

# macOS frameworks + GLFW
# LIBS := -lglfw -framework OpenGL -framework Cocoa -framework IOKit

LIBS = -lGLEW -lglfw -lGL -lm

SRC := flip_fluid.cpp ui.cpp main.cpp
OBJ := $(SRC:.cpp=.o)
BIN := flip

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp flip_fluid.h ui.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all clean
