CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native -mavx -pthread -ffast-math
LDFLAGS = -pthread -lstdc++fs

SRC = groundtruth.cpp
TARGET = groundtruth

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
