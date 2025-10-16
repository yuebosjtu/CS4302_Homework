# Makefile for Floyd-Warshall project
# Compatible with mingw32-make

# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++11 -fopenmp
INCLUDES = -I1_Floyd/include

# Directories
SRC_DIR = 1_Floyd/src
BIN_DIR = bin
BUILD_DIR = build

# Source files
MAIN_SRC = 1_Floyd/main.cpp
FLOYD_SRC = $(SRC_DIR)/floyd.cpp
SOURCES = $(MAIN_SRC) $(FLOYD_SRC)

# Object files
MAIN_OBJ = $(BUILD_DIR)/main.o
FLOYD_OBJ = $(BUILD_DIR)/floyd.o
OBJECTS = $(MAIN_OBJ) $(FLOYD_OBJ)

# Target executable
TARGET = $(BIN_DIR)/floyd.exe

# Default target
all: $(TARGET)

# Create directories if they don't exist
$(BUILD_DIR):
	@if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"

$(BIN_DIR):
	@if not exist "$(BIN_DIR)" mkdir "$(BIN_DIR)"

# Link the final executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

# Compile main.cpp
$(MAIN_OBJ): $(MAIN_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile floyd.cpp
$(FLOYD_OBJ): $(FLOYD_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	@if exist "$(BUILD_DIR)" rmdir /s /q "$(BUILD_DIR)"
	@if exist "$(BIN_DIR)" rmdir /s /q "$(BIN_DIR)"

# Run the program (with default parameters)
run: $(TARGET)
	$(TARGET)

# Run with custom parameters (matrix size, density, threads)
# Usage: mingw32-make run-custom SIZE=200 DENSITY=0.5 THREADS=4
run-custom: $(TARGET)
	$(TARGET) $(SIZE) $(DENSITY) $(THREADS)

# Show help
help:
	@echo Available targets:
	@echo   all      - Build the project (default)
	@echo   clean    - Remove build files
	@echo   run      - Run the program with default parameters
	@echo   run-custom - Run with custom parameters
	@echo   help     - Show this help message
	@echo.
	@echo Example usage:
	@echo   mingw32-make
	@echo   mingw32-make clean
	@echo   mingw32-make run
	@echo   mingw32-make run-custom SIZE=200 DENSITY=0.5 THREADS=4

.PHONY: all clean run run-custom help