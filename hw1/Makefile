# Compiler settings
CC = g++
CFLAGS = -std=c++11 -O2 -Wall -fopenmp

# Build directories
BUILD_DIR = build

# Floyd project settings
FLOYD_DIR = 1_Floyd
FLOYD_SRC_DIR = $(FLOYD_DIR)/src
FLOYD_INC_DIR = $(FLOYD_DIR)/include
FLOYD_TARGET = $(BUILD_DIR)/floyd.exe
FLOYD_SOURCES = $(FLOYD_DIR)/main.cpp $(FLOYD_SRC_DIR)/floyd.cpp
FLOYD_HEADERS = $(FLOYD_INC_DIR)/floyd.h $(FLOYD_INC_DIR)/utils.h

# MonteCarlo project settings
MONTE_DIR = 2_MonteCarlo
MONTE_SRC_DIR = $(MONTE_DIR)/src
MONTE_INC_DIR = $(MONTE_DIR)/include
MONTE_TARGET = $(BUILD_DIR)/montecarlo.exe
MONTE_SOURCES = $(MONTE_DIR)/main.cpp $(MONTE_SRC_DIR)/MonteCarlo.cpp
MONTE_HEADERS = $(MONTE_INC_DIR)/MonteCarlo.h

# Convolution project settings
CONV_DIR = 3_Convolution
CONV_SRC_DIR = $(CONV_DIR)/src
CONV_INC_DIR = $(CONV_DIR)/include
CONV_TARGET = $(BUILD_DIR)/convolution.exe
CONV_SOURCES = $(CONV_DIR)/main.cpp $(CONV_SRC_DIR)/Convolution.cpp
CONV_HEADERS = $(CONV_INC_DIR)/Convolution.h

# Default target
all: $(FLOYD_TARGET) $(MONTE_TARGET) $(CONV_TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Floyd target
$(FLOYD_TARGET): $(FLOYD_SOURCES) $(FLOYD_HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(FLOYD_INC_DIR) -o $@ $(FLOYD_SOURCES)

# MonteCarlo target
$(MONTE_TARGET): $(MONTE_SOURCES) $(MONTE_HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(MONTE_INC_DIR) -o $@ $(MONTE_SOURCES)

# Convolution target
$(CONV_TARGET): $(CONV_SOURCES) $(CONV_HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(CONV_INC_DIR) -o $@ $(CONV_SOURCES)

# Individual targets
floyd: $(FLOYD_TARGET)

montecarlo: $(MONTE_TARGET)
convolution: $(CONV_TARGET)

# Clean target
clean:
	rm -rf $(BUILD_DIR)

# Test targets
test-floyd: $(FLOYD_TARGET)
	$(FLOYD_TARGET) 10 0.5 4

test-montecarlo: $(MONTE_TARGET)
	$(MONTE_TARGET) 1000000 4

test-convolution: $(CONV_TARGET)
	$(CONV_TARGET) 256 256 3 4

test: test-floyd test-montecarlo

# Help target
help:
	@echo Available targets:
	@echo   all          - Build both projects (default)
	@echo   floyd        - Build Floyd-Warshall project only
	@echo   montecarlo   - Build Monte Carlo project only
	@echo   test         - Run tests for both projects
	@echo   test-floyd   - Test Floyd-Warshall with sample parameters
	@echo   test-montecarlo - Test Monte Carlo with sample parameters
	@echo   clean        - Remove build directory and executables
	@echo   help         - Show this help message

.PHONY: all floyd montecarlo clean test test-floyd test-montecarlo help
