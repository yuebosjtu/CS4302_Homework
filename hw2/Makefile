# CUDA Compiler settings
NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_50

# Build directories
BUILD_DIR = build

# MatrixMul project settings
MATRIXMUL_DIR = 1_MatrixMul
MATRIXMUL_SRC_DIR = $(MATRIXMUL_DIR)/src
MATRIXMUL_TARGET = $(BUILD_DIR)/MatrixMul.exe
MATRIXMUL_SOURCES = $(MATRIXMUL_SRC_DIR)/MatrixMul.cu

# ArraySum project settings
ARRAYSUM_DIR = 2_ArraySum
ARRAYSUM_SRC_DIR = $(ARRAYSUM_DIR)/src
ARRAYSUM_TARGET = $(BUILD_DIR)/ArraySum.exe
ARRAYSUM_SOURCES = $(ARRAYSUM_SRC_DIR)/ArraySum.cu

# Default target
all: $(MATRIXMUL_TARGET) $(ARRAYSUM_TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	@powershell -Command "if (!(Test-Path '$(BUILD_DIR)')) { New-Item -ItemType Directory -Path '$(BUILD_DIR)' | Out-Null }"

# MatrixMul target
$(MATRIXMUL_TARGET): $(MATRIXMUL_SOURCES) | $(BUILD_DIR)
	@powershell -Command "if (!(Test-Path '$(BUILD_DIR)')) { New-Item -ItemType Directory -Path '$(BUILD_DIR)' | Out-Null }"
	$(NVCC) $(NVCCFLAGS) -o $@ $(MATRIXMUL_SOURCES)

# ArraySum target
$(ARRAYSUM_TARGET): $(ARRAYSUM_SOURCES) | $(BUILD_DIR)
	@powershell -Command "if (!(Test-Path '$(BUILD_DIR)')) { New-Item -ItemType Directory -Path '$(BUILD_DIR)' | Out-Null }"
	$(NVCC) $(NVCCFLAGS) -o $@ $(ARRAYSUM_SOURCES)

# Clean target
clean:
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean
