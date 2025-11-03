# CUDA Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -arch=sm_50 -O3

# Detect OS
ifeq ($(OS),Windows_NT)
    RM = rmdir /s /q
    EXE_EXT = .exe
    SEP = \\
else
    RM = rm -rf
    EXE_EXT =
    SEP = /
endif

# Directories
BUILD_DIR = build
MATRIXMUL_DIR = 1_MatrixMul
ARRAYSUM_DIR = 2_ArraySum

# Source files
MATRIXMUL_SRC = $(MATRIXMUL_DIR)$(SEP)src$(SEP)MatrixMul.cu
ARRAYSUM_SRC = $(ARRAYSUM_DIR)$(SEP)src$(SEP)ArraySum.cu

# Executable files
MATRIXMUL_BIN = $(BUILD_DIR)$(SEP)MatrixMul$(EXE_EXT)
ARRAYSUM_BIN = $(BUILD_DIR)$(SEP)ArraySum$(EXE_EXT)

# Default target: build all
all: $(BUILD_DIR) $(MATRIXMUL_BIN) $(ARRAYSUM_BIN)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Build MatrixMul
$(MATRIXMUL_BIN): $(MATRIXMUL_SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@
	@echo "MatrixMul compiled successfully!"

# Build ArraySum
$(ARRAYSUM_BIN): $(ARRAYSUM_SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@
	@echo "ArraySum compiled successfully!"

# Run MatrixMul
run-matrixmul: $(MATRIXMUL_BIN)
	@echo "Running MatrixMul..."
	$(MATRIXMUL_BIN)

# Run ArraySum
run-arraysum: $(ARRAYSUM_BIN)
	@echo "Running ArraySum..."
	$(ARRAYSUM_BIN)

# Run all programs
run-all: run-matrixmul run-arraysum

# Clean build directory
clean:
	@$(RM) $(BUILD_DIR) 2>nul || echo "Build directory already clean"

# Rebuild everything
rebuild: clean all

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Build all programs (default)"
	@echo "  clean         - Remove build directory"
	@echo "  rebuild       - Clean and build all"
	@echo "  run-matrixmul - Build and run MatrixMul"
	@echo "  run-arraysum  - Build and run ArraySum"
	@echo "  run-all       - Build and run all programs"
	@echo "  help          - Show this help message"

.PHONY: all clean rebuild run-matrixmul run-arraysum run-all help
