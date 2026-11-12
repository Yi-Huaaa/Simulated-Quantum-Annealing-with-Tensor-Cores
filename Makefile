NVCC      ?= nvcc
# GPU architecture, e.g. sm_70 (V100), sm_80 (A100), sm_86 (RTX 30xx / A4000),
# sm_89 (RTX 40xx), sm_90 (H100). "native" targets the GPU of the build machine.
ARCH      ?= native
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler -Wall,-Wextra

BUILD_DIR := build
TARGET    := $(BUILD_DIR)/sqa
SRCS      := src/main.cu src/sqa.cu src/graph.cpp
HEADERS   := $(wildcard src/*.h)

.PHONY: all test gset bench clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $(SRCS) -o $@ -lcublas

test: $(TARGET)
	./tests/run_tests.sh $(TARGET)

gset:
	./scripts/download_gset.sh

bench: $(TARGET)
	./scripts/benchmark.sh $(TARGET)

clean:
	rm -rf $(BUILD_DIR)
