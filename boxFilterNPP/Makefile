# Makefile for Gaussian Blur with CUDA and NPP

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 sbsa ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE} -std=c++17
CCFLAGS     := -std=c++17
LDFLAGS     := -lstdc++fs

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
endif

ifdef TARGET_OVERRIDE # cuda toolkit targets override
    NVCCFLAGS += -target-dir $(TARGET_OVERRIDE)
endif

# Debug build flags
ifeq ($(dbg),1)
    NVCCFLAGS += -g -G
    BUILD_TYPE := debug
else
    BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

SAMPLE_ENABLED := 1

# This sample is not supported on QNX
ifeq ($(TARGET_OS),qnx)
  $(info >>> WARNING - Gaussian Blur Sample is not supported on QNX - waiving sample <<<)
  SAMPLE_ENABLED := 0
endif

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I../Common -I../Common/UtilNPP
LIBRARIES := -lnppisu_static -lnppif_static -lnppc_static -lculibos -lfreeimage

# Gencode arguments
SMS ?=
ifeq ($(GENCODE_FLAGS),)
# Generate PTX code from SM 35
GENCODE_FLAGS += -gencode arch=compute_35,code=compute_35
endif

ALL_CCFLAGS += --threads 0

# Target rules
all: build

build: gaussianBlurNPP
	@echo "Building project..." > build_and_run.txt
	$(MAKE) gaussianBlurNPP >> build_and_run.txt 2>&1

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

gaussianBlurNPP.o: gaussianBlurNPP.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

gaussianBlurNPP: gaussianBlurNPP.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	mkdir -p ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
	cp $@ ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

run: build
	@echo "Running project..." >> build_and_run.txt
	./gaussianBlurNPP -input=/home/coder/project/boxFilterNPP/PGM_Images -output=/home/coder/project/boxFilterNPP/PGM_Images/processed >> build_and_run.txt 2>&1

clean:
	rm -f gaussianBlurNPP gaussianBlurNPP.o
	rm -rf ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/gaussianBlurNPP

clobber: clean
