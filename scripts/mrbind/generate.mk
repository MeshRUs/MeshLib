# Where the makefile is located.
override makefile_dir := $(patsubst ./,.,$(dir $(firstword $(MAKEFILE_LIST))))

override define lf :=
$(call)
$(call)
endef

# This function encloses $1 in quotes. We also replace newlines with spaces.
override quote = '$(subst ','"'"',$(subst $(lf), ,$1))'

# You must set those when executing manually: [

# Where to find MRBind.
MESHLIB_SHLIB_DIR := build/Release/bin

# Source directory of MRBind.
MRBIND_SOURCE := _mrbind
# MRBind executable .
MRBIND_EXE := $(MRBIND_SOURCE)/build/mrbind

# The C++ compiler.
CXX ?= $(error Must set CXX)

# Look for `./lib` and `./include` relative to this.
DEPS_BASE_DIR := .

# ]

MODULE_OUTPUT_DIR := $(MESHLIB_SHLIB_DIR)/meshlib2
MRBIND_FLAGS := $(file <$(makefile_dir)/mrbind_flags.txt)

# Those variables are for mrbind/scripts/apply_to_files.mk
INPUT_DIRS := $(makefile_dir)/../../source/MRMesh $(makefile_dir)
INPUT_FILES_BLACKLIST := $(file <$(makefile_dir)/input_file_blacklist.txt)
OUTPUT_DIR := build/binds
INPUT_GLOBS := *.h
MRBIND := $(MRBIND_EXE) $(MRBIND_FLAGS)
COMPILER_FLAGS := $(file <$(makefile_dir)/common_compiler_parser_flags.txt) $(shell pkg-config --cflags python3-embed) -I. -I$(DEPS_BASE_DIR)/include -I$(makefile_dir)/../../source
COMPILER_FLAGS_LIBCLANG := $(file <$(makefile_dir)/parser_only_flags.txt)
COMPILER := $(CXX) $(file <$(makefile_dir)/compiler_only_flags.txt) -I$(MRBIND_SOURCE)/include
LINKER_OUTPUT := $(MODULE_OUTPUT_DIR)/mrmeshpy$(shell python3-config --extension-suffix)
LINKER := $(CXX) -fuse-ld=lld
LINKER_FLAGS := -Wl,-rpath='$$ORIGIN/..:$$ORIGIN' $(shell pkg-config --libs python3-embed) -L$(DEPS_BASE_DIR)/lib -L$(MESHLIB_SHLIB_DIR) -lMRMesh -shared $(file <$(makefile_dir)/linker_flags.txt)
NUM_FRAGMENTS := 4

override mrbind_vars = $(subst $,$$$$, \
	INPUT_DIRS=$(call quote,$(INPUT_DIRS)) \
	INPUT_FILES_BLACKLIST=$(call quote,$(INPUT_FILES_BLACKLIST)) \
	OUTPUT_DIR=$(call quote,$(OUTPUT_DIR)) \
	INPUT_GLOBS=$(call quote,$(INPUT_GLOBS)) \
	MRBIND=$(call quote,$(MRBIND)) \
	COMPILER_FLAGS=$(call quote,$(COMPILER_FLAGS)) \
	COMPILER_FLAGS_LIBCLANG=$(call quote,$(COMPILER_FLAGS_LIBCLANG)) \
	COMPILER=$(call quote,$(COMPILER)) \
	LINKER_OUTPUT=$(call quote,$(LINKER_OUTPUT)) \
	LINKER=$(call quote,$(LINKER)) \
	LINKER_FLAGS=$(call quote,$(LINKER_FLAGS)) \
	NUM_FRAGMENTS=$(call quote,$(NUM_FRAGMENTS)) \
)

# Generated mrmeshpy.
$(LINKER_OUTPUT): | $(MODULE_OUTPUT_DIR)
	$(MAKE) -f $(MRBIND_SOURCE)/scripts/apply_to_files.mk $(mrbind_vars)

# Only generate mrmeshpy, but don't compile.
.PHONY: only-generate
only-generate:
	$(MAKE) -f $(MRBIND_SOURCE)/scripts/apply_to_files.mk generate $(mrbind_vars)

# Handwritten mrmeshnumpy.
MRMESHPY_MODULE := $(MODULE_OUTPUT_DIR)/mrmeshnumpy$(shell python3-config --extension-suffix)
$(MRMESHPY_MODULE): | $(MODULE_OUTPUT_DIR)
	$(CXX) \
		-o $@ \
		$(makefile_dir)/../../source/mrmeshnumpy/*.cpp \
		$(COMPILER_FLAGS) \
		-fPIC \
		$(LINKER_FLAGS) \
		-DMRMESHNUMPY_PARENT_MODULE_NAME=$(notdir $(MODULE_OUTPUT_DIR))

# All modules.
.DEFAULT_GOAL := all
.PHONY: all
all: $(LINKER_OUTPUT) $(MRMESHPY_MODULE)

# The directory for the modules.
$(MODULE_OUTPUT_DIR):
	mkdir -p $@
