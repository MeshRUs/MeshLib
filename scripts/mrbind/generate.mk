# Some helper funcions. See below for the configuration variables...

# Where the makefile is located.
# Not entirely sure why I need to adjust `\` to `/` on Windows, since non-mingw32 Make should already operate on Linux-style paths?
override makefile_dir := $(patsubst ./,.,$(subst \,/,$(dir $(firstword $(MAKEFILE_LIST)))))

# A newline.
override define lf :=
$(call)
$(call)
endef

# This function encloses $1 in quotes. We also replace newlines with spaces.
override quote = '$(subst ','"'"',$(subst $(lf), ,$1))'

# Same as `$(shell ...)`, but triggers an error on failure.
override safe_shell =
ifneq ($(filter --trace,$(MAKEFLAGS)),)
override safe_shell = $(info Shell command: $1)$(shell $1)$(if $(filter-out 0,$(.SHELLSTATUS)),$(error Unable to execute `$1`, exit code $(.SHELLSTATUS)))
else
override safe_shell = $(shell $1)$(if $(filter-out 0,$(.SHELLSTATUS)),$(error Unable to execute `$1`, exit code $(.SHELLSTATUS)))
endif

# Loads the contents of file $1, replacing newlines with spaces.
override load_file = $(subst $(lf), ,$(file <$1))




# --- Configuration variables start here:


# Are we on Windows?
ifeq ($(OS),Windows_NT)
IS_WINDOWS := 1
else
IS_WINDOWS := 0
endif
override IS_WINDOWS := $(filter-out 0,$(IS_WINDOWS))

# On Windows, check that we are in the VS prompt, or at least that `VCToolsInstallDir` is defined (which is what Clang needs).
# Otherwise Clang may or may not choose some weird system libraries.
ifneq ($(IS_WINDOWS),)
ifeq ($(origin VCToolsInstallDir),undefined)
$(error Must run this in Visual Studio developer command prompt, or at least copy the value of the `VCToolsInstallDir` env variable)
endif
endif

# Windows-only vars: [

# For Windows, set this to Debug or Release. This controls which MeshLib build we'll be using.
VS_MODE := Debug

# Vcpkg installation directory. We try to auto-detect it.
ifneq ($(IS_WINDOWS),)
VCPKG_DIR :=
ifeq ($(VCPKG_DIR),)
override vcpkg_marker_path := $(LOCALAPPDATA)\vcpkg\vcpkg.path.txt
VCPKG_DIR := $(call load_file,$(vcpkg_marker_path))
ifeq ($(VCPKG_DIR),)
$(error Can't find vcpkg! The path to it should be stored in `$(vcpkg_marker_path)`, but it's not there)
endif
$(info Using vcpkg at: $(VCPKG_DIR))
endif
else
VCPKG_DIR = $(error We're only using vcpkg on Windows)
endif

# ]


# Where to find MeshLib.
ifneq ($(IS_WINDOWS),)
MESHLIB_SHLIB_DIR := source/x64/$(VS_MODE)
else
MESHLIB_SHLIB_DIR := build/Release/bin
endif
ifeq ($(wildcard $(MESHLIB_SHLIB_DIR)),)
$(warning MeshLib build directory `$(abspath $(MESHLIB_SHLIB_DIR))` doesn't exist! You either forgot to build MeshLib, or are running this script with the wrong current directory. Call this from your project's root)
endif

# Source directory of MRBind.
ifneq ($(IS_WINDOWS),)
MRBIND_SOURCE := $(HOME)/_mrbind
else
MRBIND_SOURCE := _mrbind
endif

# MRBind executable .
MRBIND_EXE := $(MRBIND_SOURCE)/build/mrbind

# The C++ compiler.
ifneq ($(IS_WINDOWS),)
CXX = clang++
else
CXX ?= $(error Must set `CXX=...`)
endif

# Extra compiler and linker flags.
EXTRA_CFLAGS :=
EXTRA_LDLAGS :=

# Look for MeshLib  dependencies relative to this. On Linux should point to the project root, because that's where `./include` and `./lib` are.
ifneq ($(IS_WINDOWS),)
DEPS_BASE_DIR := $(VCPKG_DIR)/installed/x64-windows-meshlib
DEPS_LIB_DIR := $(DEPS_BASE_DIR)/$(if $(filter Debug,$(VS_MODE)),debug/)lib
else
DEPS_BASE_DIR := .
DEPS_LIB_DIR := $(DEPS_BASE_DIR)/lib
endif
DEPS_INCLUDE_DIR := $(DEPS_BASE_DIR)/include

# Pkg-config name for Python.
ifneq ($(IS_WINDOWS),)
PYTHON_PKGCONF_NAME := $(basename $(notdir $(lastword $(sort $(wildcard $(DEPS_BASE_DIR)/lib/pkgconfig/python-*-embed.pc)))))
else
PYTHON_PKGCONF_NAME := python3-embed
endif
$(if $(PYTHON_PKGCONF_NAME),$(info Using Python version: $(PYTHON_PKGCONF_NAME:-embed=)),$(error Can't find the Python package in vcpkg))

# Python-config executable. Currently not used on Windows.
# Returns `python3-config`, or `python-3.XX-config`.
PYTHON_CONFIG := $(subst -,,$(PYTHON_PKGCONF_NAME:-embed=))-config

# Python compilation flags.
PYTHON_CFLAGS :=
PYTHON_LDFLAGS :=
ifeq ($(PYTHON_CFLAGS)$(PYTHON_LDFLAGS),)
ifneq ($(IS_WINDOWS),)
# Intentionally using non-debug Python even in Debug builds, to mimic what MeshLib does. Unsure why we do this.
PYTHON_CFLAGS := $(call safe_shell,PKG_CONFIG_PATH=$(call quote,$(DEPS_BASE_DIR)/lib/pkgconfig) PKG_CONFIG_LIBDIR=- pkg-config --cflags $(PYTHON_PKGCONF_NAME))
PYTHON_LDFLAGS := $(call safe_shell,PKG_CONFIG_PATH=$(call quote,$(DEPS_BASE_DIR)/lib/pkgconfig) PKG_CONFIG_LIBDIR=- pkg-config --libs $(PYTHON_PKGCONF_NAME))
else
PYTHON_CFLAGS := $(call safe_shell,pkg-config --cflags $(PYTHON_PKGCONF_NAME))
PYTHON_LDFLAGS := $(call safe_shell,pkg-config --libs $(PYTHON_PKGCONF_NAME))
endif
endif

# Python module suffix.
ifneq ($(IS_WINDOWS),)
PYTHON_MODULE_SUFFIX := .pyd
else
# This is a bit sketchy, because it does
PYTHON_MODULE_SUFFIX := $(call safe_shell,$(PYTHON_CONFIG) --extension-suffix)
endif
$(info Using Python module suffix: $(PYTHON_MODULE_SUFFIX))

# --- End of configuration variables.





MODULE_OUTPUT_DIR := $(MESHLIB_SHLIB_DIR)/meshlib2

# Those variables are for mrbind/scripts/apply_to_files.mk
INPUT_DIRS := $(addprefix $(makefile_dir)/../../source/,MRMesh MRSymbolMesh) $(makefile_dir)/extra_headers
INPUT_FILES_BLACKLIST := $(call load_file,$(makefile_dir)/input_file_blacklist.txt)
ifneq ($(IS_WINDOWS),)
OUTPUT_DIR := source/TempOutput/PythonBindings/x64/$(VS_MODE)
else
OUTPUT_DIR := build/binds
endif
INPUT_GLOBS := *.h
MRBIND := $(MRBIND_EXE)
MRBIND_FLAGS := $(call load_file,$(makefile_dir)/mrbind_flags.txt)
MRBIND_FLAGS_FOR_EXTRA_INPUTS := $(call load_file,$(makefile_dir)/mrbind_flags_for_helpers.txt)
COMPILER_FLAGS := $(EXTRA_CFLAGS) $(call load_file,$(makefile_dir)/common_compiler_parser_flags.txt) $(PYTHON_CFLAGS) -I. -I$(DEPS_INCLUDE_DIR) -I$(makefile_dir)/../../source
COMPILER_FLAGS_LIBCLANG := $(call load_file,$(makefile_dir)/parser_only_flags.txt)
COMPILER := $(CXX) $(subst $(lf), ,$(call load_file,$(makefile_dir)/compiler_only_flags.txt)) -I$(MRBIND_SOURCE)/include -I$(makefile_dir)
LINKER_OUTPUT := $(MODULE_OUTPUT_DIR)/mrmeshpy$(PYTHON_MODULE_SUFFIX)
LINKER := $(CXX) -fuse-ld=lld
LINKER_FLAGS := $(EXTRA_LDFLAGS) -L$(DEPS_LIB_DIR) $(PYTHON_LDFLAGS) -L$(MESHLIB_SHLIB_DIR) -lMRMesh -lMRSymbolMesh -shared $(call load_file,$(makefile_dir)/linker_flags.txt)
NUM_FRAGMENTS := 4
EXTRA_INPUT_SOURCES := $(makefile_dir)/helpers.cpp

ifneq ($(IS_WINDOWS),)
# "Cross"-compile to MSVC.
COMPILER_FLAGS += --target=x86_64-pc-windows-msvc
LINKER_FLAGS += --target=x86_64-pc-windows-msvc
# Set resource directory. Otherwise e.g. `offsetof` becomes non-constexpr,
#   because the header override with it being constexpr is in this resource directory.
COMPILER_FLAGS += -resource-dir=$(strip $(call safe_shell,$(CXX) -print-resource-dir))
# This seems to be undocumented?! MSYS2 CLANG64 needs it to successfully cross-compile, because the default `-rtlib=compiler-rt` causes it to choke.
# For some reason MIGNW64 and UCRT64 correctly guess the right default.
LINKER_FLAGS += -rtlib=platform
# Don't generate .lib files.
LINKER_FLAGS += -Wl,-noimplib
# Library paths:
COMPILER_FLAGS += -isystem $(makefile_dir)/../../thirdparty/pybind11/include
COMPILER_FLAGS += -isystem $(makefile_dir)/../../thirdparty/parallel-hashmap
COMPILER_FLAGS += -D_DLL -D_MT
ifeq ($(VS_MODE),Debug)
COMPILER_FLAGS += -Xclang --dependent-lib=msvcrtd -D_DEBUG
# Override to match meshlib:
COMPILER_FLAGS += -D_ITERATOR_DEBUG_LEVEL=0
else
COMPILER_FLAGS += -Xclang --dependent-lib=msvcrt
endif
else # Linux:
COMPILER += -fPIC
COMPILER_FLAGS += -I/usr/include/jsoncpp -isystem/usr/include/freetype2 -isystem/usr/include/gdcm-3.0
LINKER_FLAGS += -Wl,-rpath='$$ORIGIN/..:$$ORIGIN'
endif


override mrbind_vars = $(subst $,$$$$, \
	INPUT_DIRS=$(call quote,$(INPUT_DIRS)) \
	INPUT_FILES_BLACKLIST=$(call quote,$(INPUT_FILES_BLACKLIST)) \
	OUTPUT_DIR=$(call quote,$(OUTPUT_DIR)) \
	INPUT_GLOBS=$(call quote,$(INPUT_GLOBS)) \
	MRBIND=$(call quote,$(MRBIND)) \
	MRBIND_FLAGS=$(call quote,$(MRBIND_FLAGS)) \
	MRBIND_FLAGS_FOR_EXTRA_INPUTS=$(call quote,$(MRBIND_FLAGS_FOR_EXTRA_INPUTS)) \
	COMPILER_FLAGS=$(call quote,$(COMPILER_FLAGS)) \
	COMPILER_FLAGS_LIBCLANG=$(call quote,$(COMPILER_FLAGS_LIBCLANG)) \
	COMPILER=$(call quote,$(COMPILER)) \
	LINKER_OUTPUT=$(call quote,$(LINKER_OUTPUT)) \
	LINKER=$(call quote,$(LINKER)) \
	LINKER_FLAGS=$(call quote,$(LINKER_FLAGS)) \
	NUM_FRAGMENTS=$(call quote,$(NUM_FRAGMENTS)) \
	EXTRA_INPUT_SOURCES=$(call quote,$(EXTRA_INPUT_SOURCES)) \
)

# Generated mrmeshpy.
$(LINKER_OUTPUT): | $(MODULE_OUTPUT_DIR)
	$(MAKE) -f $(MRBIND_SOURCE)/scripts/apply_to_files.mk $(mrbind_vars)

# Only generate mrmeshpy, but don't compile.
.PHONY: only-generate
only-generate:
	$(MAKE) -f $(MRBIND_SOURCE)/scripts/apply_to_files.mk generate $(mrbind_vars)

# Handwritten mrmeshnumpy.
MRMESHNUMPY_MODULE := $(MODULE_OUTPUT_DIR)/mrmeshnumpy$(PYTHON_MODULE_SUFFIX)
$(MRMESHNUMPY_MODULE): | $(MODULE_OUTPUT_DIR)
	$(COMPILER) \
		-o $@ \
		$(makefile_dir)/../../source/mrmeshnumpy/*.cpp \
		$(COMPILER_FLAGS) $(LINKER_FLAGS) \
		-DMRMESHNUMPY_PARENT_MODULE_NAME=$(notdir $(MODULE_OUTPUT_DIR))

# The init script.
INIT_SCRIPT := $(MODULE_OUTPUT_DIR)/__init__.py
$(INIT_SCRIPT): $(makefile_dir)/__init__.py
	cp $< $@
ifeq ($(IS_WINDOWS),) # If not on Windows, strip the windows-only part.
	gawk -i inplace '/### windows-only: \[/{x=1} {if (!x) print} x && /### \]/{x=0}' $@
endif

# All modules.
.DEFAULT_GOAL := all
.PHONY: all
all: $(LINKER_OUTPUT) $(MRMESHNUMPY_MODULE) $(INIT_SCRIPT)

# The directory for the modules.
$(MODULE_OUTPUT_DIR):
	mkdir -p $@
