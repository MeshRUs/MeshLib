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

# Same as `safe_shell`, but discards the output.
override safe_shell_exec = $(call,$(call safe_shell,$1))

# Loads the contents of file $1, replacing newlines with spaces.
override load_file = $(subst $(lf), ,$(file <$1))

# Compare version numbers: A <= B
override version_leq = $(shell printf '%s\n' $1 $2 | sort -CV)$(filter 0,$(.SHELLSTATUS))



# --- Configuration variables start here:


# Are we on Windows?
ifeq ($(OS),Windows_NT)
IS_WINDOWS := 1
IS_MACOS := 0
else ifeq ($(shell uname -s),Darwin)
IS_WINDOWS := 0
IS_MACOS := 1
endif
override IS_WINDOWS := $(filter-out 0,$(IS_WINDOWS))
override IS_MACOS := $(filter-out 0,$(IS_MACOS))

# On Windows, check that we are in the VS prompt, or at least that `VCToolsInstallDir` is defined (which is what Clang needs).
# Otherwise Clang may or may not choose some weird system libraries.
ifneq ($(IS_WINDOWS),)
ifeq ($(origin VCToolsInstallDir),undefined)
$(error Must run this in Visual Studio developer command prompt, or at least copy the value of the `VCToolsInstallDir` env variable)
endif
endif

# Windows-only vars: [

# For Windows, set this to Debug or Release. This controls which MeshLib build we'll be using.
VS_MODE := Release

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

# MacOS-only vars: [
ifneq ($(IS_MACOS),)
HOMEBREW_DIR := /opt/homebrew
ifeq ($(wildcard $(HOMEBREW_DIR)),)
# Apparently x86 Macs don't use `/opt/homebrew`, but rather `/usr/local`.
HOMEBREW_DIR := /usr/local
endif
$(info Using homebrew at: $(HOMEBREW_DIR))
endif

# Min version. Not setting this seems to cause warnings when linking against MeshLib built with Apple Clang, which seems to have different defaults.
MACOS_MIN_VER :=
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
MRBIND_SOURCE := ~/mrbind

# MRBind executable .
MRBIND_EXE := $(MRBIND_SOURCE)/build/mrbind

# The C++ compiler.
# Note that on Windows we don't have control over the Clang version, and ignore `preferred_clang_version.txt`.
ifneq ($(IS_WINDOWS),)
CXX_FOR_BINDINGS := clang++
else ifneq ($(IS_MACOS),)
CXX_FOR_BINDINGS := $(HOMEBREW_DIR)/opt/llvm@$(strip $(file <$(makefile_dir)/preferred_clang_version.txt))/bin/clang++
else
# Only on Ubuntu we don't want the default Clang version, as it can be outdated. Use the suffixed one.
CXX_FOR_BINDINGS := clang++-$(strip $(file <$(makefile_dir)/preferred_clang_version.txt))
endif

# Which C++ compiler we should try to match for ABI.
# Ignored on Windows.
CXX_FOR_ABI := $(if $(CXX),$(CXX),g++)
ABI_COMPAT_FLAG :=
# On Linux and MacOS, check if this compiler mangles C++20 constraints into function names. If not (old compilers), pass `-fclang-abi-compat=17` to prevent Clang 18 from mangling those.
ifeq ($(IS_WINDOWS),)# If not on Windows:
$(call safe_shell_exec,which $(CXX_FOR_ABI) >/dev/null 2>/dev/null)# Make sure this compiler exists.
ifneq ($(shell echo "template <typename T> void foo() requires true {} template void foo<int>();" | $(CXX_FOR_ABI) -xc++ - -std=c++20 -S -o - | grep -m1 '\b_Z3fooIiEvvQLb1E\b')$(filter 0,$(.SHELLSTATUS)),)
$(info ABI check: $(CXX_FOR_ABI) DOES mangle C++20 constraints into the function names.)
else
$(info ABI check: $(CXX_FOR_ABI) DOESN'T mangle C++20 constraints into the function names, enabling `-fclang-abi-compat=17`)
ABI_COMPAT_FLAG := -fclang-abi-compat=17
endif
endif
override CXX_FOR_BINDINGS += $(ABI_COMPAT_FLAG)


# Extra compiler and linker flags. `EXTRA_CFLAGS` also affect the parser.
EXTRA_CFLAGS :=
EXTRA_LDLAGS :=

# Flag presets.
MODE := release
ifeq ($(MODE),release)
override EXTRA_CFLAGS += -Oz -flto=thin
override EXTRA_LDFLAGS += -Oz -flto=thin $(if $(IS_MACOS),,-s)# No `-s` on macos. It seems to have no effect, and the linker warns about it.
else ifeq ($(MODE),debug)
override EXTRA_CFLAGS += -g
override EXTRA_LDFLAGS += -g
else ifeq ($(MODE),none)
# Nothing.
else
$(error Unknown MODE=$(MODE))
endif

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
ifneq ($(and $(value PYTHON_CFLAGS),$(value PYTHON_LDFLAGS)),)
$(info Using custom Python flags.)
else
ifneq ($(IS_WINDOWS),)
PYTHON_PKGCONF_NAME := $(basename $(notdir $(lastword $(sort $(wildcard $(DEPS_BASE_DIR)/lib/pkgconfig/python-*-embed.pc)))))
else
PYTHON_PKGCONF_NAME := python3-embed
endif
$(if $(PYTHON_PKGCONF_NAME),$(info Using Python version: $(PYTHON_PKGCONF_NAME:-embed=)),$(error Can't find the Python package in vcpkg))
endif

# Python compilation flags.
ifneq ($(IS_WINDOWS),)
# Intentionally using non-debug Python even in Debug builds, to mimic what MeshLib does. Unsure why we do this.
PYTHON_CFLAGS := $(if $(PYTHON_PKGCONF_NAME),$(call safe_shell,PKG_CONFIG_PATH=$(call quote,$(DEPS_BASE_DIR)/lib/pkgconfig) PKG_CONFIG_LIBDIR=- pkg-config --cflags $(PYTHON_PKGCONF_NAME)))
PYTHON_LDFLAGS := $(if $(PYTHON_PKGCONF_NAME),$(call safe_shell,PKG_CONFIG_PATH=$(call quote,$(DEPS_BASE_DIR)/lib/pkgconfig) PKG_CONFIG_LIBDIR=- pkg-config --libs $(PYTHON_PKGCONF_NAME)))
else # Linux or MacOS:
PYTHON_CFLAGS := $(call safe_shell,pkg-config --cflags $(PYTHON_PKGCONF_NAME))
ifneq ($(IS_MACOS),)
# On MacOS we don't link Python, instead we use `-Xlinker -undefined -Xlinker dynamic_lookup` to avoid the errors.
# This is important to avoid segfaults when importing the wheel.
PYTHON_LDFLAGS :=
else
PYTHON_LDFLAGS := $(call safe_shell,pkg-config --libs $(PYTHON_PKGCONF_NAME))
endif
endif

# Python module suffix.
ifneq ($(IS_WINDOWS),)
PYTHON_MODULE_SUFFIX := .pyd
else
PYTHON_MODULE_SUFFIX := .so
# # Python-config executable. Returns `python3-config`, or `python-3.XX-config`.
# PYTHON_CONFIG := $(subst -,,$(PYTHON_PKGCONF_NAME:-embed=))-config
# PYTHON_MODULE_SUFFIX := $(call safe_shell,$(PYTHON_CONFIG) --extension-suffix)
endif
$(info Using Python module suffix: $(PYTHON_MODULE_SUFFIX))


INPUT_PROJECTS := MRMesh MRIOExtras MRSymbolMesh MRVoxels

# --- End of configuration variables.




PACKAGE_NAME := meshlib2
MODULE_OUTPUT_DIR := $(MESHLIB_SHLIB_DIR)/$(PACKAGE_NAME)

# Those variables are for mrbind/scripts/apply_to_files.mk
INPUT_DIRS := $(addprefix $(makefile_dir)/../../source/,$(INPUT_PROJECTS)) $(makefile_dir)/extra_headers
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
# Need whitespace before `$(MRBIND_SOURCE)` to handle `~` correctly.
COMPILER := $(CXX_FOR_BINDINGS) $(subst $(lf), ,$(call load_file,$(makefile_dir)/compiler_only_flags.txt)) -I $(MRBIND_SOURCE)/include -I$(makefile_dir)
LINKER_OUTPUT := $(MODULE_OUTPUT_DIR)/mrmeshpy$(PYTHON_MODULE_SUFFIX)
LINKER := $(CXX_FOR_BINDINGS) -fuse-ld=lld
# Unsure if `-dynamiclib` vs `-shared` makes any difference on MacOS. I'm using the former because that's what CMake does.
LINKER_FLAGS := $(EXTRA_LDFLAGS) -L$(DEPS_LIB_DIR) $(PYTHON_LDFLAGS) -L$(MESHLIB_SHLIB_DIR) $(addprefix -l,$(INPUT_PROJECTS)) -lMRPython $(if $(IS_MACOS),-dynamiclib,-shared) $(call load_file,$(makefile_dir)/linker_flags.txt)
NUM_FRAGMENTS := 4
EXTRA_INPUT_SOURCES := $(makefile_dir)/helpers.cpp

ifneq ($(IS_WINDOWS),)
# "Cross"-compile to MSVC.
COMPILER_FLAGS += --target=x86_64-pc-windows-msvc
LINKER_FLAGS += --target=x86_64-pc-windows-msvc
# Set resource directory. Otherwise e.g. `offsetof` becomes non-constexpr,
#   because the header override with it being constexpr is in this resource directory.
COMPILER_FLAGS += -resource-dir=$(strip $(call safe_shell,$(CXX_FOR_BINDINGS) -print-resource-dir))
# This seems to be undocumented?! MSYS2 CLANG64 needs it to successfully cross-compile, because the default `-rtlib=compiler-rt` causes it to choke.
# For some reason MIGNW64 and UCRT64 correctly guess the right default.
LINKER_FLAGS += -rtlib=platform
# Don't generate .lib files.
LINKER_FLAGS += -Wl,-noimplib
# Library paths:
COMPILER_FLAGS += -isystem $(makefile_dir)/../../thirdparty/pybind11/include
COMPILER_FLAGS += -isystem $(makefile_dir)/../../thirdparty/parallel-hashmap
COMPILER_FLAGS += -D_DLL -D_MT
# Only seems to matter on VS2022 and not on VS2019, for some reason.
COMPILER_FLAGS += -DNOMINMAX
COMPILER_FLAGS += -D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS
ifeq ($(VS_MODE),Debug)
COMPILER_FLAGS += -Xclang --dependent-lib=msvcrtd -D_DEBUG
# Override to match meshlib:
COMPILER_FLAGS += -D_ITERATOR_DEBUG_LEVEL=0
else # VS_MODE == Release
COMPILER_FLAGS += -Xclang --dependent-lib=msvcrt
endif
else # Linux or MacOS:
COMPILER += -fPIC
COMPILER += -fvisibility=hidden
# MacOS rpath is quirky: 1. Must use `-rpath,` instead of `-rpath=`. 2. Must specify the flag several times, apparently can't use
#   `:` or `;` as a separators inside of one big flag. 3. As you've noticed, it uses `@loader_path` instead of `$ORIGIN`.
rpath_origin := $(if $(IS_MACOS),@loader_path,$$ORIGIN)
LINKER_FLAGS += -Wl,-rpath,'$(rpath_origin)' -Wl,-rpath,'$(rpath_origin)/..' -Wl,-rpath,$(call quote,$(abspath $(MODULE_OUTPUT_DIR))) -Wl,-rpath,$(call quote,$(abspath $(MESHLIB_SHLIB_DIR))) -Wl,-rpath,$(call quote,$(abspath $(DEPS_LIB_DIR)))
ifneq ($(IS_MACOS),)
# Hmm.
COMPILER_FLAGS_LIBCLANG += -resource-dir=$(strip $(call safe_shell,$(CXX_FOR_BINDINGS) -print-resource-dir))
# Our dependencies are here.
COMPILER_FLAGS += -I$(HOMEBREW_DIR)/include
# Boost.stacktrace complains otherwise.
COMPILER_FLAGS += -D_GNU_SOURCE
LINKER_FLAGS += -L$(HOMEBREW_DIR)/lib
LINKER_FLAGS += -ltbb
# This fixes an error during wheel creation:
#   /Library/Developer/CommandLineTools/usr/bin/install_name_tool: changing install names or rpaths can't be redone for: /private/var/folders/c2/_t7lgq_s3zb_r01vy_1qd6nh0000gs/T/tmpatczljnu/wheel/meshlib/mrmeshpy.so (for architecture arm64) because larger updated load commands do not fit (the program must be relinked, and you may need to use -headerpad or -headerpad_max_install_names)
# Apparently there's not enough space in the binary to fit longer library paths, and this pads it to have to up MAXPATHLEN space for each path.
LINKER_FLAGS += -Wl,-headerpad_max_install_names
# Those fix a segfault when importing the module, that only happens for wheels, not raw binaries.
# Pybind manual says you must use those.
# Also note that this is one long flag (`-undefined dynamic_lookup`), not two independent fones.
# Also not that you MUST NOT link the Python libs in addition to doing this, otherwise you'll still get segfaults.
LINKER_FLAGS += -Xlinker -undefined -Xlinker dynamic_lookup
# The min version. We override it to avoid incompatibility warnings against Apple Clang when linking.
ifneq ($(MACOS_MIN_VER),)
COMPILER_FLAGS += -mmacosx-version-min=$(MACOS_MIN_VER)
LINKER_FLAGS += -mmacosx-version-min=$(MACOS_MIN_VER)
endif
else # Linux:
COMPILER_FLAGS += -I/usr/include/jsoncpp -isystem/usr/include/freetype2 -isystem/usr/include/gdcm-3.0
endif
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
	@$(MAKE) -f $(MRBIND_SOURCE)/scripts/apply_to_files.mk $(mrbind_vars)

# Only generate mrmeshpy, but don't compile.
.PHONY: only-generate
only-generate:
	@$(MAKE) -f $(MRBIND_SOURCE)/scripts/apply_to_files.mk generate $(mrbind_vars)

# Handwritten mrmeshnumpy.
MRMESHNUMPY_MODULE := $(MODULE_OUTPUT_DIR)/mrmeshnumpy$(PYTHON_MODULE_SUFFIX)
$(MRMESHNUMPY_MODULE): | $(MODULE_OUTPUT_DIR)
	@echo $(call quote,[Compiling] mrmeshnumpy)
	@$(COMPILER) \
		-o $@ \
		$(makefile_dir)/../../source/mrmeshnumpy/*.cpp \
		$(COMPILER_FLAGS) $(LINKER_FLAGS) \
		-DMRMESHNUMPY_PARENT_MODULE_NAME=$(PACKAGE_NAME)

# The init script.
INIT_SCRIPT := $(MODULE_OUTPUT_DIR)/__init__.py
$(INIT_SCRIPT): $(makefile_dir)/__init__.py
	@cp $< $@
ifeq ($(IS_WINDOWS),) # If not on Windows, strip the windows-only part.
	@gawk -i inplace '/### windows-only: \[/{x=1} {if (!x) print} x && /### \]/{x=0}' $@
endif

ALL_OUTPUTS := $(LINKER_OUTPUT) $(MRMESHNUMPY_MODULE) $(INIT_SCRIPT)

# Copying modules next to the exe on Windows.
ifneq ($(IS_WINDOWS),)
ALL_OUTPUTS += $(MESHLIB_SHLIB_DIR)/__init__.py
$(MESHLIB_SHLIB_DIR)/__init__.py: $(INIT_SCRIPT)
	@cp $< $@
ALL_OUTPUTS += $(MESHLIB_SHLIB_DIR)/mrmeshpy$(PYTHON_MODULE_SUFFIX)
$(MESHLIB_SHLIB_DIR)/mrmeshpy$(PYTHON_MODULE_SUFFIX): $(MODULE_OUTPUT_DIR)/mrmeshpy$(PYTHON_MODULE_SUFFIX)
	@cp $< $@
ALL_OUTPUTS += $(MESHLIB_SHLIB_DIR)/mrmeshnumpy$(PYTHON_MODULE_SUFFIX)
$(MESHLIB_SHLIB_DIR)/mrmeshnumpy$(PYTHON_MODULE_SUFFIX): $(MODULE_OUTPUT_DIR)/mrmeshnumpy$(PYTHON_MODULE_SUFFIX)
	@cp $< $@
endif

# All modules.
.DEFAULT_GOAL := all
.PHONY: all
all: $(ALL_OUTPUTS)

# The directory for the modules.
$(MODULE_OUTPUT_DIR):
	@mkdir -p $@
