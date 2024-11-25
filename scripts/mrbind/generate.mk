# Some helper funcions. See below for the configuration variables...

# Where the makefile is located.
# Not entirely sure why I need to adjust `\` to `/` on Windows, since non-mingw32 Make should already operate on Linux-style paths?
override makefile_dir := $(patsubst ./,.,$(subst \,/,$(dir $(firstword $(MAKEFILE_LIST)))))

# A string of all single-letter Make flags, without spaces.
override single_letter_makeflags := $(filter-out -%,$(firstword $(MAKEFLAGS)))
# Non-empty if this is a dry run with `-n`.
override dry_run := $(findstring n,$(single_letter_makeflags))
# Non-empty if `--trace` is present.
override tracing := $(filter --trace,$(MAKEFLAGS))

# A newline.
override define lf :=
$(call)
$(call)
endef

# This function encloses $1 in quotes. We also replace newlines with spaces.
override quote = '$(subst ','"'"',$(subst $(lf), ,$1))'

# Same as `$(shell ...)`, but triggers an error on failure.
override safe_shell = $(if $(dry_run),$(warning Would run command: $1),$(if $(tracing),$(warning Running command: $1))$(shell $1)$(if $(tracing),$(warning Command returned $(.SHELLSTATUS)))$(if $(filter 0,$(.SHELLSTATUS)),,$(error Command failed with exit code $(.SHELLSTATUS): `$1`)))

# Same as `safe_shell`, but discards the output.
override safe_shell_exec = $(call,$(call safe_shell,$1))

# Loads the contents of file $1, replacing newlines with spaces.
override load_file = $(strip $(file <$1))

# Compare version numbers: A <= B
override version_leq = $(shell printf '%s\n' $1 $2 | sort -CV)$(filter 0,$(.SHELLSTATUS))

# Recursive wildcard function. $1 is a list of directories, $2 is a list of wildcards.
override rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

# Assign to a variable safely, e.g. `$(call var,foo := 42)`.
override var = $(eval override $(subst $,$$$$,$1))





# --- Configuration variables start here:


# What OS?
ifeq ($(OS),Windows_NT)
IS_WINDOWS := 1
IS_MACOS := 0
else
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
IS_WINDOWS := 0
IS_MACOS := 1
endif
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

# Set to 1 if you're planning to make a wheel from this module.
FOR_WHEEL := 0
override FOR_WHEEL := $(filter-out 0,$(FOR_WHEEL))

# Set to 1 if MeshLib was built in debug mode. Ignore this on Windows. By default we're trying to guess this based on the CMake cache.
# Currently this isn't needed for anything, hence commented out.
# MESHLIB_IS_DEBUG :=
# ifeq ($(IS_WINDOWS),)
# MESHLIB_IS_DEBUG := $(if $(filter Debug,$(shell cmake -L $(MESHLIB_SHLIB_DIR)/.. 2>/dev/null | grep -Po '(?<=CMAKE_BUILD_TYPE:STRING=).*')),1)
# $(info MeshLib built in debug mode? $(if $(filter-out 0,$(MESHLIB_IS_DEBUG)),YES,NO))
# endif
# override MESHLIB_IS_DEBUG := $(filter-out 0,$(MESHLIB_IS_DEBUG))


# ---- Windows-only vars: [

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
# ] ----

# ---- MacOS-only vars: [
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
# ] ----


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
MRBIND_SOURCE := $(makefile_dir)../../mrbind

# MRBind executable .
MRBIND_EXE := $(MRBIND_SOURCE)/build/mrbind

# The C++ compiler.
ifneq ($(IS_WINDOWS),)
CXX_FOR_BINDINGS := clang++
else ifneq ($(IS_MACOS),)
CXX_FOR_BINDINGS := $(HOMEBREW_DIR)/opt/llvm@$(strip $(file <$(makefile_dir)/clang_version.txt))/bin/clang++
else
# Only on Ubuntu we don't want the default Clang version, as it can be outdated. Use the suffixed one.
CXX_FOR_BINDINGS := clang++-$(strip $(file <$(makefile_dir)/clang_version.txt))
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

# Look for MeshLib dependencies relative to this. On Linux should point to the project root, because that's where `./include` and `./lib` are.
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
# Note that we're not using `DEPS_LIB_DIR` here, to always use the release Python on Windows, because that's what MeshLib itself seems to do.
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


# Which MeshLib projects to bind.
INPUT_PROJECTS := MRMesh MRIOExtras MRSymbolMesh MRVoxels

# 1 or 0. Whether to build mrmeshnumpy (if false you should build it with CMake with the rest of MeshLib).
# Currently defaults to 1 because otherwise we get an incompatibility on Ubuntu x86 20.04 and 22.04 when MeshLib is built in debug mode,
#   resulting in this error: `ImportError: arg(): could not convert default argument 'settings: MR::MeshBuilder::BuildSettings' in function 'meshFromFacesVerts' into a Python object`.
# If you fix this and want to change the default: 1. Remove this line and uncomment the one below. 2. In `distribution.sh` stop calling patchelf for `mrmeshnumpy.so`. 3. In `MeshLib/CMakeLists.txt`, move `mrmeshnumpy` out from `MESHLIB_BUILD_MRMESH_PY_LEGACY`.
BUILD_MRMESHNUMPY := 1
# Defaults to 0, but only if if `PACKAGE_NAME == meshlib`.
# BUILD_MRMESHNUMPY := $(if $(filter $(PACKAGE_NAME),meshlib),1)
override BUILD_MRMESHNUMPY := $(filter-out 0,$(BUILD_MRMESHNUMPY))

# Enable PCH.
ENABLE_PCH := 1
override ENABLE_PCH := $(filter-out 0,$(ENABLE_PCH))

# If this isn't empty, those are passed when compiling the PCH, and then the PCH is compiled to an object file and linked into the final result.
PCH_CODEGEN_FLAGS :=
# Those don't work at the moment (undefined references, d).
# The `-fpch-instantiate-templates` flag is optional, while the other two are necessary (at least the first one), and are usually used together.
# PCH_CODEGEN_FLAGS := -fpch-codegen -fpch-debuginfo -fpch-instantiate-templates



# --- Guess the build settings for the optimal speed:

# Guess the amount of RAM we have (in gigabytes), to select an appropriate build profile.
# Also guess the number of CPU cores.
ifneq ($(IS_MACOS),)
ASSUME_RAM := $(shell bash -c 'echo $$(($(shell sysctl -n hw.memsize) / 1000000000))')
ASSUME_NPROC := $(call safe_shell,sysctl -n hw.ncpu)
else
# `--giga` uses 10^3 instead of 2^10, which is actually good for us, since it overreports a bit, which counters computers typically having slightly less RAM than 2^N gigs.
ASSUME_RAM := $(shell LANG= free --giga 2>/dev/null | gawk 'NR==2{print $$2}')
ASSUME_NPROC := $(call safe_shell,nproc)
endif

# We clamp the nproc to this value, because when you have more cores, our heuristics fall apart (and you might run out of ram).
# The heuristics are not necessarily bad though, it's possible that less cores than jobs can be better in some cases?
MAX_NPROC := 16
CAPPED_NPROC := $(ASSUME_NPROC)
override nproc_string := $(ASSUME_NPROC) cores
ifeq ($(call safe_shell,echo $$(($(ASSUME_NPROC) >= $(MAX_NPROC)))),1)
CAPPED_NPROC := $(MAX_NPROC)
override nproc_string := >=$(MAX_NPROC) cores
endif

ifneq ($(ASSUME_RAM),)
ifeq ($(call safe_shell,echo $$(($(ASSUME_RAM) >= 64))),1)
override ram_string := >=64G RAM
# The default number of jobs. Override with `-jN` or `JOBS=N`, both work fine.
JOBS := $(CAPPED_NPROC)
# How many translation units to use for the bindings. Bigger value = less RAM usage, but usually slower build speed.
# When changing this, update the default value for `-j` above.
NUM_FRAGMENTS := $(CAPPED_NPROC)
else ifeq ($(call safe_shell,echo $$(($(ASSUME_RAM) >= 32))),1)
override ram_string := ~32G RAM
NUM_FRAGMENTS := $(call safe_shell,echo $$(($(CAPPED_NPROC) * 2)))# = CAPPED_NPROC * 2
JOBS := $(CAPPED_NPROC)
else ifeq ($(call safe_shell,echo $$(($(ASSUME_RAM) >= 16))),1)
# At this point we have so little RAM that we ignore nproc completely (or would need to clamp it to something like ~8, but who even has less cores than that?).
override ram_string := ~16G RAM
NUM_FRAGMENTS := 64
JOBS := 8
else
override ram_string := ~8G RAM (oof)
NUM_FRAGMENTS := 64
JOBS := 4
endif
else
override ram_string := unknown, assuming ~16G
NUM_FRAGMENTS := 64
JOBS := 8
endif
MAKEFLAGS += -j$(JOBS)
ifeq ($(filter-out file,$(origin NUM_FRAGMENTS) $(origin JOBS)),)
$(info Build machine: $(nproc_string), $(ram_string); defaulting to NUM_FRAGMENTS=$(NUM_FRAGMENTS) -j$(JOBS))
else
$(info Build machine: $(nproc_string), $(ram_string); NUM_FRAGMENTS=$(NUM_FRAGMENTS) -j$(JOBS))# This can print the wrong `-j` if you override it using `-j` instead of `JOBS=N`.
endif




# --- End of configuration variables.




.DELETE_ON_ERROR: # Delete output on command failure. Otherwise you'll get incomplete bindings.


# You can change this to something else to rename the module, to have it side-by-side with the legacy one.
PACKAGE_NAME := meshlib
MODULE_OUTPUT_DIR := $(MESHLIB_SHLIB_DIR)/$(PACKAGE_NAME)

INPUT_DIRS := $(addprefix $(makefile_dir)/../../source/,$(INPUT_PROJECTS)) $(makefile_dir)/extra_headers
INPUT_FILES_BLACKLIST := $(call load_file,$(makefile_dir)/input_file_blacklist.txt)
INPUT_FILES_WHITELIST := %
ifneq ($(IS_WINDOWS),)
TEMP_OUTPUT_DIR := source/TempOutput/PythonBindings/x64/$(VS_MODE)
else
TEMP_OUTPUT_DIR := build/binds
endif
INPUT_GLOBS := *.h
# Note that we're ignoring `operator<=>` in `mrbind_flags.txt` because it causes errors on VS2022:
# `undefined symbol: void __cdecl std::_Literal_zero_is_expected(void)`,
# `referenced by source/TempOutput/PythonBindings/x64/Release/binding.0.o:(public: __cdecl std::_Literal_zero::_Literal_zero<int>(int))`.
MRBIND_FLAGS := $(call load_file,$(makefile_dir)/mrbind_flags.txt)
MRBIND_FLAGS_FOR_EXTRA_INPUTS := $(call load_file,$(makefile_dir)/mrbind_flags_for_helpers.txt)
COMPILER_FLAGS := $(ABI_COMPAT_FLAG) $(EXTRA_CFLAGS) $(call load_file,$(makefile_dir)/common_compiler_parser_flags.txt) $(PYTHON_CFLAGS) -I. -I$(DEPS_INCLUDE_DIR) -I$(makefile_dir)/../../source
COMPILER_FLAGS_LIBCLANG := $(call load_file,$(makefile_dir)/parser_only_flags.txt)
# Need whitespace before `$(MRBIND_SOURCE)` to handle `~` correctly.
COMPILER := $(CXX_FOR_BINDINGS) $(subst $(lf), ,$(call load_file,$(makefile_dir)/compiler_only_flags.txt)) -I $(MRBIND_SOURCE)/include -I$(makefile_dir)
LINKER_OUTPUT := $(MODULE_OUTPUT_DIR)/mrmeshpy$(PYTHON_MODULE_SUFFIX)
LINKER := $(CXX_FOR_BINDINGS) -fuse-ld=lld
# Unsure if `-dynamiclib` vs `-shared` makes any difference on MacOS. I'm using the former because that's what CMake does.
LINKER_FLAGS := $(EXTRA_LDFLAGS) -L$(DEPS_LIB_DIR) $(PYTHON_LDFLAGS) -L$(MESHLIB_SHLIB_DIR) $(addprefix -l,$(INPUT_PROJECTS)) -lMRPython $(if $(IS_MACOS),-dynamiclib,-shared) $(call load_file,$(makefile_dir)/linker_flags.txt)
COMBINED_HEADER_OUTPUT := $(TEMP_OUTPUT_DIR)/combined.hpp
GENERATED_SOURCE_OUTPUT := $(TEMP_OUTPUT_DIR)/binding.cpp

# Those files are parsed and baked into the final bindings.
EXTRA_INPUT_SOURCES := $(makefile_dir)helpers.cpp
# Those files compiled as is and baked into the final bindings.
EXTRA_GENERATED_SOURCES := $(makefile_dir)aliases.cpp

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
COMPILER += -fvisibility=hidden
COMPILER_FLAGS += -fPIC
# Override Pybind ABI identifiers to force compatibility with `mrviewerpy` (which is compiled with some other compiler, but is also made to define those).
COMPILER_FLAGS += -DPYBIND11_COMPILER_TYPE='"_meshlib"' -DPYBIND11_BUILD_ABI='"_meshlib"'
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

# PCH.
PCH_IMPORT_FLAG :=
PCH_OBJECT :=
ifneq ($(ENABLE_PCH),)
COMPILED_PCH_FILE := $(TEMP_OUTPUT_DIR)/combined_pch.hpp.gch
PCH_IMPORT_FLAG := -include$(COMPILED_PCH_FILE:.gch=)
$(COMPILED_PCH_FILE): $(COMBINED_HEADER_OUTPUT)
	@echo $(call quote,[Compiling PCH] $@)
	@$(COMPILER) -o $@ -xc++-header $< $(COMPILER_FLAGS) $(PCH_CODEGEN_FLAGS)
# PCH object file, if enabled.
# We strip the include directories from the flags here, because Clang warns that those are unused.
ifneq ($(PCH_CODEGEN_FLAGS),)
PCH_OBJECT := $(TEMP_OUTPUT_DIR)/combined_pch.hpp.o
$(PCH_OBJECT): $(COMPILED_PCH_FILE)
	@echo $(call quote,[Compiling PCH object] $@)
	@$(filter-out -isystem% -I%,$(subst -isystem ,-isystem,$(subst -I ,-I,$(COMPILER) $(COMPILER_FLAGS)))) -c -o $@ $(COMPILED_PCH_FILE)
endif
endif


# Directories:
# Temporary output.
$(TEMP_OUTPUT_DIR):
	@mkdir -p $(call quote,$@)
# Module output.
$(MODULE_OUTPUT_DIR):
	@mkdir -p $(call quote,$@)

# The single header including all target headers.
override input_files := $(filter-out $(INPUT_FILES_BLACKLIST),$(filter $(INPUT_FILES_WHITELIST),$(call rwildcard,$(INPUT_DIRS),$(INPUT_GLOBS))))
$(COMBINED_HEADER_OUTPUT): $(input_files) | $(TEMP_OUTPUT_DIR)
	$(file >$@,#pragma once$(lf))
	$(foreach f,$(input_files),$(file >>$@,#include "$f"$(lf)))
ifneq ($(ENABLE_PCH),) # Additional headers to bake into the PCH. The condition is to speed up parsing a bit.
	$(file >>$@,#ifndef MR_PARSING_FOR_PB11_BINDINGS$(lf)#include <pybind11/pybind11.h>$(lf)#endif)
endif
# This version bakes the whole our `core.h` (which includes `<pybind11/pybind11.h>), but for some reason my measurements show it to be a tiny bit slower. Weird.
# $(file >>$@,#ifndef MR_PARSING_FOR_PB11_BINDINGS$(lf)#define MB_PB11_STAGE -1$(lf)#include MRBIND_HEADER$(lf)#undef MB_PB11_STAGE$(lf)#endif$(lf)) # Note temporarily setting `MB_PB11_STAGE=-1`, we don't want to bake any of the macros.

# The generated binding source.
# Note, this DOESN'T use the PCH, because the macros are different (PCH enables `-DMR_COMPILING_PB11_BINDINGS`, but this needs `-DMR_PARSING_FOR_PB11_BINDINGS`).
.PHONY: only-generate
only-generate: $(GENERATED_SOURCE_OUTPUT)
$(GENERATED_SOURCE_OUTPUT): $(COMBINED_HEADER_OUTPUT) | $(TEMP_OUTPUT_DIR)
	@echo $(call quote,[Generating] binding.cpp)
	@$(MRBIND_EXE) $(MRBIND_FLAGS) $(call quote,$<) -o $(call quote,$@) -- $(COMPILER_FLAGS_LIBCLANG) $(COMPILER_FLAGS)
# The object files for all fragments of the generated source.
override object_files := $(PCH_OBJECT) $(patsubst %,$(TEMP_OUTPUT_DIR)/binding.%.o,$(call safe_shell,bash -c $(call quote,echo {0..$(call safe_shell,bash -c 'echo $$(($(strip $(NUM_FRAGMENTS))-1))')})))
$(TEMP_OUTPUT_DIR)/binding.%.o: $(GENERATED_SOURCE_OUTPUT) $(COMPILED_PCH_FILE) | $(TEMP_OUTPUT_DIR)
	@echo $(call quote,[Compiling] $< (fragment $*))
	@$(COMPILER) $(call quote,$<) -c -o $(call quote,$@) $(COMPILER_FLAGS) $(PCH_IMPORT_FLAG) -DMB_NUM_FRAGMENTS=$(strip $(NUM_FRAGMENTS)) -DMB_FRAGMENT=$* $(if $(filter 0,$*),-DMB_DEFINE_IMPLEMENTATION)
# Generate and compile extra files.
override define extra_file_snippet =
$(call var,_generated := $(TEMP_OUTPUT_DIR)/binding_extra.$(basename $(notdir $1)).cpp)
$(call var,_object := $(_generated:.cpp=.o))
$(call var,object_files += $(_object))
only-generate: $(_generated)
$(_generated): $1 | $(TEMP_OUTPUT_DIR)
	@echo $(call quote,[Generating] $(notdir $(_generated)))
	@$(MRBIND_EXE) $(MRBIND_FLAGS_FOR_EXTRA_INPUTS) $(call quote,$1) -o $(call quote,$(_generated)) -- $(COMPILER_FLAGS_LIBCLANG) $(COMPILER_FLAGS)
$(_object): $(_generated) | $(TEMP_OUTPUT_DIR)
	@echo $(call quote,[Compiling] $(_generated))
	@$(COMPILER) $(call quote,$(_generated)) -c -o $(call quote,$(_object)) $(COMPILER_FLAGS)
endef
# Linking the primary module.
$(foreach s,$(EXTRA_INPUT_SOURCES),$(eval $(call extra_file_snippet,$s)))
# Compile extra files that don't need generating.
override define extra_pregen_file_snippet =
$(call var,_object := $(TEMP_OUTPUT_DIR)/$(notdir $(1:.cpp=.o)))
$(call var,object_files += $(_object))
$(_object): $1 | $(TEMP_OUTPUT_DIR)
	@echo $(call quote,[Compiling] $1)
	@$(COMPILER) $(call quote,$1) -c -o $(call quote,$(_object)) $(COMPILER_FLAGS)
endef
$(foreach s,$(EXTRA_GENERATED_SOURCES),$(eval $(call extra_pregen_file_snippet,$s)))
# Linking the primary module.
$(LINKER_OUTPUT): $(object_files) | $(MODULE_OUTPUT_DIR)
	@echo $(call quote,[Linking] $@)
	@$(LINKER) $^ -o $(call quote,$@) $(LINKER_FLAGS)




# Handwritten mrmeshnumpy. But only if we can't reuse it from the default build.
ifneq ($(BUILD_MRMESHNUMPY),)
MRMESHNUMPY_MODULE := $(MODULE_OUTPUT_DIR)/mrmeshnumpy$(PYTHON_MODULE_SUFFIX)
else
MRMESHNUMPY_MODULE :=
endif
ifneq ($(MRMESHNUMPY_MODULE),)
$(MRMESHNUMPY_MODULE): | $(MODULE_OUTPUT_DIR)
	@echo $(call quote,[Compiling] mrmeshnumpy)
	@$(COMPILER) \
		-o $@ \
		$(makefile_dir)/../../source/mrmeshnumpy/*.cpp \
		$(COMPILER_FLAGS) $(LINKER_FLAGS) \
		-DMRMESHNUMPY_PARENT_MODULE_NAME=$(PACKAGE_NAME)
endif

# The init script.
INIT_SCRIPT := $(MODULE_OUTPUT_DIR)/__init__.py
$(INIT_SCRIPT): $(makefile_dir)/__init__.py
	@cp $< $@
ifeq ($(IS_WINDOWS),) # If not on Windows, strip the windows-only part.
	@gawk -i inplace '/### windows-only: \[/{x=1} {if (!x) print} x && /### \]/{x=0}' $@
endif
ifeq ($(FOR_WHEEL),) # If not on building a wheel, strip the wheel-only part.
	@gawk -i inplace '/### wheel-only: \[/{x=1} {if (!x) print} x && /### \]/{x=0}' $@
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
ifneq ($(MRMESHNUMPY_MODULE),)
ALL_OUTPUTS += $(MESHLIB_SHLIB_DIR)/mrmeshnumpy$(PYTHON_MODULE_SUFFIX)
$(MESHLIB_SHLIB_DIR)/mrmeshnumpy$(PYTHON_MODULE_SUFFIX): $(MODULE_OUTPUT_DIR)/mrmeshnumpy$(PYTHON_MODULE_SUFFIX)
	@cp $< $@
endif
endif

# All modules.
.DEFAULT_GOAL := all
.PHONY: all
all: $(ALL_OUTPUTS)
