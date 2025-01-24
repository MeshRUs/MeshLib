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

# A space.
override space := $(call) $(call)
# A comma.
override comma := ,

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

# Given a single number $1 = N, returns numbers from 0 to N-1 inclusive, space-separated.
# This is relatively expensive, so don't call this more than necessary.
override seq = $(call safe_shell,bash -c $(call quote,echo {0..$(call safe_shell,bash -c 'echo $$(($(strip $1)-1))')}))

# Enable double expansion of prerequisites. `$1_ObjectFiles` below needs this.
.SECONDEXPANSION:





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
$(info Those modules are for a Python wheel? $(if $(FOR_WHEEL),YES,NO))

# Build `libpybind11nonlimitedapi_meshlib_X.Y.so` shims automatically?
# You can always build them manually via `make shims -B`.
BUILD_SHIMS := $(FOR_WHEEL)
override BUILD_SHIMS := $(filter-out 0,$(BUILD_SHIMS))
$(info Build shims? $(if $(BUILD_SHIMS),YES,NO))

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
override valid_vs_modes := Debug Release
$(if $(filter-out $(valid_vs_modes),$(VS_MODE)),$(error Invalid `VS_MODE=$(VS_MODE)`, expected one of: $(valid_vs_modes)))

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
CXX_FOR_BINDINGS := $(HOMEBREW_DIR)/opt/llvm@$(strip $(file <$(makefile_dir)clang_version.txt))/bin/clang++
else
# Only on Ubuntu we don't want the default Clang version, as it can be outdated. Use the suffixed one.
CXX_FOR_BINDINGS := clang++-$(strip $(file <$(makefile_dir)clang_version.txt))
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
override EXTRA_CFLAGS += -Oz -flto=thin -DNDEBUG
override EXTRA_LDFLAGS += -Oz -flto=thin $(if $(IS_MACOS),,-s)# No `-s` on macos. It seems to have no effect, and the linker warns about it.
else ifeq ($(MODE),debug)
override EXTRA_CFLAGS += -g
override EXTRA_LDFLAGS += -g
else ifeq ($(MODE),none)
# Nothing.
else
$(error Unknown MODE=$(MODE))
endif
$(info MODE: $(MODE))

# Look for MeshLib dependencies relative to this. On Linux should point to the project root, because that's where `./include` and `./lib` are.
ifneq ($(IS_WINDOWS),)
DEPS_BASE_DIR := $(VCPKG_DIR)/installed/x64-windows-meshlib
DEPS_LIB_DIR := $(DEPS_BASE_DIR)/$(if $(filter Debug,$(VS_MODE)),debug/)lib
else
DEPS_BASE_DIR := .
DEPS_LIB_DIR := $(DEPS_BASE_DIR)/lib
endif
DEPS_INCLUDE_DIR := $(DEPS_BASE_DIR)/include

# The list of Python versions, in the format `X.Y`.
# When setting this manually, both spaces and commas work as separators.
ifneq ($(IS_WINDOWS),)
override localappdata := $(subst \,/,$(LOCALAPPDATA))
ifneq ($(FOR_WHEEL),)
# On Windows wheel we use all versions we can find in appdata.
PYTHON_VERSIONS := $(patsubst $(localappdata)/Programs/Python/Python3%,3.%,$(filter $(localappdata)/Programs/Python/Python3%,$(wildcard $(localappdata)/Programs/Python/Python3*)))
else
# On Windows non-wheel we detect only one version by default, the one in pkg-config.
PYTHON_VERSIONS := $(patsubst python-%-embed,%,$(basename $(notdir $(lastword $(sort $(wildcard $(DEPS_BASE_DIR)/lib/pkgconfig/python-*-embed.pc))))))
endif
else
PYTHON_VERSIONS := $(patsubst python-%-embed,%,$(call safe_shell,pkg-config --list-all | grep -Po 'python-\d.\d+-embed' | sort))
endif
$(if $(PYTHON_VERSIONS),,$(error Unable to guess the Python versions))
# Adjust commas to spaces for easier manual setup.
override PYTHON_VERSIONS := $(subst $(comma), ,$(PYTHON_VERSIONS))
# Sort the versions.
override PYTHON_VERSIONS := $(call safe_shell,echo $(call quote,$(PYTHON_VERSIONS)) | tr ' ' '\n' | sort -V)
$(info Targeting Python versions: $(PYTHON_VERSIONS)$(if $(BUILD_SHIMS),, (doesn't matter because we're not building shims)))

# Pick a Python version to use the headers from. Normally it doesn't matter which one we use, since defining `Py_LIMITED_API` should make them almost equivalent.
# Here we use the smallest version from `PYTHON_VERSIONS`.
PYTHON_HEADER_VERSION := $(firstword $(PYTHON_VERSIONS))
$(info Using Python headers from version: $(PYTHON_HEADER_VERSION))

# The min Python version to support. We use this to compute the value of the `Py_LIMITED_API` macro.
# It's probably a good idea to have this match the value in `source/MRPython/CMakeLists.txt`.
PYTHON_MIN_VERSION := 3.8
override python_min_version_hex := 0x$(call safe_shell,printf "%02x%02x" $(subst ., ,$(PYTHON_MIN_VERSION)))00f0
$(info Python min version: $(PYTHON_MIN_VERSION) (Py_LIMITED_API=$(python_min_version_hex)))

# Python compilation flags.
# Normally we guess the flags using pkg-config, in an OS-dependent way.
# You can override this by setting `PYTHON_CFLAGS` and `PYTHON_LDFLAGS`,
#   the values of which should contain `@X.Y@` which will be replaced with the `X.Y` Python version, or `@XY@` for the two numbers concatenated.
# Obtain the resulting flags by calling `$(call get_python_cflags,3.10)` (and similarly for ldflags).
PYTHON_CFLAGS :=
PYTHON_LDFLAGS :=
ifneq ($(and $(IS_WINDOWS),$(FOR_WHEEL)),)
# On Windows wheel, hardcode the flags to point to appdata.
PYTHON_CFLAGS := -I$(localappdata)/Programs/Python/Python@XY@/Include
PYTHON_LDFLAGS := -L$(localappdata)/Programs/Python/Python@XY@/libs -lpython@XY@
endif

ifeq ($(PYTHON_CFLAGS)$(PYTHON_LDFLAGS),) # If no custom flags are specified...
ifneq ($(IS_WINDOWS),)
# Intentionally using non-debug Python even in Debug builds, to mimic what MeshLib does. Unsure why we do this.
override get_python_cflags = $(call safe_shell,PKG_CONFIG_PATH=$(call quote,$(DEPS_BASE_DIR)/lib/pkgconfig) PKG_CONFIG_LIBDIR=- pkg-config --cflags python-$1-embed)
override get_python_ldflags = $(call safe_shell,PKG_CONFIG_PATH=$(call quote,$(DEPS_BASE_DIR)/lib/pkgconfig) PKG_CONFIG_LIBDIR=- pkg-config --libs python-$1-embed) -L$(DEPS_BASE_DIR)/lib
else # Linux or MacOS:
override get_python_cflags = $(call safe_shell,pkg-config --cflags python-$1-embed)
ifneq ($(IS_MACOS),)
# On MacOS we don't link Python, instead we use `-Xlinker -undefined -Xlinker dynamic_lookup` to avoid the errors.
# This is important to avoid segfaults when importing the wheel.
override get_python_ldflags =
else # not MacOS:
override get_python_ldflags = $(call safe_shell,pkg-config --libs python-$1-embed)
endif # not MacOS
endif # Linux or MacOS
else # if using custom flags
override get_python_cflags = $(subst @XY@,$(subst .,,$1),$(subst @X.Y@,$1,$(PYTHON_CFLAGS)))
override get_python_ldflags = $(subst @XY@,$(subst .,,$1),$(subst @X.Y@,$1,$(PYTHON_LDFLAGS)))
endif # using custom flags


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

# Shared library naming pattern.
ifneq ($(IS_WINDOWS),)
SHIM_SHLIB_NAMING := %.dll
else
ifneq ($(IS_MACOS),)
# For consistency with the shims that CMake builds.
SHIM_SHLIB_NAMING := lib%.dylib
else
SHIM_SHLIB_NAMING := lib%.so
endif
endif
$(info Shared library name pattern: $(SHIM_SHLIB_NAMING))


# Enable PCH.
# Not all modules we build use PCHs even if this is true. But if this is false, PCHs are disabled for all modules.
ENABLE_PCH := 1
override ENABLE_PCH := $(filter-out 0,$(ENABLE_PCH))

# Those are passed when compiling the PCH. Can be empty.
# If this is non-empty and has any flags other than `-fpch-instantiate-templates`, we compile an additional `.o` for the PCH and link it to the result.
# (And building this `.o` even when it's not needed doesn't seem to cause any issues.)
# There are three flags that can be used in any combination here: `-fpch-codegen -fpch-debuginfo -fpch-instantiate-templates`.
# `-fpch-codegen` seems to be buggy, causing weird errors: undefined references to libfmt/gtest/some other functions when used with `-fpch-instantiate-templates`,
#   or weird overload resolution errors during compilation without that.
PCH_CODEGEN_FLAGS := -fpch-debuginfo -fpch-instantiate-templates



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

# You can change this to something else to rename the module, to have it side-by-side with the legacy one.
PACKAGE_NAME := meshlib




# --- The list of modules:
MODULES :=

MODULES += mrmeshpy
mrmeshpy_InputProjects := MRMesh MRIOExtras MRSymbolMesh MRVoxels
mrmeshpy_ExtraMrbindFlags := --allow MR
mrmeshpy_EnablePch := 1
mrmeshpy_ExtraInputDirs := $(makefile_dir)extra_headers
mrmeshpy_NumFragments := $(NUM_FRAGMENTS)
# Those files are parsed and baked into the final bindings.
mrmeshpy_ExtraInputFiles := $(makefile_dir)helpers.cpp
# Those files are compiled as is and linked into the final bindings.
mrmeshpy_ExtraSourceFiles := $(makefile_dir)aliases.cpp

# Include the `MRCuda` project?
# Defaults to 0 on Mac (no Cuda there!), and 1 elsewhere. Can set to 0 if you don't have Cuda installed.
ENABLE_CUDA := $(if $(IS_MACOS),0,1)
override ENABLE_CUDA := $(filter-out 0,$(ENABLE_CUDA))
$(info Enable Cuda: $(if $(ENABLE_CUDA),YES,NO))
ifneq ($(ENABLE_CUDA),)
MODULES += mrcudapy
mrcudapy_InputProjects := MRCuda
mrcudapy_ExtraMrbindFlags := --allow MR::Cuda
mrcudapy_DependsOn := $(PACKAGE_NAME).mrmeshpy
endif





# ----------- End of configuration variables.




.DELETE_ON_ERROR: # Delete output on command failure. Otherwise you'll get incomplete bindings.

MODULE_OUTPUT_DIR := $(MESHLIB_SHLIB_DIR)/$(PACKAGE_NAME)

INPUT_FILES_BLACKLIST := $(call load_file,$(makefile_dir)input_file_blacklist.txt)
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
MRBIND_FLAGS := $(call load_file,$(makefile_dir)mrbind_flags.txt)
MRBIND_FLAGS_FOR_EXTRA_INPUTS := $(call load_file,$(makefile_dir)mrbind_flags_for_helpers.txt)
COMPILER_FLAGS := $(ABI_COMPAT_FLAG) $(EXTRA_CFLAGS) $(call load_file,$(makefile_dir)common_compiler_parser_flags.txt) -I. -I$(DEPS_INCLUDE_DIR) -I$(makefile_dir)../../source
COMPILER_FLAGS_LIBCLANG := $(call load_file,$(makefile_dir)parser_only_flags.txt)
# Need whitespace before `$(MRBIND_SOURCE)` to handle `~` correctly.
COMPILER := $(CXX_FOR_BINDINGS) $(subst $(lf), ,$(call load_file,$(makefile_dir)compiler_only_flags.txt)) -I $(MRBIND_SOURCE)/include -I$(makefile_dir)
LINKER := $(CXX_FOR_BINDINGS) -fuse-ld=lld
# Unsure if `-dynamiclib` vs `-shared` makes any difference on MacOS. I'm using the former because that's what CMake does.
# No $(PYTHON_LDFLAGS) here, that's only for our patched Pybind library.
LINKER_FLAGS := $(EXTRA_LDFLAGS) -L$(DEPS_LIB_DIR) -L$(DEPS_BASE_DIR)/lib -L$(MESHLIB_SHLIB_DIR) $(addprefix -l,$(INPUT_PROJECTS)) -lMRPython $(if $(IS_MACOS),-dynamiclib,-shared) $(call load_file,$(makefile_dir)linker_flags.txt)

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
COMPILER_FLAGS += -isystem $(makefile_dir)../../thirdparty/pybind11/include
COMPILER_FLAGS += -isystem $(makefile_dir)../../thirdparty/parallel-hashmap
COMPILER_FLAGS += -D_DLL -D_MT
# Only seems to matter on VS2022 and not on VS2019, for some reason.
COMPILER_FLAGS += -DNOMINMAX
COMPILER_FLAGS += -D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS
# Don't export Pybind exceptions. This works around Clang bug: https://github.com/llvm/llvm-project/issues/118276
# And I'm not sure if exporting them even did anything useful on Windows in the first place.
COMPILER_FLAGS += -DPYBIND11_EXPORT_EXCEPTION=
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
rpath_origin := $(if $(IS_MACOS),@loader_path,$$$$ORIGIN)
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

# Directories:
# Temporary output.
$(TEMP_OUTPUT_DIR):
	@mkdir -p $(call quote,$@)
# Module output.
$(MODULE_OUTPUT_DIR):
	@mkdir -p $(call quote,$@)


override all_outputs :=

# Things for our patched pybind: --- [
# Also setting `Py_LIMITED_API` is a part of this, but it's spread all over this makefile.
COMPILER += -DPYBIND11_NONLIMITEDAPI_LIB_SUFFIX_FOR_MODULE='"meshlib"'
# Pybind normally sets this to 5 in Python 3.12 and newer, and to 4 before that. But we need the same number everywhere for our modules to work on
#   multiple different Python versions. We can't set it to 4 (since that's not compatible with the new Python, see https://github.com/pybind/pybind11/pull/4570),
#   but we can set it to 5 unconditionally (Pybind doesn't do it by default only for ABI compatibility).
COMPILER += -DPYBIND11_INTERNALS_VERSION=5
# ] --- end things for our patched pybind

# Build the library files for different Python versions, for our patched Pybind.
PYBIND_LIBS_OUTPUT_DIR := $(MODULE_OUTPUT_DIR)
PYBIND_SOURCE_DIR := $(makefile_dir)../../thirdparty/pybind11
PYBIND_NONLIMITEDAPI_CPP := $(PYBIND_SOURCE_DIR)/source/non_limited_api/non_limited_api.cpp
PYBIND_NONLIMITEDAPI_LIB_NAME_PREFIX := pybind11nonlimitedapi_meshlib_

override shim_outputs :=
$(foreach v,$(PYTHON_VERSIONS),\
    $(call var,_obj := $(TEMP_OUTPUT_DIR)/$(PYBIND_NONLIMITEDAPI_LIB_NAME_PREFIX)$v.o)\
    $(call var,_shlib := $(PYBIND_LIBS_OUTPUT_DIR)/$(patsubst %,$(SHIM_SHLIB_NAMING),$(PYBIND_NONLIMITEDAPI_LIB_NAME_PREFIX)$v))\
    $(call var,shim_outputs += $(_shlib))\
    $(eval $(_obj): $(PYBIND_NONLIMITEDAPI_CPP) | $(TEMP_OUTPUT_DIR) ; @echo $(call quote,[Compiling Pybind shim] $(_obj)) && $(COMPILER) $(call get_python_cflags,$v) $(COMPILER_FLAGS) $$< -c -o $$@)\
    $(eval $(_shlib): $(_obj) ; @echo $(call quote,[Linking Pybind shim] $(_shlib)) && $(LINKER) $$^ -o $$@ $(call get_python_ldflags,$v) $(LINKER_FLAGS) -lpybind11nonlimitedapi_stubs)\
)
.PHONY: shims
shims: $(shim_outputs)
ifneq ($(BUILD_SHIMS),)
override all_outputs += $(shim_outputs)
endif

# Those are used by `module_snippet` below.
.PHONY: only-generate

# This code segment is repeated for every module. $1 is the module name.
override define module_snippet =
# Which directories we search for headers.
$(call var,_input_dirs := $(addprefix $(makefile_dir)../../source/,$($1_InputProjects)) $($1_ExtraInputDirs))
# Input headers.
$(call var,_input_files := $(filter-out $(INPUT_FILES_BLACKLIST),$(filter $(INPUT_FILES_WHITELIST),$(call rwildcard,$(_input_dirs),$(INPUT_GLOBS)))))

# Unset the PCH flag on the module if PCHs are disabled globally.
$(if $(ENABLE_PCH),,$(call var,$1_EnablePch :=))

# Set the default number of fragments, if not specified.
$(if $($1_NumFragments),,$(call var,$1_NumFragments := 1))

# Pick whatever Python version for get_python_cflags, it shouldn't matter.
$(call var,$1_CompilerFlagsPython := -DPy_LIMITED_API=$(python_min_version_hex) $(call get_python_cflags,$(PYTHON_HEADER_VERSION)))

# Compiler + compiler-only flags, adjusted per module. Don't use those for parsing.
$(call var,$1_CompilerFlagsFixed := $($1_CompilerFlagsPython) $(COMPILER_FLAGS) -DMB_PB11_MODULE_NAME=$1 $(if $($1_DependsOn),-DMB_PB11_MODULE_DEPS=$(call quote,$(subst $(space),$(comma),$(patsubst %,"%",$($1_DependsOn))))))

# Produce the one combined header including all our input headers.
# And if PCH is enabled, this also includes the headers to bake.
$(call var,$1__CombinedHeaderOutput := $(TEMP_OUTPUT_DIR)/$1.combined.hpp)
$($1__CombinedHeaderOutput): $(_input_files) | $(TEMP_OUTPUT_DIR)
	$$(file >$$@,#pragma once$$(lf))
	$$(foreach f,$(_input_files),$$(file >>$$@,#include "$$f"$$(lf)))
	$(call,### Additional headers to bake into the PCH. The condition is to speed up parsing a bit.)
	$$(if $($1_EnablePch),$$(file >>$$@,#ifndef MR_PARSING_FOR_PB11_BINDINGS$$(lf)#include <pybind11/pybind11.h>$$(lf)#endif))
	$(call,### This alternative version bakes the whole our `core.h` [which includes `<pybind11/pybind11.h>], but for some reason my measurements show it to be a tiny bit slower. Weird.)
	$(call,###   #ifndef MR_PARSING_FOR_PB11_BINDINGS$(lf)#define MB_PB11_STAGE -1$(lf)#include MRBIND_HEADER$(lf)#undef MB_PB11_STAGE$(lf)#endif$(lf))
	$(call,### Note temporarily setting `MB_PB11_STAGE=-1`, we don't want to bake any of the macros.)

# Run the parser.
# Note, this DOESN'T use the PCH, because the macros are different (PCH enables `-DMR_COMPILING_PB11_BINDINGS`, but this needs `-DMR_PARSING_FOR_PB11_BINDINGS`).
$(call var,$1__ParserSourceOutput := $(TEMP_OUTPUT_DIR)/$1.generated.cpp)
only-generate: $($1__ParserSourceOutput)
$($1__ParserSourceOutput): $($1__CombinedHeaderOutput) | $(TEMP_OUTPUT_DIR)
	@echo $(call quote,[$1] [Generating] $($1__ParserSourceOutput))
	@$(MRBIND_EXE) $(MRBIND_FLAGS) $($1_ExtraMrbindFlags) $$(call quote,$$<) -o $$(call quote,$$@) -- $(COMPILER_FLAGS_LIBCLANG) $(COMPILER_FLAGS) $($1_CompilerFlagsPython)

# Compile the PCH.
$(call var,$1__BakedPch :=)
$(call var,$1__PchImportFlag :=)
$(call var,$1__PchObject :=)
$(if $($1_EnablePch),\
  $(call var,$1__BakedPch := $(TEMP_OUTPUT_DIR)/$1.combined_pch.hpp.gch)\
  $(call var,$1__PchImportFlag := -include$($1__BakedPch:.gch=))\
  \
  $($1__BakedPch): $($1__CombinedHeaderOutput) ; @echo $(call quote,[$1] [Compiling PCH] $($1__BakedPch)) && $(COMPILER) -o $$@ -xc++-header $$< $($1_CompilerFlagsFixed) $(PCH_CODEGEN_FLAGS)\
)
# PCH object file, if enabled.
# We strip the include directories from the flags here, because Clang warns that those are unused.
$(if $(and $($1_EnablePch),$(filter-out -fpch-instantiate-templates,$(PCH_CODEGEN_FLAGS))),\
  $(call var,$1__PchObject := $(TEMP_OUTPUT_DIR)/$1.combined_pch.hpp.o)\
  \
  $($1__PchObject): $($1__BakedPch) ; @echo $(call quote,[$1] [Compiling PCH object] $($1__PchObject)) && $(filter-out -isystem% -I%,$(subst -isystem ,-isystem,$(subst -I ,-I,$(COMPILER) $($1_CompilerFlagsFixed)))) -c -o $$@ $($1__BakedPch)\
)

# Compile N object files (fragments) from the generated source.
$(TEMP_OUTPUT_DIR)/$1.fragment.%.o: $($1__ParserSourceOutput) $($1__BakedPch) | $(TEMP_OUTPUT_DIR)
	@echo $$(call quote,[$1] [Compiling] $$< (fragment $$*))
	@$(COMPILER) $$(call quote,$$<) -c -o $$(call quote,$$@) $($1_CompilerFlagsFixed) $($1__PchImportFlag) -DMB_NUM_FRAGMENTS=$(strip $($1_NumFragments)) -DMB_FRAGMENT=$$* $$(if $$(filter 0,$$*),-DMB_DEFINE_IMPLEMENTATION)

# A list of all object files.
# NOTE: This is amended later, so we must refer to it lazily.
$(call var,$1__ObjectFiles := $(patsubst %,$(TEMP_OUTPUT_DIR)/$1.fragment.%.o,$(call seq,$($1_NumFragments))))

# Link the module.
# Have to evaluate `$1_ObjectFiles` lazily to observe the later updates to it. This also relies on `.SECONDEXPANSION`.
$(call var,$1__LinkerOutput := $(MODULE_OUTPUT_DIR)/$1$(PYTHON_MODULE_SUFFIX))
$(call var,all_outputs += $($1__LinkerOutput))
$($1__LinkerOutput): $$$$($1__ObjectFiles) | $(MODULE_OUTPUT_DIR)
	@echo $$(call quote,[$1] [Linking] $$@)
	@$(LINKER) $$^ -o $$(call quote,$$@) $(LINKER_FLAGS) $(addprefix -l,$($1_InputProjects) pybind11nonlimitedapi_stubs)

# A pretty target.
.PHONY: $1
$1: $($1__LinkerOutput)

endef
$(foreach x,$(MODULES),$(eval $(call module_snippet,$x)))

# This snippet parses and compiles an extra file.
# $1 is the module name, $2 is the input file name.
override define extra_file_snippet =
$(call var,_generated := $(TEMP_OUTPUT_DIR)/$1.extra.$(basename $(notdir $2)).cpp)
$(call var,_object := $(_generated:.cpp=.o))
$(call var,$1__ObjectFiles += $(_object))
only-generate: $(_generated)
$(_generated): $2 | $(TEMP_OUTPUT_DIR)
	@echo $(call quote,[$1] [Generating] $(notdir $(_generated)))
	@$(MRBIND_EXE) $(MRBIND_FLAGS_FOR_EXTRA_INPUTS) $(call quote,$2) -o $(call quote,$(_generated)) -- $(COMPILER_FLAGS_LIBCLANG) $(COMPILER_FLAGS) $($1_CompilerFlagsPython)
$(_object): $(_generated) | $(TEMP_OUTPUT_DIR)
	@echo $(call quote,[$1] [Compiling] $(_generated))
	@$(COMPILER) $(call quote,$(_generated)) -c -o $(call quote,$(_object)) $($1_CompilerFlagsFixed)
endef
$(foreach x,$(MODULES),$(foreach y,$($x_ExtraInputFiles),$(eval $(call extra_file_snippet,$x,$y))))

# This snippet compiles an extra file as is, without passing it through the parser/generator.
# $1 is the module name, $2 is the input file name.
override define extra_pregen_file_snippet =
$(call var,_object := $(TEMP_OUTPUT_DIR)/$1.custom.$(notdir $(1:.cpp=.o)))
$(call var,$1__ObjectFiles += $(_object))
$(_object): $2 | $(TEMP_OUTPUT_DIR)
	@echo $(call quote,[$1] [Compiling] $2)
	@$(COMPILER) $(call quote,$2) -c -o $(call quote,$(_object)) $($1_CompilerFlagsFixed)
endef
$(foreach x,$(MODULES),$(foreach y,$($x_ExtraSourceFiles),$(eval $(call extra_pregen_file_snippet,$x,$y))))



# The init script.
INIT_SCRIPT := $(MODULE_OUTPUT_DIR)/__init__.py
$(INIT_SCRIPT): $(makefile_dir)__init__.py
	@cp $< $@
ifeq ($(IS_WINDOWS),) # If not on Windows, strip the windows-only part.
	@gawk -i inplace '/### windows-only: \[/{x=1} {if (!x) print} x && /### \]/{x=0}' $@
endif
ifeq ($(FOR_WHEEL),) # If not on building a wheel, strip the wheel-only part.
	@gawk -i inplace '/### wheel-only: \[/{x=1} {if (!x) print} x && /### \]/{x=0}' $@
endif
override all_outputs += $(INIT_SCRIPT)

# Copying modules next to the exe on Windows.
# I don't think this is actually needed, since we erase them when creating the installer.
# That was originally done because it's hard to make VS build them directly in the correct directory,
#   so some of our scripts look for them outside of `meshlib/`. Probably a good idea to fix that and not copy here at all.
ifneq ($(IS_WINDOWS),)
override all_outputs += $(MESHLIB_SHLIB_DIR)/__init__.py
$(MESHLIB_SHLIB_DIR)/__init__.py: $(INIT_SCRIPT)
	@cp $< $@
override modules_copied_to_bin_dir := $(patsubst %,,$(MODULES))
$(foreach m,$(MODULES),\
	$(call var,_in := $(MODULE_OUTPUT_DIR)/$m$(PYTHON_MODULE_SUFFIX))\
	$(call var,_out := $(MESHLIB_SHLIB_DIR)/$m$(PYTHON_MODULE_SUFFIX))\
	$(call var,all_outputs += $(_out))\
	$(eval $(_out): $(_in) ; @cp $(_in) $(_out))\
)
endif



# All modules.
.DEFAULT_GOAL := all
.PHONY: all
all: $(all_outputs)
