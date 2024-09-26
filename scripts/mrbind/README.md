# Generating the bindings

[Installing prerequisites](#installing-prerequisites) — [Generating bindings](#generating-bindings) — [Troubleshooting](#troubleshooting)

## Installing prerequisites

Run **`scripts/mrbind/install_deps_<platform>`** to install the dependencies (on Linux and MacOS - as root), then **`scripts/mrbind/install_mrbind_<platform>`** to build MRBind (not at root).

You can re-run those scripts to update the dependencies and/or MRBind itself.

There's no rocket science in those, and you can do it manually instead of running the scripts (especially building MRBind yourself), see below.

Among other things, the scripts can do following:

* On Linux and MacOS, create `~/mrbind` to build MRBind in.

* On Ubuntu, add [the LLVM repository](https://apt.llvm.org/) to install the latest Clang and libclang from.

* On Windows, install MSYS2 to `C:\msys64_meshlib_mrbind`.

* On Ubuntu 20.04, build GNU Make from source and install it to `/usr/local`, because the default one is outdated.

**More details and manual instructions:**

<details><summary><b>Windows</b></summary>

* **Installing dependencies:**

    On Windows we use MSYS2, because it provides prebuilt libclang and provides GNU Make to run our makefile.

    MSYS2 is a package manager, roughly speaking. They provide a bunch of MinGW-related packages (compilers and prebuilt libraries). Luckily Clang can always cross-compile, so MSYS2's MinGW Clang can produce MSVC-compatible executables with the correct flags. You still need to have VS installed though, since it will use its libraries.

    We use [MSYS2 CLANG64](https://www.msys2.org/docs/environments/) environment. Consult `install_deps_windows_msys2.bat` for the list of packages we install in it.

    Notably on Windows we don't have control over the Clang version, since MSYS2 only supports installing the latest one.

* **Building MRBind:**

    MRBind source code is at https://github.com/MeshInspector/mrbind/.

    We build MRBind at `C:\msys64_meshlib_mrbind\home\username\mrbind`, but you can build it elsewhere manually.

    We build in [MSYS2 CLANG64](https://www.msys2.org/docs/environments/) environment, using MSYS2's Clang. Other compilers are not guaranteed to work.


</details>

<details><summary><b>Ubuntu</b></summary>

* **Installing dependencies:**

    We want a certain version of Clang (see `preferred_clang_version.txt`), and since older versions of Ubuntu don't have it, we add Clang's repository: https://apt.llvm.org

    Also we need a certain version of GNU Make, so on Ubuntu 20.04 we build it from source and install to `/usr/local`.

    And obviously we install some packages, see `install_deps_ubuntu.sh` for the list.

* **Building MRBind:**

    MRBind source code is at https://github.com/MeshInspector/mrbind/.

    We build MRBind at `~/mrbind`, but you can build it elsewhere manually.

    You might want to pass `-DClang_DIR=/usr/lib/cmake/clang-VERSION` (where `VERSION` is the one mentioned in `preferred_clang_version.txt`) if you have several versions of libclang installed, because otherwise CMake might pick an arbitrary one (apparently it picks the first one returned by globbing `clang-*`, which might not be the latest one).

    Use `CC=clang-VERSION CXX=clang++-VERSION cmake ....` to build using Clang. Other compilers might work, but that's not guaranteed.

</details>

<details><summary><b>MacOS</b></summary>

* **Installing dependencies:**

    Homebrew must already be installed.

    We install a certain version of Clang and libclang from it (see `preferred_clang_version.txt`), and also GNU Make and Gawk. MacOS has its own Make, but it's outdated. It seems to have Gawk, but we install our own just in case.

    What we install from Brew is the regular Clang, not Apple Clang (Apple's fork for Clang), because that is based on an outdated branch of Clang.

    You must run following to add the installed things to your PATH. On Arm Macs:
    ```sh
    export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"
    export PATH="/opt/homebrew/opt/llvm/bin@VERSION:$PATH" # See the correct VERSION in `preferred_clang_version.txt`.
    ```
    And on x86 Macs the installation directory seems to be `/usr/local/...` instead of `/opt/homebrew/...`.

* **Building MRBind:**

    MRBind source code is at https://github.com/MeshInspector/mrbind/.

    We build MRBind at `~/mrbind`, but you can build it elsewhere manually.

    Make sure your PATH is correct, as explained in the previous step.

    Use `CC=clang-VERSION CXX=clang++-VERSION cmake ....` to build using the Clang we've installed. It might build using Apple Clang as well (if you don't set your PATH as explained above), but that's not guaranteed to work. (If you want to try it, you must pass `-DCMAKE_PREFIX_PATH=/opt/homebrew/opt/llvm` for it to find our libclang.)

</details>

## Generating bindings

First, build MeshLib as usual.

Then generate the bindings:
* **On Windows:** `scripts/mrbind/generate_win.bat -B --trace MODE=none` from VS developer command prompt (use the `x64 Native` one!).

  This will look for MeshLib in `./source/x64/Release` (so the current directory matters). Add `VS_MODE=Debug` at the end if you built MeshLib in debug mode.

  The `generate_win.bat` file merely calls `generate.mk` (see below) inside of MSYS2 shell. You can use that directly if you want.

* **On Linux:** `make -f scripts/mrbind/generate.mk -B --trace MODE=none`

  This will look for MeshLib in `./build/Release/bin`. Pass `MESHLIB_SHLIB_DIR=path/to/bin` for a different directory.

* **On MacOS:** Same as on Linux, but before that must adjust the PATH. On Arm Macs: `export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"`, and on x86 Macs `/usr/local/...` instead of `/opt/homebrew/...`. This adds the version of Make installed in Homebrew to PATH, because the default one is outdated. Confirm the version with `make --version`, must be 4.x or newer.

### Some common flags:

* **`--trace` — enable verbose logs.**

* **`MODE=none` — disable optimization** for faster build times. The default is `MODE=release`. To enable debug symbols, use `MODE=debug`. To set completely custom compiler flags, set `EXTRA_CFLAGS` and `EXTRA_LDFLAGS`.

* **`-B` — force a full rebuild of the bindings.** Incremental builds are not very useful, because they're not perfect and can miss changes. Use incremental builds e.g. when you're fixing linker errors.

* **`NUM_FRAGMENTS=??` — adjust RAM usage vs build speed tradeoff.** `4` is the default, good for 16 GB of RAM. Use `2` if you have 32 GB of RAM to build ~2 times faster. Less fragments = faster builds but more RAM usage.

* **`PYTHON_PKGCONF_NAME=python-3.??-embed` — select Python version.** We try to guess this one. You can set this to `python3-embed` to use whatever the OS considers to be the default version.

### Selecting the compiler:

For simplicity, we compile the bindings with the same Clang that we use for parsing the code. (Consult `preferred_clang_version.txt` for the current version.) But you can override this using `CXX_FOR_BINDINGS`.

**ABI compatibility:** Since MeshLib is compiled using a different compiler, we must ensure the two use the same ABI. `CXX_FOR_ABI` should be set to the compiler the ABI of which we're trying to match. (Defaults to `CXX` environment variable, or `g++` if not set.) At the moment, if `CXX_FOR_ABI` is GCC 13 or older or Clang 17 or older (note that Apple Clang uses a [different version numbering scheme](https://en.wikipedia.org/wiki/Xcode#Xcode_15.0_-_(since_visionOS_support)_2)), we pass `-fclang-abi-compat=17` to our Clang 18 or newer. This flag *disables* mangling `requires` constraints into function names. If we guess incorrectly, you'll get undefined references to functions with `requires` constraints on them.

### Less common flags:

* **Selecting MRBind installation:** if you installed MRBind to a non-default location (`~/mrbind` on Linux and MacOS, `C:\msys64_meshlib_mrbind\home\username\mrbind` on Windows), you must pass this location to `MRBIND_SOURCE=path/to/mrbind`.

    Additionally, if the MRBind binary is not at `$MRBIND_SOURCE/build/mrbind`, you must pass `MRBIND_EXE=...` (path to the executable itself, not its directory).

You can find some undocumented flags/variables in `generate.mk`.

## Troubleshooting

* **`could not open 'MRMesh.lib': No such file or directory`**

  * MeshLib wasn't built, or `VS_MODE` is set incorrectly.

* **`machine type x86 conflicts with x64`**

  * You opened `x86 ...` VS developer command prompt, but we need `x64 Native`. Rebuild the bindings in x64 prompt.

* **`undefined symbol: void __cdecl std::_Literal_zero_is_expected(void)`**

  * Update your VS 2022.

* **Importing the wheel segfaults on MacOS**

  * Make sure you're not linking against Python **and** do use `-Xlinker -undefined -Xlinker dynamic_lookup` linker flags. The `generate.mk` should already do it correctly, just keep this in mind. Also transitively linking Python seems to be fine (`-lMRPython` is fine).

    Failure to do this will have no effect when importing the module directly, but will segfault when importing it as a wheel, **or** when using a wrong Python version even without the wheel.
