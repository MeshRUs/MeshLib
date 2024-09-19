#!/bin/bash

# Downloads the source code for MRBind and builds it at `~/mrbind/build`.

set -euxo pipefail

MRBIND_DIR=~/mrbind

# Read the Clang version from `preferred_clang_version.txt`. `xargs` trims the whitespace.
# Some versions of MacOS seem to lack `realpath`, so not using it here.
SCRIPT_DIR="$(dirname "$BASH_SOURCE")"
CLANG_VER="$(cat $SCRIPT_DIR/preferred_clang_version.txt | xargs)"
[[ $CLANG_VER ]] || (echo "Not sure what version of Clang to use." && false)

# Clone mrbind, or pull the latest version.
# We don't install our own Git for this, because there's an official installer and the Brew package,
#   and I'm unsure what to choose. The user can choose that themselves.
if [[ -d $MRBIND_DIR ]]; then
    cd "$MRBIND_DIR"
    git checkout master
    git pull
else
    git clone https://github.com/MeshInspector/mrbind "$MRBIND_DIR"
    cd "$MRBIND_DIR"
fi

rm -rf build

# Add `make` to PATH.
export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"
# Add Clang to PATH.
# I'm not entirely sure why this directory sometimes doesn't exit. It exists on the Mac I tested on, but not on our github runner. Hmm.
if [[ -d "/opt/homebrew/opt/llvm@$CLANG_VER/bin" ]]; then
    export PATH="/opt/homebrew/opt/llvm@$CLANG_VER/bin:$PATH"
else
    export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
fi

CC=clang CXX=clang++ cmake -B build
cmake --build build -j4
