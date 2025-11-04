#!/bin/bash

set -e

if [ $# -lt 1 ]; then
  echo "Usage $0 <path-to-llvm-build-dir>"
  exit 1
fi

LLVM_BUILD_DIR=$(realpath -L $1)

for dir in program_similarity/*; do
  if [ -f $dir ]; then
    continue
  fi
  cd $dir
  mkdir -p build && cd build
  cmake .. -DLLVM_INSTALL_DIR=${LLVM_BUILD_DIR}
  make -j$(nproc)
  cd ../../../
done