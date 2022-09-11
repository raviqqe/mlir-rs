#!/bin/sh

set -e

# TODO Use Homebrew to install LLVM 15.
# llvm_version=15

# brew update
# brew install llvm@$llvm_version

curl -fsSL https://apt.llvm.org/llvm.sh | sudo bash -s 15 all
sudo apt -y install libmlir-15-dev mlir-15-tools
echo PATH=/usr/lib/llvm-15/bin:$PATH >>$GITHUB_ENV

if [ -n "$GITHUB_ENV" ]; then
  echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
fi
