#!/bin/sh

set -e

llvm_version=17

if [ -n "$CI" ]; then
  brew install --overwrite python@3.11
fi

brew install llvm@$llvm_version

echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
