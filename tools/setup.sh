#!/bin/sh

set -e

llvm_version=14

brew update
brew install llvm@$llvm_version

if [ -n "$GITHUB_ENV" ]; then
  echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
fi
