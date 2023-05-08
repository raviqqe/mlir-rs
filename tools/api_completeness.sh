#!/bin/sh

set -e

cd $(dirname $0)/..
cd melior

count() {
  sort -u | wc -l
}

implemented_count() {
  cargo install cargo-expand
  cargo expand | grep -o '\(m\|M\)lir[A-Z][a-zA-Z0-9]*' | count
}

upstream_count() {
  cat $(find $(brew --prefix llvm)/include/mlir-c -type f) | grep -o '\(m\|M\)lir[A-Z][a-zA-Z0-9]*' | count
}

echo $(implemented_count) / $(upstream_count) | bc -l
