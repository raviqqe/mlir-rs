#!/bin/sh

set -e

cd $(dirname $0)/..
cd melior

cargo install cargo-expand
cargo expand >/tmp/cargo_expand.txt
