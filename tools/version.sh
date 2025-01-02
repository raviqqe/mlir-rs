#!/bin/sh

set -e

level=${1:-patch}

if [ $# -gt 0 ]; then
  shift 1
fi

options="$@"

cd $(dirname $0)/..

cargo release version $level --execute --no-confirm --allow-branch '*' $options
