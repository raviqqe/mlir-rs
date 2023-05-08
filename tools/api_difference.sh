#!/bin/sh

set -e

. $(dirname $0)/utility.sh

mkdir -p /tmp/melior

implemented_api >/tmp/melior/implemented.txt
all_api >/tmp/all.txt

diff -u /tmp/implemented.txt /tmp/all.txt | grep '^+[mM]' | sed s/^.//
