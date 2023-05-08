#!/bin/sh

set -e

. $(dirname $0)/../utility.sh

directory=/tmp/melior

mkdir -p $directory

implemented_api >$directory/implemented.txt
all_api >$directory/all.txt

diff -u $directory/implemented.txt $directory/all.txt | grep '^+[mM]' | sed s/^.//
