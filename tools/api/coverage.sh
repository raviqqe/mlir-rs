#!/bin/sh

set -e

. $(dirname $0)/../utility.sh

echo $(implemented_api | count) / $(all_api | count) | bc -l
