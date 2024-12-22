#!/bin/bash

set -ex

MY_SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"
BUILD_DIR="${MY_SCRIPT_DIR}/build"

DEBUG=""
#DEBUG="-DCMAKE_BUILD_TYPE=Debug"

export LD_LIBRARY_PATH=/usr/lib/habanatools:/usr/lib/habanatools/habana_plugins:${LD_LIBRARY_PATH}

if [ ! -d "${BUILD_DIR}" ]; then
    mkdir "${BUILD_DIR}"
fi

cd "${BUILD_DIR}"
cmake ${DEBUG} ..
make -j
