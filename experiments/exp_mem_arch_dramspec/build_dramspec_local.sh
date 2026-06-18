#!/usr/bin/env bash
# Build DRAMSpec without installing system packages.

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
DRAMSPEC_DIR="$EXP_DIR/tools/DRAMSpec"
DEPS_DIR="$EXP_DIR/tools/deps"
APT_DIR="$DEPS_DIR/apt"
BOOST_ROOT="$DEPS_DIR/boost_root"
BOOST_INCLUDE="$BOOST_ROOT/usr/include"

mkdir -p "$APT_DIR" "$BOOST_ROOT"

if [[ ! -f "$BOOST_INCLUDE/boost/units/static_constant.hpp" ]]; then
  (
    cd "$APT_DIR"
    if ! ls libboost1.83-dev_*.deb >/dev/null 2>&1; then
      apt-get download libboost1.83-dev
    fi
    dpkg-deb -x libboost1.83-dev_*.deb "$BOOST_ROOT"
  )
fi

(
  cd "$DRAMSPEC_DIR"
  git submodule update --init --recursive
  mkdir -p build/release
  g++ -std=c++11 -O2 -Wall -Wextra \
    -I. \
    -Iparser/rapidjson/include \
    -I"$BOOST_INCLUDE" \
    core/SubArray.cpp \
    core/Tile.cpp \
    core/Bank.cpp \
    core/Channel.cpp \
    core/Timing.cpp \
    core/Current.cpp \
    utils/utils.cpp \
    parser/ArgumentsParser.cpp \
    parser/TechnologyValues.cpp \
    parser/DramSpec.cpp \
    main.cpp \
    -o build/release/dramspec
)

echo "Built $DRAMSPEC_DIR/build/release/dramspec"
