#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rustc -O --crate-type cdylib \
  "$script_dir/mutant_pool.rs" \
  -o "$script_dir/libfoldy_mutant_pool.so"
