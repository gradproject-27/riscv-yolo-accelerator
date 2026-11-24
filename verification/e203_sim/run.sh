#!/usr/bin/env bash
# Run a program on the E203 RTL (Icarus Verilog) with the extra monitor.
# Usage: run.sh <path/to/program>.verilog
#        DUMPWAVE=1 run.sh <path/to/program>.verilog   (also writes a waveform)
# Output: printed to the terminal and saved to build/<program>.log
# Env:   E203_DIR  path to the e203_hbirdv2 clone (default ~/e203_hbirdv2)
# The .verilog file must start at @00000000 (ITCM 0x80000000).
set -euo pipefail

PROG=$(readlink -f "$1")
E203=${E203_DIR:-$HOME/e203_hbirdv2}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/build
VSRC=$E203/vsim/install/rtl

if [ ! -d "$VSRC" ]; then
  echo "E203 RTL not installed. Run once:  cd $E203/vsim && make install" >&2
  exit 1
fi

mkdir -p "$OUT"
if [ ! -f "$OUT/vvp.exec" ] || [ "$HERE/tb_monitor.v" -nt "$OUT/vvp.exec" ] || [ "$HERE/run.sh" -nt "$OUT/vvp.exec" ]; then
  echo "Compiling E203 RTL + testbench ..."
  iverilog -o "$OUT/vvp.exec" -g2005-sv -D DISABLE_SV_ASSERTION=1 -D iverilog=1 \
    -I "$VSRC/core/" -I "$VSRC/perips/" -I "$VSRC/perips/apb_i2c/" \
    $(ls "$VSRC"/*/*.v "$VSRC"/*/*/*.v) \
    "$E203/vsim/install/tb/tb_top.v" "$HERE/tb_monitor.v"
fi

cd "$OUT"
# The terminal output is also saved to build/<program>.log (overwritten on each run).
LOG=$OUT/$(basename "${PROG%.verilog}").log
# DUMPWAVE=1 run.sh prog.verilog  -> writes build/tb_top.vcd (LXT2 format) for GTKWave
vvp "$OUT/vvp.exec" -lxt2 +DUMPWAVE="${DUMPWAVE:-0}" +TESTCASE="${PROG%.verilog}" 2>&1 | tee "$LOG"
echo "Log saved to $LOG"
