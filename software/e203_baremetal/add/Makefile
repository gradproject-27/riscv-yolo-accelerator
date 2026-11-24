# 1 + 1 on the Hummingbird E203 (no SDK, minimal startup).
#   make        build add.elf / add.verilog / add.dump
#   make sim    run it on the E203 RTL (Icarus Verilog)
#
# Needs: Nuclei RISC-V GCC 2020.08 (riscv-nuclei-elf-gcc), see docs/E203_SIMULATION_REPORT.md

NUCLEI_TOOL_ROOT ?= $(HOME)/Nuclei
PREFIX           := $(NUCLEI_TOOL_ROOT)/gcc/bin/riscv-nuclei-elf-
RUN_SIM          := ../../../verification/e203_sim/run.sh

CFLAGS := -march=rv32imac -mabi=ilp32 -O1 -nostdlib -nostartfiles -ffreestanding

all: add.verilog add.dump

add.elf: start.S main.c link.ld
	$(PREFIX)gcc $(CFLAGS) -T link.ld start.S main.c -o $@

add.verilog: add.elf
	$(PREFIX)objcopy -O verilog $< $@
	sed -i 's/@800/@000/g' $@

add.dump: add.elf
	$(PREFIX)objdump -d $< > $@

sim: add.verilog
	$(RUN_SIM) add.verilog

clean:
	rm -f add.elf add.verilog add.dump

.PHONY: all sim clean
