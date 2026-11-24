# 5x5 * 3x3 convolution on the Hummingbird E203, built with the Hummingbird SDK.
#   make dasm   build conv2d_sim.elf / .verilog / .dump
#   make sim    run it on the E203 RTL (Icarus Verilog, ~3 minutes)
#
# Needs: hbird-sdk (HBIRD_SDK_ROOT) and Nuclei RISC-V GCC 2020.08 on PATH,
# see docs/E203_SIMULATION_REPORT.md

TARGET = conv2d_sim

HBIRD_SDK_ROOT ?= $(HOME)/hbird-sdk

SRCDIRS = .

INCDIRS = .

# Run from the E203 ITCM, built for the RTL testbench
SOC      = hbirdv2
BOARD    = ddr200t
DOWNLOAD = ilm
SIMU     = rtl

COMMON_FLAGS := -O2

include $(HBIRD_SDK_ROOT)/Build/Makefile.base

sim: dasm
	../../../verification/e203_sim/run.sh $(TARGET).verilog
