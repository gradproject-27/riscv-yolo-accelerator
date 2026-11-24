# Running C programs on the Hummingbird E203 RISC-V core (RTL simulation)

Date: 2026-11-24
Machine: Ubuntu Linux

## Goal

Run small C programs on a real RISC-V CPU design (not just an emulator):

1. A trivial `1 + 1` program.
2. The small 2D convolution test (5×5 image, 3×3 kernel) from the team progress report
   `riscv-yolo-accelerator/docs/GP_Progress_RISCV_YOLOv3tiny.pdf` (page 12, "E203 Flow Example").

Both programs run on the Hummingbird **E203** RISC-V core RTL (Verilog), simulated with
**Icarus Verilog (iverilog)**, using the **Hummingbird SDK** and the **Nuclei RISC-V GCC** compiler.

---

## Final result

| Program | Result | Whole run (testbench) | Convolution only (`mcycle`/`minstret`) |
|---|---|---|---|
| `1 + 1` | **TEST_PASS** (x3 = 1, i.e. the core computed 2) | 4,410 cycles, 23 instr. | — |
| `conv2d_sim` | **PROGRAM PASS** (DTCM result = `0xC0DE0001`) | 350,309 cycles, 125,117 instr. | **2,306 cycles, 836 instr.** |

Output printed by the E203 over its UART:

```
HummingBird SDK Build Time: Nov 24 2026, 22:22:16
Download Mode: ILM
CPU Frequency 16500000 Hz
Convolution output:
63 72 81
108 117 126
153 162 171
Convolution: 2306 cycles, 836 instructions
Printing output: 85219 cycles

==========================================
        PROGRAM PASS
DTCM result             = 0xc0de0001
Total cycle_count       = 350309
Valid instruction count = 125117
Program-reported counters (mcycle/minstret):
  section cycles        = 2306
  section instructions  = 836
  extra cycles          = 85219
==========================================
```

The numbers `63 72 81 / 108 117 126 / 153 162 171` match the progress report exactly.
(Check: top-left = 1+2+3+6+7+8+11+12+13 = 63.)

### Where the cycles go

The program reads the E203's `mcycle` and `minstret` counters right before and right after the
convolution loops, and again after printing:

| Part | Cycles | Share of run |
|---|---|---|
| Convolution (9 outputs × 9 multiply-adds = 81 MACs) | 2,306 | 0.7% |
| Printing the 3×3 result with `printf` | 85,219 | 24% |
| Everything else (SDK startup, UART init, banner, timing printout) | ~263,000 | 75% |

- **2,306 cycles for the convolution is the software baseline** to compare against the NICE
  accelerator. That is about 28 cycles and 10 instructions per multiply-accumulate.
- CPI is about 2.8. The E203 multiplier is multi-cycle (`mul` takes many cycles, not 1),
  which is a large part of the cost and exactly what an accelerator removes.
- Earlier runs without the timing code took 241,980 cycles. The extra ~108k cycles come from
  the two additional `printf` lines that report the timing.

**Total cycle count differs from the progress report** (report: 35,326 cycles, 18,049
instructions for the whole run). The whole-run number is dominated by startup and `printf`,
which depend on the testbench and SDK setup. The report's run also shows a different measured CPU
frequency (540,344 Hz vs. our 16,500,000 Hz), so the authors' setup was not identical to ours.
Use the convolution-only number (2,306 cycles) when comparing hardware against software.

---

## What we did, step by step

### 1. Cloned the repositories into the home folder

```bash
cd ~
git clone https://github.com/gradproject-27/riscv-yolo-accelerator.git
git clone https://github.com/riscv-mcu/hbird-sdk.git
git clone https://github.com/riscv-mcu/e203_hbirdv2.git
# pin to the commits these results were produced with
git -C hbird-sdk checkout 0e0e9083569721578c42f07a95b05a26c0760db6     # 2023-01-17
git -C e203_hbirdv2 checkout 9f3e34f337f5d0a1236a285871ffa2032491617a  # 2025-08-06
```

Tested versions: hbird-sdk `0e0e908`, e203_hbirdv2 `9f3e34f`, Nuclei GCC 2020.08 (GCC 9.2.0),
Icarus Verilog 11.0, Ubuntu 22.04.5 LTS.

- `riscv-yolo-accelerator`: the team project (tracks GitHub `main`; update with `git pull`).
- `hbird-sdk`: Hummingbird software development kit (startup code, drivers, printf, build system).
- `e203_hbirdv2`: the E203 RISC-V core + SoC in Verilog, plus a testbench (`tb/tb_top.v`) and
  simulation scripts (`vsim/`).

### 2. Installed the RISC-V compiler

Downloaded the **Nuclei RISC-V GCC 2020.08** (GCC 9.2.0, ~106 MB) from
`https://download.nucleisys.com/upload/files/toolchain/gcc/nuclei_riscv_newlibc_prebuilt_linux64_2020.08.tar.bz2`
and unpacked it to `~/Nuclei/gcc`.

```bash
~/Nuclei/gcc/bin/riscv-nuclei-elf-gcc --version   # riscv-nuclei-elf-gcc (GCC) 9.2.0
```

Created `~/hbird-sdk/setup_config.sh` containing `NUCLEI_TOOL_ROOT=~/Nuclei`
so the SDK's `source setup.sh` finds the compiler.

### 3. Built the E203 RTL simulation

```bash
cd ~/e203_hbirdv2/vsim
make install
make compile SIM=iverilog
```

### 4. Wrote and ran the `1 + 1` program

Folder: `software/e203_baremetal/add/`

| File | Purpose |
|---|---|
| `main.c` | `result = a + b;` with `volatile` so the core really executes the add |
| `start.S` | Minimal startup: set stack, call `main`, set x3 = 1 if the result is 2, then loop at `0x80000086` |
| `link.ld` | Code in ITCM (`0x80000000`), data in DTCM (`0x90000000`) |
| `Makefile` | `make` builds `add.elf/.verilog/.dump`; `make sim` runs it on the E203 |

```bash
cd ~/riscv-yolo-accelerator/software/e203_baremetal/add
make
make sim
```

### 5. Recreated and ran the convolution test from the PDF

Folder: `software/e203_baremetal/conv2d_sim/` (built with the Hummingbird SDK from `~/hbird-sdk`)

- `main.c`: the code transcribed from the PDF screenshots (5×5 image, 3×3 all-ones kernel,
  prints the 3×3 output, writes `0xC0DE0001` (pass) or `0xC0DE0002` (fail) to `0x90001000`,
  then loops forever).
- `Makefile`: SDK build with `SOC=hbirdv2 BOARD=ddr200t DOWNLOAD=ilm SIMU=rtl`.

```bash
cd ~/riscv-yolo-accelerator/software/e203_baremetal/conv2d_sim
export PATH=~/Nuclei/gcc/bin:$PATH
make dasm      # -> conv2d_sim.elf, .verilog, .dump
make sim       # runs on the E203 (~3 minutes)
```

### 6. Added a small testbench monitor

Folder: `verification/e203_sim/`

| File | Purpose |
|---|---|
| `tb_monitor.v` | Extra Verilog module compiled next to the stock `tb_top.v`. It only watches: prints every byte written to UART0 (so `printf` shows up), and ends the simulation when `0xC0DE0001`/`0xC0DE0002` appears at DTCM `0x90001000`. |
| `run.sh` | Compiles the E203 RTL + `tb_top.v` + `tb_monitor.v` with iverilog (once, and again whenever `tb_monitor.v` or `run.sh` change), then runs any `.verilog` program with `vvp`, saving its output to `build/<program>.log`. Compiles with `-D iverilog=1` so `DUMPWAVE=1` can write a waveform. |

The original E203 files were **not modified**.

---

## Problems we hit and how we solved them

| # | Problem | Cause | Solution |
|---|---|---|---|
| 1 | Nothing to run code on inside `riscv-yolo-accelerator` | Its `hardware/` and `verification/` folders are empty (only `.gitkeep`) | Used the open-source E203 core (`e203_hbirdv2`) as the CPU |
| 2 | No RISC-V compiler on the machine | Nothing installed; `apt install` needs the sudo password | Downloaded the prebuilt Nuclei toolchain into `~/Nuclei/gcc` (no sudo needed) |
| 3 | Which toolchain version? | `hbird-sdk` is old: it calls `riscv-nuclei-elf-gcc` and uses `-march=rv32imac`. Newer Nuclei toolchains (2024+) are renamed `riscv64-unknown-elf-` and GCC 12+ needs `_zicsr_zifencei` in `-march` | Picked **2020.08 (GCC 9.2)**, which matches the SDK exactly |
| 4 | Cloning `e203_hbirdv2` took over 5 minutes | Large repo, slow network | Let it finish in the background |
| 5 | The stock E203 testbench does not "see" an ordinary program finish | `tb_top.v` was written for the official `riscv-tests`: it stops only after the PC reaches `0x80000086` eight times, then reports PASS if register `x3 == 1` | For `1 + 1`: wrote `start.S` with an endless loop placed exactly at offset `0x86` (`.org 0x86`) and set `x3 = 1` when the answer is 2 |
| 6 | The end loop landed at `0x80000080` instead of `0x80000086` | Linker relaxation shrank the `call main` from 8 bytes to 2, shifting everything after it | Added `.option norelax` in `start.S` |
| 7 | `make sim` failed: `run/Makefile: No such file or directory` | E203's `vsim/Makefile` uses `${PWD}` to find itself, so calling it from another folder breaks | Wrote `verification/e203_sim/run.sh`, which calls `iverilog`/`vvp` directly |
| 8 | Re-running E203's `make install` would break the build | `cp -rf rtl/e203 install/rtl` copies the RTL *inside* the existing folder the second time, causing duplicate modules | Run `make install` only once; removed it from our scripts |
| 9 | The convolution code was only a picture in the PDF | The code is a screenshot, not text | Extracted the images from the PDF (`pdfimages`), enlarged them and transcribed the code |
| 10 | The convolution program never ends in the stock testbench | It ends with `while(1) nop` and reports through memory `0x90001000`, not through `x3`/`0x80000086`. The authors' run (PDF) shows "CONVOLUTION PASS", so they used a modified testbench | Wrote `tb_monitor.v` that watches `0x90001000` and the UART, without editing the E203 files |
| 11 | The `printf` text appears **twice** in the simulation log: once at the very top (even before the testbench's time-0 `!!!!` banner) and once in the right place before `PROGRAM PASS` | Not a second run: the pass check fired once and there is a single cycle count. Text cannot be printed before time 0, so this is an output-buffering artifact of the simulator log. The PDF screenshot shows the same thing (text above the `!!!!` banner) | Harmless. Read the copy just above `PROGRAM PASS`. Root cause not investigated further |
| 12 | Could not delete the empty download folder `~/Nuclei/dl` | Blocked by a safety check (it was the current working folder at the time) | Deleted manually afterwards (archive removed, then `rmdir ~/Nuclei/dl`) |
| 13 | `DUMPWAVE=1` wrote no waveform: GTKWave failed with `Error opening .vcd file 'verification/e203_sim/build/tb_top.vcd'` | `tb_top.v` only calls `$dumpfile`/`$dumpvars` inside `` `ifdef iverilog ``, and `run.sh` (like E203's own `vsim` Makefile) never defined that macro. The old `vvp.exec` was also reused because `run.sh` only rebuilt when `tb_monitor.v` changed | Added `-D iverilog=1` to the `iverilog` command in `run.sh`, and made it rebuild when `run.sh` changes too. A waveform run now prints `iverlog used` (spelling from `tb_top.v`) and writes `build/tb_top.vcd` |

---

## File locations

In this repository:

```
docs/E203_SIMULATION_REPORT.md           this file
software/e203_baremetal/
  add/                                   1+1 on the E203 (start.S, main.c, link.ld, Makefile)
  conv2d_sim/                            convolution test (main.c, Makefile, uses hbird-sdk)
verification/e203_sim/
  run.sh                                 build + run a program on the E203 RTL
  tb_monitor.v                           UART + SIM_RESULT monitor
  conv2d_sim.gtkw                        GTKWave signal view
  build/                                 (generated, not committed) simulator, waveform, <program>.log
```

Outside the repository (cloned/installed separately):

```
~/hbird-sdk/                             Hummingbird SDK (setup_config.sh: NUCLEI_TOOL_ROOT=~/Nuclei)
~/e203_hbirdv2/                          E203 core RTL + testbench (vsim/install = installed copy)
~/Nuclei/gcc/                            Nuclei RISC-V GCC 9.2.0 toolchain
```

## Quick re-run

Every command starts from an absolute path, so the steps work from any folder and in any order.

```bash
# 1 + 1
cd ~/riscv-yolo-accelerator/software/e203_baremetal/add && make && make sim

# convolution
cd ~/riscv-yolo-accelerator/software/e203_baremetal/conv2d_sim
export PATH=~/Nuclei/gcc/bin:$PATH
make dasm && make sim

# waveform (~10 minutes, writes a ~108 MB file; the log shows "iverlog used")
cd ~/riscv-yolo-accelerator
DUMPWAVE=1 verification/e203_sim/run.sh software/e203_baremetal/conv2d_sim/conv2d_sim.verilog
gtkwave verification/e203_sim/build/tb_top.vcd verification/e203_sim/conv2d_sim.gtkw

# output of the last run of each program (gtkwave does not print it)
cat ~/riscv-yolo-accelerator/verification/e203_sim/build/conv2d_sim.log
```
