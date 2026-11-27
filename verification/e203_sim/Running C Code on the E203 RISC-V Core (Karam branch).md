# Running C Code on the E203 RISC-V Core (Karam branch)

Nov 27, 2026 · @Samehh

## Summary

Both C programs on the Karam branch build and pass on the Hummingbird E203 RISC-V core in RTL simulation (Icarus Verilog). No files in the repo were changed and nothing new was installed.

| Program | What it does | Result | Cycles |
| --- | --- | --- | --- |
| `add` | 1 + 1, minimal startup, no SDK | `TEST_PASS` | 4,410 whole run, 23 instructions |
| `conv2d_sim` | 5×5 image \* 3×3 kernel convolution, Hummingbird SDK | `PROGRAM PASS`, output `63 72 81 / 108 117 126 / 153 162 171` | Convolution: 2,335 cycles, 836 instructions; whole run 349,593 |

## Step 1: Downloading the Karam branch

The branch was cloned into the home folder at `~/riscv-yolo-accelerator`.

1. List the branches on GitHub to get the exact name. The branch is `Karam` with a capital K; `karam` does not exist.

   ```bash
   git ls-remote --heads https://github.com/gradproject-27/riscv-yolo-accelerator
   ```

   This shows three branches: `Karam`, `Sameh` and `main`.
2. Clone only that branch into the home folder:

   ```bash
   cd ~
   git clone -b Karam https://github.com/gradproject-27/riscv-yolo-accelerator
   ```
3. Check it: `git branch` shows `* Karam`, and the latest commit is `fff39a9` ("Fix E203 waveform dump, save simulation logs, use absolute paths in docs"). 617 files were downloaded.

## Step 2: Checking the tools

Everything the README asks for was already on the machine. The SDK and E203 RTL were not in the home folder where the README expects them, though: they were in `~/projects/riscv-yolo/`.

| Tool | Where it is | Version found | README's tested version |
| --- | --- | --- | --- |
| Nuclei RISC-V GCC (`riscv-nuclei-elf-gcc`) | `~/Nuclei/gcc/bin` | 10.2.0 | 9.2.0 (2020.08 release) |
| Hummingbird SDK (`hbird-sdk`) | `~/projects/riscv-yolo/hbird-sdk` | commit `0e0e908` | `0e0e908` (same) |
| E203 core RTL (`e203_hbirdv2`) | `~/projects/riscv-yolo/e203_hbirdv2` | commit `9f3e34f`, RTL already installed in `vsim/install` | `9f3e34f` (same) |
| Icarus Verilog | `/usr/bin/iverilog` | 11.0 | 11.0 (same) |
| GTKWave, Verilator | `/usr/bin` | installed | — |

Commands used to check:

```bash
~/Nuclei/gcc/bin/riscv-nuclei-elf-gcc --version
git -C ~/projects/riscv-yolo/hbird-sdk rev-parse HEAD
git -C ~/projects/riscv-yolo/e203_hbirdv2 rev-parse HEAD
ls ~/projects/riscv-yolo/e203_hbirdv2/vsim/install
iverilog -V
```

Because the SDK and RTL are in a different folder, two environment variables point the repo's scripts at them instead of moving anything: `HBIRD_SDK_ROOT` (used by the conv2d Makefile) and `E203_DIR` (used by `verification/e203_sim/run.sh`).

## Step 3: Running 1 + 1 on the E203

`add` passed with `TEST_PASS` in 4,410 cycles and 23 instructions, the same numbers as the README. It takes about 1 second.

```bash
cd ~/riscv-yolo-accelerator/software/e203_baremetal/add
make
E203_DIR=~/projects/riscv-yolo/e203_hbirdv2 make sim
```

What each part does:

1. `make` compiles `start.S` and `main.c` with `riscv-nuclei-elf-gcc -march=rv32imac -mabi=ilp32` using the linker script `link.ld`. That produces `add.elf`.
2. It then converts the ELF to `add.verilog`, a memory image the testbench loads into the E203's instruction memory (ITCM at `0x8000_0000`). The `sed` step rewrites addresses from `@800…` to `@000…` because the testbench counts from the start of ITCM. It also writes the disassembly `add.dump`.
3. `make sim` calls `verification/e203_sim/run.sh`. The first time, it compiles the E203 RTL and testbench with `iverilog`. Then `vvp` runs the simulation, which boots the core from the image.
4. The program puts its result in register `x3`. When it finishes, the testbench checks `x3 = 1` and prints `TEST_PASS` with the cycle count.

The output is also saved to `verification/e203_sim/build/add.log`.

## Step 4: Running the 2D convolution on the E203

`conv2d_sim` passed with `PROGRAM PASS` and the correct 3×3 output. The convolution itself took 2,335 cycles and 836 instructions. The simulation takes about 3 minutes.

```bash
cd ~/riscv-yolo-accelerator/software/e203_baremetal/conv2d_sim
export PATH=~/Nuclei/gcc/bin:$PATH
export HBIRD_SDK_ROOT=~/projects/riscv-yolo/hbird-sdk
export E203_DIR=~/projects/riscv-yolo/e203_hbirdv2
make dasm
make sim
```

What each part does:

1. `export PATH=…` puts the RISC-V compiler on the path; the SDK calls `riscv-nuclei-elf-gcc` by name.
2. `make dasm` builds `main.c` with the Hummingbird SDK (`SOC=hbirdv2`, `BOARD=ddr200t`, `DOWNLOAD=ilm`, `SIMU=rtl`, `-O2`). The SDK adds the startup code, UART driver and `printf`. It outputs `conv2d_sim.elf`, the disassembly `.dasm` and the memory image `conv2d_sim.verilog`.
3. `make sim` runs the image on the E203 RTL through `run.sh`, the same as in step 3.
4. The program reads the `mcycle` and `minstret` counters before and after the convolution, prints the result over UART, and writes `0xC0DE0001` (pass) to DTCM address `0x9000_1000`. The repo's `tb_monitor.v` sees that value and prints the `PROGRAM PASS` summary.

Output from the run:

```text
Convolution output:
63 72 81
108 117 126
153 162 171
Convolution: 2335 cycles, 836 instructions
Printing output: 84790 cycles
        PROGRAM PASS
DTCM result             = 0xc0de0001
Total cycle_count       = 349593
Valid instruction count = 125069
```

Printing the result with `printf` took 84,790 cycles, 36× more than the convolution. To compare hardware with software, use the section counters, not the whole-run total. The full log is in `verification/e203_sim/build/conv2d_sim.log`.

## How to run it yourself

Open a terminal and set the environment once. It lasts for that terminal only:

```bash
export PATH=~/Nuclei/gcc/bin:$PATH
export HBIRD_SDK_ROOT=~/projects/riscv-yolo/hbird-sdk
export E203_DIR=~/projects/riscv-yolo/e203_hbirdv2
```

Then run either program:

```bash
# 1 + 1  (about 1 second)
cd ~/riscv-yolo-accelerator/software/e203_baremetal/add && make && make sim

# 2D convolution  (about 3 minutes)
cd ~/riscv-yolo-accelerator/software/e203_baremetal/conv2d_sim && make dasm && make sim
```

Read the last result without running it again:

```bash
cat ~/riscv-yolo-accelerator/verification/e203_sim/build/conv2d_sim.log
```

To see the waveform (about 10 minutes and a 108 MB file):

```bash
cd ~/riscv-yolo-accelerator
DUMPWAVE=1 verification/e203_sim/run.sh software/e203_baremetal/conv2d_sim/conv2d_sim.verilog
gtkwave verification/e203_sim/build/tb_top.vcd verification/e203_sim/conv2d_sim.gtkw
```

To run your own C code, copy the `conv2d_sim` folder, change `main.c` and `TARGET` in its `Makefile`, then run `make dasm && make sim`. Code and data must fit in the 64 KB ITCM and 64 KB DTCM.

## Notes and next steps

- **Convolution cycle count differs slightly.** We measured 2,335 cycles; the README says 2,306. The instruction count (836) is identical. The likely cause is the compiler: GCC 10.2 here vs. the 2020.08 GCC 9.2 the README used. Install the 2020.08 toolchain to match the README exactly.
- **Folder paths.** The README expects `~/hbird-sdk` and `~/e203_hbirdv2`. On this machine they are in `~/projects/riscv-yolo/`, so the three `export` lines are needed in every new terminal.
- **Own programs.** Keep a copied program folder inside `software/e203_baremetal/`, because `make sim` finds `run.sh` by a relative path (`../../../verification/e203_sim/run.sh`). To get a `PROGRAM PASS` summary, the program has to write `0xC0DE0001` to `0x9000_1000` as `conv2d_sim/main.c` does.
- **`printf` appears twice in the log.** It is a logging artifact, not a second run (see `docs/E203_SIMULATION_REPORT.md`).
- **Testbench edit in the other clone.** `~/projects/riscv-yolo/e203_hbirdv2/tb/tb_top.v` has an uncommitted change (an `_exit()` stop at PC `0x80000248`). It did not affect these runs, which use the copy in `vsim/install/tb`.
- **Next: the full YOLO engine.** The C INT8 engine in `software/c_inference` has run only on the PC, not on the E203. Its 8.44 MiB of weights and 3.30 MiB feature-map arena do not fit in the 64 KB ITCM/DTCM. Running it needs external memory or a cut-down model, and probably Verilator or the FPGA instead of Icarus for speed.
