# Bare-metal programs for the E203

Small C programs that run on the Hummingbird E203 RISC-V core in RTL simulation
(`verification/e203_sim`). Full write-up: [docs/E203_SIMULATION_REPORT.md](../../docs/E203_SIMULATION_REPORT.md).

| Folder | What it does | Build system | Result |
|---|---|---|---|
| `add/` | `1 + 1`, minimal startup (`start.S`) that follows the stock testbench rule | plain Makefile | `TEST_PASS`, 4,410 cycles |
| `conv2d_sim/` | 5×5 image * 3×3 kernel convolution from the progress report, prints the output and its own cycle count | Hummingbird SDK | `PROGRAM PASS`, output `63 72 81 / 108 117 126 / 153 162 171`, convolution = 2,306 cycles / 836 instructions |

## Tested versions

Results in this folder were produced with exactly these versions. Use the same
commits so you get the same build and cycle counts.

| Component | Version |
|---|---|
| [hbird-sdk](https://github.com/riscv-mcu/hbird-sdk) | commit `0e0e9083569721578c42f07a95b05a26c0760db6` (2023-01-17) |
| [e203_hbirdv2](https://github.com/riscv-mcu/e203_hbirdv2) | commit `9f3e34f337f5d0a1236a285871ffa2032491617a` (2025-08-06) |
| Nuclei RISC-V GCC | 2020.08 prebuilt, `riscv-nuclei-elf-gcc (GCC) 9.2.0` |
| Icarus Verilog | 11.0 |
| OS | Ubuntu 22.04.5 LTS |

Clone at the tested commits:

```bash
git clone https://github.com/riscv-mcu/hbird-sdk.git ~/hbird-sdk
git -C ~/hbird-sdk checkout 0e0e9083569721578c42f07a95b05a26c0760db6

git clone https://github.com/riscv-mcu/e203_hbirdv2.git ~/e203_hbirdv2
git -C ~/e203_hbirdv2 checkout 9f3e34f337f5d0a1236a285871ffa2032491617a
```

Check what you have: `git -C ~/hbird-sdk rev-parse HEAD` and `git -C ~/e203_hbirdv2 rev-parse HEAD`.

## Requirements

- Nuclei RISC-V GCC **2020.08** (GCC 9.2, `riscv-nuclei-elf-gcc`) in `~/Nuclei/gcc`
  ([download](https://download.nucleisys.com/upload/files/toolchain/gcc/nuclei_riscv_newlibc_prebuilt_linux64_2020.08.tar.bz2)).
  Newer Nuclei toolchains use a different prefix and need `_zicsr` flags, so they do not work
  with `hbird-sdk` as is.
- `hbird-sdk` cloned to `~/hbird-sdk` (only for `conv2d_sim`; override with `HBIRD_SDK_ROOT`).
- E203 simulation set up, see `verification/e203_sim/README.md`.

## Usage

```bash
# 1 + 1
cd software/e203_baremetal/add
make && make sim

# convolution
cd software/e203_baremetal/conv2d_sim
export PATH=~/Nuclei/gcc/bin:$PATH
make dasm && make sim
```
