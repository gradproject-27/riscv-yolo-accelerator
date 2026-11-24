# Bare-metal programs for the E203

Small C programs that run on the Hummingbird E203 RISC-V core in RTL simulation
(`verification/e203_sim`). Full write-up: [docs/E203_SIMULATION_REPORT.md](../../docs/E203_SIMULATION_REPORT.md).

| Folder | What it does | Build system | Result |
|---|---|---|---|
| `add/` | `1 + 1`, minimal startup (`start.S`) that follows the stock testbench rule | plain Makefile | `TEST_PASS`, 4,410 cycles |
| `conv2d_sim/` | 5×5 image * 3×3 kernel convolution from the progress report, prints the output | Hummingbird SDK | `PROGRAM PASS`, output `63 72 81 / 108 117 126 / 153 162 171` |

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
