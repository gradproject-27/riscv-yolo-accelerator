# E203 RTL simulation

Runs a compiled program (`.verilog` memory image) on the Hummingbird E203 RISC-V core RTL
with Icarus Verilog, using the stock E203 testbench plus a small monitor.

| File | Purpose |
|---|---|
| `run.sh` | Compiles E203 RTL + `tb_top.v` + `tb_monitor.v` into `build/` (once, and again when `tb_monitor.v` or `run.sh` change), then runs a program with `vvp` and saves its output to `build/<program>.log` |
| `tb_monitor.v` | Observe-only module: prints UART0 output (`printf`) and ends the run when the program writes `0xC0DE0001` (PASS) / `0xC0DE0002` (FAIL) to DTCM `0x90001000` |
| `conv2d_sim.gtkw` | GTKWave view: clock/reset, PC, `x3`, DTCM writes, UART bytes |

## One-time setup

```bash
git clone https://github.com/riscv-mcu/e203_hbirdv2.git ~/e203_hbirdv2
git -C ~/e203_hbirdv2 checkout 9f3e34f337f5d0a1236a285871ffa2032491617a   # tested commit
cd ~/e203_hbirdv2/vsim && make install      # run only once
sudo apt install iverilog gtkwave
```

Tested with e203_hbirdv2 `9f3e34f` (2025-08-06) and Icarus Verilog 11.0 on Ubuntu 22.04.
All versions are listed in `software/e203_baremetal/README.md`.

Set `E203_DIR` if the E203 clone is somewhere other than `~/e203_hbirdv2`.

## Usage

```bash
cd ~/riscv-yolo-accelerator/verification/e203_sim
./run.sh path/to/program.verilog              # run (output also saved to build/<program>.log)
DUMPWAVE=1 ./run.sh path/to/program.verilog   # run and write build/tb_top.vcd (much slower, large file)
gtkwave build/tb_top.vcd conv2d_sim.gtkw      # view the waveform
cat build/conv2d_sim.log                      # read the last run's output again
```

Each run's terminal output (the `printf` text and the PASS/FAIL summary) is also written to
`build/<program>.log`, for example `build/add.log` or `build/conv2d_sim.log`. The next run of the
same program overwrites it. `gtkwave` only opens a saved waveform and prints none of this.

`run.sh` compiles with `-D iverilog=1`: the stock `tb_top.v` only writes the waveform inside
`` `ifdef iverilog ``, so without it `DUMPWAVE=1` silently does nothing. A waveform run prints
`iverlog used`. The file is in LXT2 format despite the `.vcd` name; GTKWave reads it.

The programs live in `software/e203_baremetal/` (`make sim` there calls this script).

## How a run ends

- **Stock testbench rule** (used by `add`): PC reaches `0x80000086` eight times, then
  `TEST_PASS` if register `x3 == 1`.
- **Monitor rule** (used by `conv2d_sim`): a write of `0xC0DE0001` / `0xC0DE0002` to `0x90001000`.
- Otherwise the testbench times out after 10M cycles.
