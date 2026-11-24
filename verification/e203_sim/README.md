# E203 RTL simulation

Runs a compiled program (`.verilog` memory image) on the Hummingbird E203 RISC-V core RTL
with Icarus Verilog, using the stock E203 testbench plus a small monitor.

| File | Purpose |
|---|---|
| `run.sh` | Compiles E203 RTL + `tb_top.v` + `tb_monitor.v` once (into `build/`), then runs a program with `vvp` |
| `tb_monitor.v` | Observe-only module: prints UART0 output (`printf`) and ends the run when the program writes `0xC0DE0001` (PASS) / `0xC0DE0002` (FAIL) to DTCM `0x90001000` |
| `conv2d_sim.gtkw` | GTKWave view: clock/reset, PC, `x3`, DTCM writes, UART bytes |

## One-time setup

```bash
git clone https://github.com/riscv-mcu/e203_hbirdv2.git ~/e203_hbirdv2
cd ~/e203_hbirdv2/vsim && make install      # run only once
sudo apt install iverilog gtkwave
```

Set `E203_DIR` if the E203 clone is somewhere other than `~/e203_hbirdv2`.

## Usage

```bash
./run.sh path/to/program.verilog              # run
DUMPWAVE=1 ./run.sh path/to/program.verilog   # run and write build/tb_top.vcd
gtkwave build/tb_top.vcd conv2d_sim.gtkw      # view the waveform
```

The programs live in `software/e203_baremetal/` (`make sim` there calls this script).

## How a run ends

- **Stock testbench rule** (used by `add`): PC reaches `0x80000086` eight times, then
  `TEST_PASS` if register `x3 == 1`.
- **Monitor rule** (used by `conv2d_sim`): a write of `0xC0DE0001` / `0xC0DE0002` to `0x90001000`.
- Otherwise the testbench times out after 10M cycles.
