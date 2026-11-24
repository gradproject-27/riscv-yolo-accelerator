# Minimal startup for running a C program on the E203 RTL testbench.
# The testbench ends the simulation once the PC reaches 0x80000086 eight
# times, then reports TEST_PASS if register x3 == 1.

    .section .text.init
    .option norelax            # keep write_tohost exactly at offset 0x86
    .globl _start
_start:
    li   sp, 0x90010000        # stack at top of the 64 KB DTCM
    call main                  # result comes back in a0
    li   x3, 0                 # assume fail
    li   t0, 2
    bne  a0, t0, write_tohost  # 1 + 1 should be 2
    li   x3, 1                 # pass
    j    write_tohost

    .org 0x86                  # 0x80000000 + 0x86 = testbench end address
write_tohost:
    j    write_tohost
