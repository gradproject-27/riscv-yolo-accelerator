OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
  ITCM (rx)  : ORIGIN = 0x80000000, LENGTH = 64K
  DTCM (rw)  : ORIGIN = 0x90000000, LENGTH = 64K
}

SECTIONS
{
  .text : { *(.text.init) *(.text*) *(.rodata*) } > ITCM
  .bss  : { *(.sbss*) *(.bss*) *(COMMON) } > DTCM
}
