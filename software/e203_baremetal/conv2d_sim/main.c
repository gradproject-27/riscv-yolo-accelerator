// Small bare-metal 2D convolution (5x5 image, 3x3 kernel) for the E203.
// Transcribed from the EECE27 progress report (GP_Progress_RISCV_YOLOv3tiny.pdf, page 12).
//
// Added on top of the report's code: mcycle/minstret measurement of the
// convolution alone and of the printf section, so the software-only cost can
// later be compared with the NICE accelerator. The numbers are printed and
// also stored next to SIM_RESULT so the testbench monitor can show them.

#include <stdio.h>
#include <stdint.h>
#include "hbird_sdk_soc.h"

#define SIM_RESULT    (*(volatile uint32_t *)0x90001000)
#define CONV_CYCLES   (*(volatile uint32_t *)0x90001004)
#define CONV_INSTRET  (*(volatile uint32_t *)0x90001008)
#define PRINT_CYCLES  (*(volatile uint32_t *)0x9000100C)

#define IMG_SIZE 5
#define K_SIZE   3
#define OUT_SIZE (IMG_SIZE - K_SIZE + 1)

// Stop the compiler from moving memory accesses across the counter reads
#define BARRIER() __asm__ volatile ("" ::: "memory")

int main(void)
{
    const int32_t image[IMG_SIZE][IMG_SIZE] = {
        { 1,  2,  3,  4,  5},
        { 6,  7,  8,  9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };

    const int32_t kernel[K_SIZE][K_SIZE] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    int32_t output[OUT_SIZE][OUT_SIZE] = {0};

    BARRIER();
    uint64_t c0 = __get_rv_cycle();
    uint64_t i0 = __get_rv_instret();
    BARRIER();

    // Convolution / sliding-window calculation
    for (int i = 0; i < OUT_SIZE; i++) {
        for (int j = 0; j < OUT_SIZE; j++) {

            int32_t sum = 0;

            for (int ki = 0; ki < K_SIZE; ki++) {
                for (int kj = 0; kj < K_SIZE; kj++) {
                    sum += image[i + ki][j + kj]
                         * kernel[ki][kj];
                }
            }

            output[i][j] = sum;
        }
    }

    BARRIER();
    uint64_t c1 = __get_rv_cycle();
    uint64_t i1 = __get_rv_instret();
    BARRIER();

    printf("Convolution output:\n");

    for (int i = 0; i < OUT_SIZE; i++) {
        for (int j = 0; j < OUT_SIZE; j++) {
            printf("%ld ", (long)output[i][j]);
        }
        printf("\n");
    }

    BARRIER();
    uint64_t c2 = __get_rv_cycle();

    uint32_t conv_cycles  = (uint32_t)(c1 - c0);
    uint32_t conv_instret = (uint32_t)(i1 - i0);
    uint32_t print_cycles = (uint32_t)(c2 - c1);

    printf("Convolution: %lu cycles, %lu instructions\n",
           (unsigned long)conv_cycles, (unsigned long)conv_instret);
    printf("Printing output: %lu cycles\n", (unsigned long)print_cycles);

    CONV_CYCLES  = conv_cycles;
    CONV_INSTRET = conv_instret;
    PRINT_CYCLES = print_cycles;

    // Easy-to-check result for debugging
    volatile int32_t check = output[0][0];

    if (check == 63) {
        SIM_RESULT = 0xC0DE0001;
    } else {
        SIM_RESULT = 0xC0DE0002;
    }

    while (1) {
        __asm__ volatile ("nop");
    }

    return 0;
}
