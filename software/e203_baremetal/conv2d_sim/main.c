// Small bare-metal 2D convolution (5x5 image, 3x3 kernel) for the E203.
// Transcribed from the EECE27 progress report (GP_Progress_RISCV_YOLOv3tiny.pdf, page 12).

#include <stdio.h>
#include <stdint.h>

#define SIM_RESULT (*(volatile uint32_t *)0x90001000)
#define IMG_SIZE 5
#define K_SIZE   3
#define OUT_SIZE (IMG_SIZE - K_SIZE + 1)

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

    printf("Convolution output:\n");

    for (int i = 0; i < OUT_SIZE; i++) {
        for (int j = 0; j < OUT_SIZE; j++) {
            printf("%ld ", (long)output[i][j]);
        }
        printf("\n");
    }

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
