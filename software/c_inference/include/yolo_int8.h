/* Golden int8 YOLOv3-tiny engine (step 2).
 * Plain C99, no malloc, no libm, no OS: the same file builds on x86 and on the E203
 * (rv32imac).  Bit-exact against yolov3_tiny_ref.py (the ref_out/int8 dumps).
 * The model description comes from the generated layers.h (put its folder on -I). */
#ifndef YOLO_INT8_H
#define YOLO_INT8_H
#include <stdint.h>
#include <stddef.h>

#ifndef YOLO_ARENA_BYTES
#define YOLO_ARENA_BYTES (4u * 1024u * 1024u)   /* feature-map arena, checked at init */
#endif

/* called after every non-yolo layer with its int8 CHW output (debug / compare hook, may be NULL) */
typedef void (*yolo_layer_cb)(int idx, const int8_t *out, size_t n, void *user);

/* Plan buffer placement by layer lifetime.  Returns 0 on success, -1 if the arena is too
 * small or the model uses something unsupported.  *peak_bytes = arena bytes really needed. */
int yolo_init(size_t *peak_bytes);

/* input image tensor, INPUT_C*INPUT_H*INPUT_W int8, CHW, scale 2^-INPUT_FA */
int8_t *yolo_input(void);
void    yolo_quantize_f32(const float *chw, int8_t *q, size_t n);   /* round(x * 2^INPUT_FA), clamp */

/* weights: weights_q.bin (int8, OIHW, layer order); bias: bias_i32.bin */
int yolo_forward(const int8_t *weights, const int32_t *bias, yolo_layer_cb cb, void *user);

/* output of layer idx after yolo_forward (valid until the next call; yolo layers alias their input) */
const int8_t *yolo_layer_output(int idx);

#endif
