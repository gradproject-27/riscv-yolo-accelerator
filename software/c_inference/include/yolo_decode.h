/* Integer-only YOLO decode + NMS (no FPU, no libm).  Needs decode_tables.h, generated from
 * layers.json by gen_decode_tables.py.  Call after yolo_forward(). */
#ifndef YOLO_DECODE_H
#define YOLO_DECODE_H
#include <stdint.h>

#ifndef YOLO_MAX_ENTRIES
#define YOLO_MAX_ENTRIES 8192   /* (box,class) candidates above threshold before NMS; ~24 B each */
#endif

/* prob is Q16 (65536 = 1.0); the box (relative to the network input) is Q20 (1048576 = 1.0) */
typedef struct {
    uint32_t cls;
    uint32_t prob;                       /* objectness * class score */
    uint32_t cx, cy, w, h;               /* relative to the network input */
} yolo_det_t;

/* thresh_q16 = round(thresh*65536); nms_pct = IoU threshold in percent (45 -> 0.45).
 * Returns the number of detections written to out (sorted by prob, descending).
 * *dropped (optional) = candidates lost because YOLO_MAX_ENTRIES was exceeded: must be 0 for exact results. */
int yolo_decode(uint32_t thresh_q16, uint32_t nms_pct, yolo_det_t *out, int max_out, int *dropped);

#endif
