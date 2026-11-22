#include "yolo_decode.h"
#include "yolo_int8.h"
#include "decode_tables.h"

static yolo_det_t s_ent[YOLO_MAX_ENTRIES];
static uint8_t    s_dead[YOLO_MAX_ENTRIES];

/* stable insertion sort; ordering: class ascending, then prob descending (by_class=1)
 * or prob descending only (by_class=0) */
static void sort_dets(yolo_det_t *a, int n, int by_class)
{
    int i, j;
    for (i = 1; i < n; i++) {
        yolo_det_t t = a[i];
        for (j = i - 1; j >= 0; j--) {
            int move;
            if (by_class && a[j].cls != t.cls) move = a[j].cls > t.cls;
            else                               move = a[j].prob < t.prob;
            if (!move) break;
            a[j + 1] = a[j];
        }
        a[j + 1] = t;
    }
}

/* true if IoU(a,b) > pct/100, 64-bit integer math (box sizes are clamped to 64.0 in the tables) */
static int iou_above(const yolo_det_t *a, const yolo_det_t *b, uint32_t pct)
{
    const int32_t al = (int32_t)a->cx - (int32_t)(a->w >> 1), bl = (int32_t)b->cx - (int32_t)(b->w >> 1);
    const int32_t at = (int32_t)a->cy - (int32_t)(a->h >> 1), bt = (int32_t)b->cy - (int32_t)(b->h >> 1);
    const int32_t ar = al + (int32_t)a->w, br = bl + (int32_t)b->w;
    const int32_t ab = at + (int32_t)a->h, bb = bt + (int32_t)b->h;
    const int32_t ow = (ar < br ? ar : br) - (al > bl ? al : bl);
    const int32_t oh = (ab < bb ? ab : bb) - (at > bt ? at : bt);
    int64_t inter, uni;
    if (ow <= 0 || oh <= 0) return 0;
    inter = (int64_t)ow * oh;
    uni = (int64_t)a->w * a->h + (int64_t)b->w * b->h - inter;
    return inter * 100 > uni * (int64_t)pct;
}

int yolo_decode(uint32_t thresh_q16, uint32_t nms_pct, yolo_det_t *out, int max_out, int *dropped)
{
    int n = 0, lost = 0, li, a, y, x, k, i, j, m;

    /* 1. candidates, in the same order as the Python reference: layer, anchor, row, column, class */
    for (li = 0; li < DEC_NYOLO; li++) {
        const int8_t *raw = yolo_layer_output(DEC_SRC[li]);
        const int gw = DEC_GRID_W[li], gh = DEC_GRID_H[li], plane = gw * gh;
        for (a = 0; a < DEC_NA; a++) {
            const int8_t *base = raw + (size_t)a * DEC_K * plane;
            for (y = 0; y < gh; y++)
                for (x = 0; x < gw; x++) {
                    const int pix = y * gw + x;
                    const uint32_t obj = DEC_SIG[li][base[4 * plane + pix] + 128];
                    uint32_t cx, cy, w, h;
                    if (obj <= thresh_q16) continue;
                    cx = (((uint32_t)x * 65536u + DEC_SIG[li][base[0 * plane + pix] + 128]) << 4) / (uint32_t)gw;   /* Q16 -> Q20 */
                    cy = (((uint32_t)y * 65536u + DEC_SIG[li][base[1 * plane + pix] + 128]) << 4) / (uint32_t)gh;
                    w = DEC_EXPW[li][a][base[2 * plane + pix] + 128];
                    h = DEC_EXPH[li][a][base[3 * plane + pix] + 128];
                    for (k = 0; k < DEC_CLASSES; k++) {
                        const uint32_t p = (uint32_t)(((uint64_t)obj * DEC_SIG[li][base[(5 + k) * plane + pix] + 128] + 32768u) >> 16);
                        if (p <= thresh_q16) continue;
                        if (n >= YOLO_MAX_ENTRIES) { lost++; continue; }
                        s_ent[n].cls = (uint32_t)k; s_ent[n].prob = p;
                        s_ent[n].cx = cx; s_ent[n].cy = cy; s_ent[n].w = w; s_ent[n].h = h;
                        n++;
                    }
                }
        }
    }
    if (dropped) *dropped = lost;

    /* 2. per-class greedy NMS (darknet do_nms_sort semantics) */
    sort_dets(s_ent, n, 1);
    for (i = 0; i < n; i++) s_dead[i] = 0;
    for (i = 0; i < n; i++) {
        if (s_dead[i]) continue;
        for (j = i + 1; j < n && s_ent[j].cls == s_ent[i].cls; j++)
            if (!s_dead[j] && iou_above(&s_ent[i], &s_ent[j], nms_pct)) s_dead[j] = 1;
    }
    for (i = 0, m = 0; i < n; i++)
        if (!s_dead[i]) s_ent[m++] = s_ent[i];

    /* 3. highest probability first */
    sort_dets(s_ent, m, 0);
    if (m > max_out) m = max_out;
    for (i = 0; i < m; i++) out[i] = s_ent[i];
    return m;
}
