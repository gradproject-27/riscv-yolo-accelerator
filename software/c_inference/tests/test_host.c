/* Host harness: runs the int8 engine on darknet's resized input, compares every layer
 * bit-for-bit with <ref_out>/int8/layer_XX.bin, then runs the integer decode + NMS and writes
 * the detections to a text file for compare_dets.py.
 *   ./test_host <ref_out/model> <input.bin (float32 CHW)> <ref_out/int8> [thresh=0.25] [nms=0.45] [dets_out=detections_c.txt] */
#include "yolo_int8.h"
#include "yolo_decode.h"
#include "layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    void *p;
    long sz;
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    p = malloc((size_t)sz + 1);
    if (fread(p, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    if (n) *n = (size_t)sz;
    return p;
}

typedef struct { const char *refdir; int bad; } ctx_t;

static void check(int idx, const int8_t *out, size_t n, void *user)
{
    ctx_t *c = (ctx_t *)user;
    char path[1024];
    size_t rn, i;
    int8_t *ref;
    snprintf(path, sizeof path, "%s/layer_%02d.bin", c->refdir, idx);
    ref = (int8_t *)slurp(path, &rn);
    if (rn != n) { printf("layer %2d  SIZE MISMATCH engine=%zu ref=%zu\n", idx, n, rn); c->bad++; free(ref); return; }
    for (i = 0; i < n && out[i] == ref[i]; i++) ;
    if (i == n) printf("layer %2d  %8zu elems  bit-exact\n", idx, n);
    else {
        const layer_t *L = &LAYERS[idx];
        size_t hw = (size_t)L->h_out * L->w_out;
        printf("layer %2d  MISMATCH at elem %zu (c=%zu y=%zu x=%zu): engine=%d ref=%d\n", idx, i,
               i / hw, (i % hw) / L->w_out, i % L->w_out, out[i], ref[i]);
        c->bad++;
    }
    free(ref);
}

int main(int argc, char **argv)
{
    size_t nw, nb, peak, nin, expect_w = 0;
    int8_t *w;
    int32_t *b;
    void *in;
    size_t input_elems = (size_t)INPUT_C * INPUT_H * INPUT_W;
    char path[1024];
    ctx_t ctx = { 0, 0 };
    clock_t t0;
    int i;

    double thresh = 0.25, nms = 0.45;
    const char *dets_path = "detections_c.txt";
    if (argc < 4 || argc > 7) { fprintf(stderr, "usage: %s <model_dir> <input.bin> <ref_int8_dir> [thresh] [nms] [dets_out]\n", argv[0]); return 2; }
    if (argc > 4) thresh = atof(argv[4]);
    if (argc > 5) nms = atof(argv[5]);
    if (argc > 6) dets_path = argv[6];
    ctx.refdir = argv[3];
    snprintf(path, sizeof path, "%s/weights_q.bin", argv[1]); w = (int8_t *)slurp(path, &nw);
    snprintf(path, sizeof path, "%s/bias_i32.bin", argv[1]);  b = (int32_t *)slurp(path, &nb);
    in = (float *)slurp(argv[2], &nin);
    for (i = 0; i < NUM_LAYERS; i++)
        if (LAYERS[i].type == L_CONV) expect_w += (size_t)LAYERS[i].c_out * LAYERS[i].c_in * LAYERS[i].k * LAYERS[i].k;
    if (nw != expect_w ||
        (nin != input_elems && nin != input_elems * sizeof(float))) {
        fprintf(stderr,
            "file size mismatch: weights %zu (expect %zu), input %zu "
            "(expect %zu for INT8 or %zu for FLOAT32)\n",
            nw, expect_w, nin,
            input_elems, input_elems * sizeof(float));
        return 2;
    }
    if (yolo_init(&peak)) { fprintf(stderr, "yolo_init failed (arena too small? need > %u)\n", YOLO_ARENA_BYTES); return 2; }
    printf("arena peak: %zu bytes (%.2f MiB); weights %.2f MiB; input %d bytes\n",
           peak, peak / 1048576.0, nw / 1048576.0, INPUT_C * INPUT_H * INPUT_W);

    if (nin == input_elems) {
    /* Input is already quantized INT8 */
    memcpy(yolo_input(), in, input_elems);
    printf("input format: INT8 (already quantized)\n");
    } else {
        /* Input is FLOAT32: quantize it here */
        yolo_quantize_f32((const float *)in, yolo_input(), input_elems);
        printf("input format: FLOAT32 (quantized by C engine)\n");
    }

    {
        size_t n;
        char p2[1024];
        int8_t *ref;

        snprintf(p2, sizeof p2, "%s/input.bin", ctx.refdir);
        ref = (int8_t *)slurp(p2, &n);

        if (!ref || n != input_elems) {
            printf("input     MISMATCH (reference input size: %zu)\n", n);
            ctx.bad++;
        } else {
            int match = !memcmp(ref, yolo_input(), input_elems);
            printf("input     %s\n", match ? "bit-exact" : "MISMATCH");
            if (!match) ctx.bad++;
        }

        free(ref);
    }
    
    t0 = clock();
    if (yolo_forward(w, b, check, &ctx)) { fprintf(stderr, "forward failed\n"); return 2; }
    printf("forward: %.2f s on host\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
    {   /* integer decode + NMS */
        static yolo_det_t dets[4096];
        int dropped = 0, nd, d;
        FILE *fo = fopen(dets_path, "w");
        t0 = clock();
        nd = yolo_decode((uint32_t)(thresh * 65536.0 + 0.5), (uint32_t)(nms * 100.0 + 0.5), dets, 4096, &dropped);
        printf("decode: %d detections (thresh %.3f, nms %.2f), dropped=%d, %.3f s\n", nd, thresh, nms, dropped, (double)(clock() - t0) / CLOCKS_PER_SEC);
        if (dropped) { printf("WARNING: candidate buffer overflowed, raise YOLO_MAX_ENTRIES or the threshold\n"); ctx.bad++; }
        if (!fo) { fprintf(stderr, "cannot write %s\n", dets_path); return 2; }
        fprintf(fo, "# class prob cx cy w h (relative to the network input)\n");
        for (d = 0; d < nd; d++)
            fprintf(fo, "%u %.6f %.6f %.6f %.6f %.6f\n", dets[d].cls, dets[d].prob / 65536.0, dets[d].cx / 1048576.0,
                    dets[d].cy / 1048576.0, dets[d].w / 1048576.0, dets[d].h / 1048576.0);
        fclose(fo);
        printf("wrote %s\n", dets_path);
    }
    free(w); free(b); free(in);
    printf(ctx.bad ? "\n%d mismatch(es)\n" : "\nALL LAYERS BIT-EXACT\n", ctx.bad);
    return ctx.bad ? 1 : 0;
}
