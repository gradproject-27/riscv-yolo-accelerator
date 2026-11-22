#!/usr/bin/env python3
"""Compare detections_c.txt (C engine) with a detections_*.txt written by yolov3_tiny_ref.py.
usage: compare_dets.py --c detections_c.txt --ref ref_out/detections_int8.txt --names coco.names --thresh 0.25
Two detections match when they have the same class and IoU > --iou.  An unmatched detection is
  borderline-prob : its probability is within --edge of the threshold (rounding of the fixed-point product)
  NMS-edge        : a higher-probability detection of the same class overlaps it with IoU within --nms-edge of
                    the NMS threshold (greedy NMS flips on such ties)
  real            : anything else -> FAIL"""
import argparse
import sys


def iou(a, b):
    def ov(x1, w1, x2, w2):
        return min(x1 + w1 / 2, x2 + w2 / 2) - max(x1 - w1 / 2, x2 - w2 / 2)
    w, h = ov(a[0], a[2], b[0], b[2]), ov(a[1], a[3], b[1], b[3])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / (a[2] * a[3] + b[2] * b[3] - i)


def read_c(path):
    d = []
    for l in open(path):
        if l.startswith('#') or not l.strip():
            continue
        c, p, x, y, w, h = l.split()
        d.append((int(c), float(p), (float(x), float(y), float(w), float(h))))
    return d


def read_ref(path, names):
    idx = {n: i for i, n in enumerate(names)} if names else {}
    d = []
    for l in open(path):
        if l.startswith('#') or not l.strip():
            continue
        a, b, _ = l.split('|')
        nm, p = a.rsplit(None, 1)
        nm = nm.strip()
        cls = idx[nm] if nm in idx else int(nm)
        d.append((cls, float(p.rstrip('%')) / 100, tuple(float(v) for v in b.split())))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--names')
    ap.add_argument('--thresh', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.9)
    ap.add_argument('--edge', type=float, default=0.01)
    ap.add_argument('--nms', type=float, default=0.45)
    ap.add_argument('--nms-edge', type=float, default=0.01)
    a = ap.parse_args()
    names = [l.strip() for l in open(a.names)] if a.names else None
    C, R = read_c(a.c), read_ref(a.ref, names)
    used, pairs = set(), []
    for ci, (cc, cp, cb) in enumerate(C):
        best, bj = a.iou, None
        for j, (rc, rp, rb) in enumerate(R):
            if j not in used and rc == cc:
                v = iou(cb, rb)
                if v > best:
                    best, bj = v, j
        if bj is not None:
            used.add(bj)
            pairs.append((ci, bj))
    mc = {ci for ci, _ in pairs}
    un_c = [C[i] for i in range(len(C)) if i not in mc]
    un_r = [R[j] for j in range(len(R)) if j not in used]
    def nms_edge(d, others):
        return any(o[0] == d[0] and o[1] > d[1] and abs(iou(o[2], d[2]) - a.nms) < a.nms_edge for o in others)
    real_c = [d for d in un_c if d[1] > a.thresh + a.edge and not nms_edge(d, C + R)]
    real_r = [d for d in un_r if d[1] > a.thresh + a.edge and not nms_edge(d, C + R)]
    pd = max((abs(C[i][1] - R[j][1]) for i, j in pairs), default=0.0)
    bd = max((max(abs(C[i][2][k] - R[j][2][k]) for k in range(4)) for i, j in pairs), default=0.0)
    print('C engine: %d detections | reference: %d | matched: %d' % (len(C), len(R), len(pairs)))
    print('max |prob diff| on matches: %.4f   max box coord diff: %.4f (reference prints 4 decimals)' % (pd, bd))
    print('unmatched: C %d (explained by rounding at threshold / NMS edge: %d, real: %d) | reference %d (explained %d, real %d)' % (
        len(un_c), len(un_c) - len(real_c), len(real_c), len(un_r), len(un_r) - len(real_r), len(real_r)))
    for tag, lst in (('C only', real_c), ('ref only', real_r)):
        for d in lst[:10]:
            nm = names[d[0]] if names else d[0]
            print('  %s: %s %.3f %s' % (tag, nm, d[1], ['%.3f' % v for v in d[2]]))
    ok = not real_c and not real_r
    print('RESULT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
