#!/usr/bin/env python3
"""
YOLOv3-tiny golden reference for the RISC-V accelerator project (step 1).

Subcommands
  run      parse darknet .cfg/.weights, fold BN, run float and fixed-point int8
           reference inference, dump every layer, decode + NMS, export the
           quantized model (weights/bias/layer table) for the C model / E203.
  compare  compare layer dumps written by the patched darknet against the
           float reference of this script (proves the reference is faithful).

Only needs numpy + Pillow.  Matches AlexeyAB/darknet `detector test` behaviour:
plain align-corners bilinear resize (letter_box=0), BN folded with
sqrt(var + 1e-5), leaky slope 0.1, maxpool pad = size-1, yolo decode + greedy
per-class NMS.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# --------------------------------------------------------------------------
# cfg parsing / network construction
# --------------------------------------------------------------------------
def parse_cfg(path):
    secs, cur = [], None
    for raw in open(path):
        line = raw.split('#')[0].strip()
        if not line:
            continue
        if line.startswith('['):
            cur = {'type': line.strip('[]').strip().lower()}
            secs.append(cur)
        else:
            k, v = line.split('=', 1)
            cur[k.strip()] = v.strip()
    return secs


def ints(s):
    return [int(v) for v in s.replace(' ', '').split(',') if v != '']


def build_layers(secs):
    net = secs[0]
    assert net['type'] in ('net', 'network'), 'first section must be [net]'
    W, H, C = int(net.get('width', 416)), int(net.get('height', 416)), int(net.get('channels', 3))
    layers, shapes = [], []
    for i, s in enumerate(secs[1:]):
        t = s['type']
        c, h, w = (C, H, W) if i == 0 else shapes[-1]
        L = {'idx': i, 'type': t, 'c_in': c, 'h_in': h, 'w_in': w}
        if t == 'convolutional':
            k = int(s['size'])
            stride = int(s.get('stride', 1))
            pad = int(s['padding']) if 'padding' in s else (k // 2 if int(s.get('pad', 0)) else 0)
            f = int(s['filters'])
            assert int(s.get('groups', 1)) == 1, 'grouped conv not supported'
            act = s.get('activation', 'linear')
            assert act in ('leaky', 'relu', 'linear'), 'unsupported activation ' + act
            L.update(filters=f, size=k, stride=stride, pad=pad,
                     bn=int(s.get('batch_normalize', 0)), activation=act)
            oh, ow = (h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1
            out = (f, oh, ow)
        elif t == 'maxpool':
            k = int(s['size'])
            stride = int(s.get('stride', 1))
            pad = int(s['padding']) if 'padding' in s else k - 1
            L.update(size=k, stride=stride, pad=pad)
            out = (c, (h + pad - k) // stride + 1, (w + pad - k) // stride + 1)
        elif t == 'route':
            assert 'groups' not in s, 'route groups not supported'
            srcs = [i + v if v < 0 else v for v in ints(s['layers'])]
            L['srcs'] = srcs
            ch = sum(shapes[j][0] for j in srcs)
            assert all(shapes[j][1:] == shapes[srcs[0]][1:] for j in srcs), 'route spatial mismatch'
            out = (ch,) + shapes[srcs[0]][1:]
        elif t == 'upsample':
            st = int(s.get('stride', 2))
            L['stride'] = st
            out = (c, h * st, w * st)
        elif t == 'yolo':
            anchors = ints(s['anchors'])
            L.update(mask=ints(s['mask']), anchors=anchors, classes=int(s['classes']))
            out = (c, h, w)
        else:
            raise ValueError('unsupported layer type [%s]' % t)
        L.update(c_out=out[0], h_out=out[1], w_out=out[2])
        layers.append(L)
        shapes.append(out)
    return layers, (C, H, W)


def load_weights(path, layers):
    with open(path, 'rb') as f:
        major, minor, _rev = struct.unpack('<3i', f.read(12))
        f.read(8 if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000 else 4)
        data = np.fromfile(f, dtype='<f4')
    p = 0

    def take(n):
        nonlocal p
        a = data[p:p + n].copy()
        if len(a) != n:
            raise ValueError('weights file too short (cfg/weights mismatch?)')
        p += n
        return a

    for L in layers:
        if L['type'] != 'convolutional':
            continue
        n, k, c = L['filters'], L['size'], L['c_in']
        L['bias'] = take(n)
        if L['bn']:
            L['scales'], L['mean'], L['var'] = take(n), take(n), take(n)
        L['w'] = take(n * c * k * k).reshape(n, c, k, k)
    if p != len(data):
        print('warning: %d floats left over in weights file' % (len(data) - p), file=sys.stderr)


def fold_bn(layers):
    """Same formula as darknet fuse_conv_batchnorm (sqrt(var + 1e-5))."""
    for L in layers:
        if L['type'] != 'convolutional':
            continue
        w, b = L['w'].astype(np.float64), L['bias'].astype(np.float64)
        if L['bn']:
            s = L['scales'].astype(np.float64) / np.sqrt(L['var'].astype(np.float64) + 1e-5)
            w = w * s[:, None, None, None]
            b = b - L['mean'].astype(np.float64) * s
        L['wf'], L['bf'] = w.astype(np.float32), b.astype(np.float32)


# --------------------------------------------------------------------------
# primitive ops (shared by float and integer paths)
# --------------------------------------------------------------------------
def conv_core(x, w, stride, pad):
    C, H, W = x.shape
    O, _, k, _ = w.shape
    oh, ow = (H + 2 * pad - k) // stride + 1, (W + 2 * pad - k) // stride + 1
    if k == 1 and stride == 1 and pad == 0:
        cols = x.reshape(C, -1)
    else:
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad))) if pad else x
        win = sliding_window_view(xp, (k, k), axis=(1, 2))[:, ::stride, ::stride][:, :oh, :ow]
        cols = win.transpose(0, 3, 4, 1, 2).reshape(C * k * k, oh * ow)
    out = w.reshape(O, -1).astype(x.dtype, copy=False) @ cols
    return out.reshape(O, oh, ow)


def maxpool(x, size, stride, pad):
    C, H, W = x.shape
    oh, ow = (H + pad - size) // stride + 1, (W + pad - size) // stride + 1
    lp = pad // 2                       # darknet: w_offset = -pad/2
    fill = -np.inf if x.dtype.kind == 'f' else -(2 ** 62)
    xp = np.pad(x, ((0, 0), (lp, pad - lp), (lp, pad - lp)), constant_values=fill)
    win = sliding_window_view(xp, (size, size), axis=(1, 2))[:, ::stride, ::stride][:, :oh, :ow]
    return win.max(axis=(3, 4))


def upsample(x, s):
    return np.repeat(np.repeat(x, s, axis=1), s, axis=2)


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


def yolo_activate(x, L):
    """darknet yolo layer forward: logistic on x,y, objectness and class scores."""
    n, k = len(L['mask']), 5 + L['classes']
    y = x.astype(np.float32).reshape(n, k, x.shape[1], x.shape[2]).copy()
    y[:, 0:2] = sigmoid(y[:, 0:2])
    y[:, 4:] = sigmoid(y[:, 4:])
    return y.reshape(x.shape)


def act_float(y, a):
    if a == 'leaky':
        return np.where(y > 0, y, np.float32(0.1) * y)
    if a == 'relu':
        return np.maximum(y, 0)
    return y


# --------------------------------------------------------------------------
# image preprocessing (darknet load_image + resize_image)
# --------------------------------------------------------------------------
def load_image_chw(path):
    from PIL import Image
    im = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    return (im.astype(np.float32) / 255.0).transpose(2, 0, 1).copy(), im.shape[1], im.shape[0]


def darknet_resize(img, w, h):
    c, ih, iw = img.shape
    if (iw, ih) == (w, h):
        return img.copy()
    f32 = np.float32
    ws, hs = f32(iw - 1) / f32(w - 1), f32(ih - 1) / f32(h - 1)
    sx = np.arange(w, dtype=f32) * ws
    ix = sx.astype(np.int64)
    dx = (sx - ix).astype(f32)
    part = (f32(1) - dx) * img[:, :, ix] + dx * img[:, :, np.minimum(ix + 1, iw - 1)]
    part[:, :, w - 1] = img[:, :, iw - 1]
    sy = np.arange(h, dtype=f32) * hs
    iy = sy.astype(np.int64)
    dy = (sy - iy).astype(f32)
    out = (f32(1) - dy)[None, :, None] * part[:, iy, :]
    add = dy[None, :, None] * part[:, np.minimum(iy + 1, ih - 1), :]
    add[:, h - 1, :] = 0                 # darknet skips the dy term on the last row
    return (out + add).astype(f32)


# --------------------------------------------------------------------------
# float forward
# --------------------------------------------------------------------------
def forward_float(layers, x0):
    outs, x = [], x0
    for L in layers:
        t = L['type']
        if t == 'convolutional':
            y = conv_core(x, L['wf'], L['stride'], L['pad']) + L['bf'][:, None, None]
            y = act_float(y, L['activation'])
        elif t == 'maxpool':
            y = maxpool(x, L['size'], L['stride'], L['pad'])
        elif t == 'route':
            y = np.concatenate([outs[j] for j in L['srcs']], axis=0)
        elif t == 'upsample':
            y = upsample(x, L['stride'])
        elif t == 'yolo':
            y = yolo_activate(x, L)
        outs.append(np.ascontiguousarray(y, dtype=np.float32))
        x = outs[-1]
    return outs


# --------------------------------------------------------------------------
# detection decode + NMS (darknet get_yolo_detections / do_nms_sort)
# --------------------------------------------------------------------------
def box_iou(a, b):
    def ov(x1, w1, x2, w2):
        return min(x1 + w1 / 2, x2 + w2 / 2) - max(x1 - w1 / 2, x2 - w2 / 2)
    w, h = ov(a[0], a[2], b[0], b[2]), ov(a[1], a[3], b[1], b[3])
    inter = 0.0 if (w < 0 or h < 0) else w * h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union


def decode(yolo_outs, yolo_layers, netw, neth, thresh, nms):
    boxes, probs = [], []
    for out, L in zip(yolo_outs, yolo_layers):
        n, k = len(L['mask']), 5 + L['classes']
        _, h, w = out.shape
        o = out.reshape(n, k, h, w)
        for a, m in enumerate(L['mask']):
            ys, xs = np.nonzero(o[a, 4] > thresh)
            for y, x in zip(ys, xs):
                obj = o[a, 4, y, x]
                bx, by = (x + o[a, 0, y, x]) / w, (y + o[a, 1, y, x]) / h
                bw = np.exp(o[a, 2, y, x]) * L['anchors'][2 * m] / netw
                bh = np.exp(o[a, 3, y, x]) * L['anchors'][2 * m + 1] / neth
                p = obj * o[a, 5:, y, x]
                p = np.where(p > thresh, p, 0.0)
                boxes.append((float(bx), float(by), float(bw), float(bh)))
                probs.append(p)
    if not boxes:
        return []
    probs = np.array(probs)
    dets = []
    for cls in range(probs.shape[1]):
        idx = [i for i in np.argsort(-probs[:, cls], kind='stable') if probs[i, cls] > 0]
        alive = np.ones(len(idx), bool)
        for ii in range(len(idx)):
            if not alive[ii]:
                continue
            for jj in range(ii + 1, len(idx)):
                if alive[jj] and box_iou(boxes[idx[ii]], boxes[idx[jj]]) > nms:
                    alive[jj] = False
        for ii in range(len(idx)):
            if alive[ii]:
                dets.append((cls, float(probs[idx[ii], cls]), boxes[idx[ii]]))
    dets.sort(key=lambda d: -d[1])
    return dets


def write_dets(path, dets, names, imw, imh):
    with open(path, 'w') as f:
        f.write('# class prob | rel cx cy w h | px left top w h (original image %dx%d)\n' % (imw, imh))
        for cls, p, (x, y, w, h) in dets:
            nm = names[cls] if names and cls < len(names) else str(cls)
            f.write('%-14s %6.2f%% | %.4f %.4f %.4f %.4f | %d %d %d %d\n' % (
                nm, 100 * p, x, y, w, h,
                round((x - w / 2) * imw), round((y - h / 2) * imh), round(w * imw), round(h * imh)))


def match_dets(a, b, iou=0.5):
    used, m = set(), 0
    for cls, _, bx in a:
        for j, (c2, _, b2) in enumerate(b):
            if j not in used and c2 == cls and box_iou(bx, b2) > iou:
                used.add(j)
                m += 1
                break
    return m


# --------------------------------------------------------------------------
# fixed-point (dynamic fixed point, power-of-two scales) int path
# --------------------------------------------------------------------------
def rnd(v):
    return np.floor(v + 0.5)             # round half up, deterministic in hardware


def frac_bits(maxabs, bits):
    qmax = 2 ** (bits - 1) - 1
    if maxabs <= 0:
        return bits - 1
    return int(np.floor(np.log2(qmax / maxabs)))


def root_of(layers, i):
    t = layers[i]['type']
    if t == 'convolutional':
        return i
    if t in ('maxpool', 'upsample'):
        return root_of(layers, i - 1)
    if t == 'route':
        return root_of(layers, layers[i]['srcs'][0])
    return None


def quantize_model(layers, calib_outs, bits, act_pct, w_pct):
    qmin, qmax = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    fa = {}
    for i, L in enumerate(layers):
        if L['type'] != 'convolutional':
            continue
        m = max(float(np.percentile(np.abs(o[i]), act_pct)) for o in calib_outs)
        fa[i] = frac_bits(m, bits)
    changed = True                       # concat inputs must share one scale
    while changed:
        changed = False
        for i, L in enumerate(layers):
            if L['type'] == 'route' and len(L['srcs']) > 1:
                roots = [root_of(layers, j) for j in L['srcs']]
                mn = min(fa[r] for r in roots)
                for r in roots:
                    if fa[r] != mn:
                        fa[r], changed = mn, True
    fa_out = {}
    for i, L in enumerate(layers):
        r = root_of(layers, i)
        fa_out[i] = fa[r] if r is not None else None
    fa_in = [bits - 1] + [fa_out[i] for i in range(len(layers) - 1)]
    for i, L in enumerate(layers):
        L['fa_in'], L['fa_out'] = fa_in[i], fa_out[i]
        if L['type'] != 'convolutional':
            continue
        wf = L['wf'].astype(np.float64)
        fw = frac_bits(float(np.percentile(np.abs(wf), w_pct)), bits)
        wq = np.clip(rnd(wf * 2.0 ** fw), qmin, qmax).astype(np.int64)
        bq = np.clip(rnd(L['bf'].astype(np.float64) * 2.0 ** (L['fa_in'] + fw)), -2 ** 31, 2 ** 31 - 1).astype(np.int64)
        L.update(fw=fw, wq=wq, bq=bq, shift=L['fa_in'] + fw - L['fa_out'])


def forward_int(layers, x0, bits):
    qmin, qmax = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    x = np.clip(rnd(x0.astype(np.float64) * 2.0 ** (bits - 1)), qmin, qmax).astype(np.int64)
    in_q = x
    outs = []
    for i, L in enumerate(layers):
        t = L['type']
        if t == 'convolutional':
            acc = conv_core(x.astype(np.float64), L['wq'], L['stride'], L['pad'])   # exact in float64
            acc = acc.astype(np.int64) + L['bq'][:, None, None]
            s = L['shift']
            y = ((acc + (1 << (s - 1))) >> s) if s > 0 else (acc << -s)
            if L['activation'] == 'relu':
                y = np.maximum(y, 0)
            elif L['activation'] == 'leaky':
                y = np.where(y < 0, (y * 13 + 64) >> 7, y)     # ~0.1 slope (13/128)
            y = np.clip(y, qmin, qmax)
        elif t == 'maxpool':
            y = maxpool(x, L['size'], L['stride'], L['pad'])
        elif t == 'route':
            y = np.concatenate([outs[j] for j in L['srcs']], axis=0)
        elif t == 'upsample':
            y = upsample(x, L['stride'])
        elif t == 'yolo':
            y = x                       # raw conv output kept; decoded in float below
        outs.append(y.astype(np.int64))
        x = outs[-1]
    return in_q, outs


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def export_model(layers, outdir, bits, net_shape, yolos):
    mdir = os.path.join(outdir, 'model')
    os.makedirs(mdir, exist_ok=True)
    wdt = np.int8 if bits <= 8 else np.int16
    wblob, bblob, woff, boff = [], [], 0, 0
    meta = []
    for L in layers:
        m = {k: L[k] for k in ('idx', 'type', 'c_in', 'h_in', 'w_in', 'c_out', 'h_out', 'w_out')}
        if L['type'] == 'convolutional':
            m.update(size=L['size'], stride=L['stride'], pad=L['pad'], activation=L['activation'],
                     fa_in=L['fa_in'], fw=L['fw'], fa_out=L['fa_out'], shift=L['shift'],
                     w_off=woff, w_count=int(L['wq'].size), b_off=boff, b_count=int(L['bq'].size))
            wblob.append(L['wq'].astype(wdt).tobytes())
            bblob.append(L['bq'].astype('<i4').tobytes())
            woff += L['wq'].size
            boff += L['bq'].size
        elif L['type'] == 'maxpool':
            m.update(size=L['size'], stride=L['stride'], pad=L['pad'], fa_out=L['fa_out'])
        elif L['type'] == 'route':
            m.update(srcs=L['srcs'], fa_out=L['fa_out'])
        elif L['type'] == 'upsample':
            m.update(stride=L['stride'], fa_out=L['fa_out'])
        elif L['type'] == 'yolo':
            m.update(mask=L['mask'], anchors=L['anchors'], classes=L['classes'], fa_in=L['fa_in'])
        meta.append(m)
    open(os.path.join(mdir, 'weights_q.bin'), 'wb').write(b''.join(wblob))
    open(os.path.join(mdir, 'bias_i32.bin'), 'wb').write(b''.join(bblob))
    json.dump({'bits': bits, 'input': list(net_shape), 'input_fa': bits - 1, 'layers': meta},
              open(os.path.join(mdir, 'layers.json'), 'w'), indent=1)

    T = {'convolutional': 'L_CONV', 'maxpool': 'L_MAXPOOL', 'route': 'L_ROUTE', 'upsample': 'L_UPSAMPLE', 'yolo': 'L_YOLO'}
    A = {'linear': 'ACT_LINEAR', 'relu': 'ACT_RELU', 'leaky': 'ACT_LEAKY'}
    h = ['/* auto-generated by yolov3_tiny_ref.py -- do not edit */',
         '#ifndef YOLO_LAYERS_H', '#define YOLO_LAYERS_H', '#include <stdint.h>', '',
         '#define NUM_LAYERS %d' % len(layers), '#define QBITS %d' % bits,
         '#define INPUT_C %d' % net_shape[0], '#define INPUT_H %d' % net_shape[1],
         '#define INPUT_W %d' % net_shape[2], '#define INPUT_FA %d  /* input = round(pixel/255 * 2^INPUT_FA) */' % (bits - 1), '',
         'enum { L_CONV, L_MAXPOOL, L_ROUTE, L_UPSAMPLE, L_YOLO };',
         'enum { ACT_LINEAR, ACT_RELU, ACT_LEAKY };', '',
         'typedef struct {',
         '    uint8_t  type, act, k, stride, pad;',
         '    uint16_t c_in, h_in, w_in, c_out, h_out, w_out;',
         '    int8_t   fa_in, fw, fa_out, shift;   /* fixed-point fraction bits; out = (acc + b) >> shift */',
         '    int16_t  src0, src1;                 /* route sources (absolute layer index), -1 if unused */',
         '    uint32_t w_off, b_off;               /* offsets in elements into weights_q.bin / bias_i32.bin */',
         '} layer_t;', '', 'static const layer_t LAYERS[NUM_LAYERS] = {']
    for m in meta:
        s = m['srcs'] if m['type'] == 'route' else []
        s0, s1 = (s + [-1, -1])[:2] if s else (-1, -1)
        if len(s) > 2:
            raise ValueError('route with >2 sources not supported in layers.h')
        h.append('    {%s, %s, %d, %d, %d, %d,%d,%d,%d,%d,%d, %d,%d,%d,%d, %d,%d, %d,%d},  /* %d */' % (
            T[m['type']], A.get(m.get('activation', 'linear'), 'ACT_LINEAR'),
            m.get('size', 0), m.get('stride', 0), m.get('pad', 0),
            m['c_in'], m['h_in'], m['w_in'], m['c_out'], m['h_out'], m['w_out'],
            m.get('fa_in', 0), m.get('fw', 0), m.get('fa_out', 0) or 0, m.get('shift', 0),
            s0, s1, m.get('w_off', 0), m.get('b_off', 0), m['idx']))
    h += ['};', '']
    for n, (i, L) in enumerate(yolos):
        h.append('#define YOLO%d_LAYER %d   /* its raw input comes from layer %d, fa = %d */' % (n, i, i - 1, L['fa_in']))
    L0 = yolos[0][1]
    h.append('#define YOLO_CLASSES %d' % L0['classes'])
    h.append('static const uint16_t YOLO_ANCHORS[%d][2] = {%s};' % (
        len(L0['anchors']) // 2, ', '.join('{%d,%d}' % (L0['anchors'][2 * j], L0['anchors'][2 * j + 1]) for j in range(len(L0['anchors']) // 2))))
    for n, (i, L) in enumerate(yolos):
        h.append('static const uint8_t YOLO%d_MASK[%d] = {%s};' % (n, len(L['mask']), ', '.join(map(str, L['mask']))))
    h += ['', '#endif', '']
    open(os.path.join(mdir, 'layers.h'), 'w').write('\n'.join(h))
    return woff, boff


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def prep_input(args, W, H, C):
    img, imw, imh = load_image_chw(args.image)
    if args.input_bin:
        x = np.fromfile(args.input_bin, dtype=np.float32)
        assert x.size == C * H * W, 'input.bin size mismatch'
        x = x.reshape(C, H, W)
    else:
        x = darknet_resize(img, W, H)
    return x, imw, imh


def dump_bins(dirpath, arrs, dtype):
    os.makedirs(dirpath, exist_ok=True)
    for name, a in arrs:
        a.astype(dtype).tofile(os.path.join(dirpath, name + '.bin'))


def cmd_run(args):
    secs = parse_cfg(args.cfg)
    layers, (C, H, W) = build_layers(secs)
    load_weights(args.weights, layers)
    fold_bn(layers)
    names = [l.strip() for l in open(args.names)] if args.names else None
    yolos = [(i, L) for i, L in enumerate(layers) if L['type'] == 'yolo']
    os.makedirs(args.out, exist_ok=True)

    x0, imw, imh = prep_input(args, W, H, C)
    print('float forward ...')
    fouts = forward_float(layers, x0)
    dump_bins(os.path.join(args.out, 'float'),
              [('input', x0)] + [('layer_%02d' % i, o) for i, o in enumerate(fouts)], np.float32)
    dets_f = decode([fouts[i] for i, _ in yolos], [L for _, L in yolos], W, H, args.thresh, args.nms)
    write_dets(os.path.join(args.out, 'detections_float.txt'), dets_f, names, imw, imh)
    print('  %d detections (float)' % len(dets_f))
    if args.no_int8:
        return

    bits = args.bits
    calib_paths = list(args.calib or [])
    if args.calib_dir:
        calib_paths += [os.path.join(args.calib_dir, f) for f in sorted(os.listdir(args.calib_dir))
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    calib_outs = [fouts]
    for p in calib_paths:
        img, _, _ = load_image_chw(p)
        calib_outs.append(forward_float(layers, darknet_resize(img, W, H)))
    print('calibrating activation ranges on %d image(s) ...' % len(calib_outs))
    if len(calib_outs) < 20:
        print('  note: use --calib-dir with 50-100 images from your target dataset for final numbers')
    quantize_model(layers, calib_outs, bits, args.act_pct, args.w_pct)
    del calib_outs

    print('int%d forward ...' % bits)
    in_q, iouts = forward_int(layers, x0, bits)
    idt = np.int8 if bits <= 8 else np.int16
    dump_bins(os.path.join(args.out, 'int%d' % bits),
              [('input', in_q)] + [('layer_%02d' % i, o) for i, o in enumerate(iouts) if layers[i]['type'] != 'yolo'], idt)

    deq = []
    for i, L in enumerate(layers):
        if L['type'] == 'yolo':
            deq.append(yolo_activate(deq[i - 1], L))
        else:
            deq.append((iouts[i].astype(np.float64) / 2.0 ** L['fa_out']).astype(np.float32))
    dets_q = decode([deq[i] for i, _ in yolos], [L for _, L in yolos], W, H, args.thresh, args.nms)
    write_dets(os.path.join(args.out, 'detections_int%d.txt' % bits), dets_q, names, imw, imh)

    nw, nb = export_model(layers, args.out, bits, (C, H, W), yolos)

    rep = ['layer type            out shape        fa_in  fw fa_out shift   SQNR(dB)  maxAbsErr  maxAbs(float)']
    for i, L in enumerate(layers):
        if L['type'] == 'yolo':
            continue
        ref, q = fouts[i].astype(np.float64), deq[i].astype(np.float64)
        err = ref - q
        sq = 10 * np.log10(max(np.sum(ref ** 2), 1e-30) / max(np.sum(err ** 2), 1e-30))
        rep.append('%3d  %-14s %-16s %5s %3s %5s %5s %10.2f %10.4f %10.4f' % (
            i, L['type'], '%dx%dx%d' % (L['c_out'], L['h_out'], L['w_out']),
            L.get('fa_in', '') if L['type'] == 'convolutional' else '',
            L.get('fw', ''), L['fa_out'], L.get('shift', ''), sq, np.abs(err).max(), np.abs(ref).max()))
    m = match_dets(dets_f, dets_q)
    rep.append('')
    rep.append('detections: float=%d int%d=%d matched(IoU>0.5, same class)=%d' % (len(dets_f), bits, len(dets_q), m))
    rep.append('exported: %d weights, %d biases (%d bytes of weights)' % (nw, nb, nw * (1 if bits <= 8 else 2)))
    open(os.path.join(args.out, 'report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    print('\noutputs in', args.out)



def load_quantized_model(layers, model_dir):
    """Load previously exported INT8 parameters without recalibration."""
    meta_path = os.path.join(model_dir, 'layers.json')
    with open(meta_path) as f:
        meta = json.load(f)

    bits = int(meta['bits'])
    expected_shape = tuple(meta['input'])
    if bits not in (8, 16):
        raise ValueError('Unsupported quantization bits: %d' % bits)

    weights_path = os.path.join(model_dir, 'weights_q.bin')
    biases_path = os.path.join(model_dir, 'bias_i32.bin')

    wdt = np.int8 if bits <= 8 else np.int16
    weights = np.fromfile(weights_path, dtype=wdt).astype(np.int64)
    biases = np.fromfile(biases_path, dtype='<i4').astype(np.int64)

    saved_layers = meta['layers']
    if len(saved_layers) != len(layers):
        raise ValueError('Layer count mismatch')

    for i, (L, M) in enumerate(zip(layers, saved_layers)):
        if M['idx'] != i or M['type'] != L['type']:
            raise ValueError('Layer metadata mismatch at layer %d' % i)

        if L['type'] == 'convolutional':
            wo, wc = int(M['w_off']), int(M['w_count'])
            bo, bc = int(M['b_off']), int(M['b_count'])

            if wc != L['wf'].size or bc != L['bf'].size:
                raise ValueError('Parameter size mismatch at layer %d' % i)

            L['wq'] = weights[wo:wo + wc].reshape(L['wf'].shape).copy()
            L['bq'] = biases[bo:bo + bc].copy()

            for key in ('fa_in', 'fw', 'fa_out', 'shift'):
                L[key] = int(M[key])

        elif L['type'] == 'yolo':
            L['fa_in'] = int(M['fa_in'])

        if M.get('fa_out') is not None:
            L['fa_out'] = int(M['fa_out'])

    if len(weights) != sum(
        int(M.get('w_count', 0)) for M in saved_layers
    ):
        raise ValueError('Unexpected weights_q.bin size')

    if len(biases) != sum(
        int(M.get('b_count', 0)) for M in saved_layers
    ):
        raise ValueError('Unexpected bias_i32.bin size')

    print('Loaded INT%d model from %s' % (bits, model_dir))
    print('Weights: %d elements | Biases: %d elements' %
          (len(weights), len(biases)))

    return bits


def cmd_infer_int8(args):
    secs = parse_cfg(args.cfg)
    layers, (C, H, W) = build_layers(secs)
    load_weights(args.weights, layers)
    fold_bn(layers)

    bits = load_quantized_model(layers, args.model)

    with open(os.path.join(args.model, 'layers.json')) as f:
        meta = json.load(f)

    if tuple(meta['input']) != (C, H, W):
        raise ValueError('Model input shape does not match cfg')

    names = [l.strip() for l in open(args.names)] if args.names else None
    yolos = [(i, L) for i, L in enumerate(layers)
             if L['type'] == 'yolo']

    img, imw, imh = load_image_chw(args.image)
    x0 = darknet_resize(img, W, H)

    print('INT%d forward using saved quantization parameters ...' % bits)
    in_q, iouts = forward_int(layers, x0, bits)

    idt = np.int8 if bits <= 8 else np.int16
    int_dir = os.path.join(args.out, 'int%d' % bits)

    dump_bins(
        int_dir,
        [('input', in_q)] +
        [('layer_%02d' % i, o)
         for i, o in enumerate(iouts)
         if layers[i]['type'] != 'yolo'],
        idt
    )

    deq = []
    for i, L in enumerate(layers):
        if L['type'] == 'yolo':
            deq.append(yolo_activate(deq[i - 1], L))
        else:
            deq.append(
                (iouts[i].astype(np.float64) /
                 2.0 ** L['fa_out']).astype(np.float32)
            )

    dets = decode(
        [deq[i] for i, _ in yolos],
        [L for _, L in yolos],
        W, H, args.thresh, args.nms
    )

    os.makedirs(args.out, exist_ok=True)
    write_dets(
        os.path.join(args.out, 'detections_int%d.txt' % bits),
        dets, names, imw, imh
    )

    print('%d detections' % len(dets))
    print('Outputs saved in', args.out)
    print('No calibration or model export was performed.')



def cmd_compare(args):
    tol = args.tol
    bad = 0
    names = ['input'] + ['layer_%02d' % i for i in range(1000)]
    print('%-10s %10s %12s %12s  %s' % ('blob', 'elems', 'maxAbsErr', 'err/maxAbs', 'status'))
    for nm in names:
        pd, pr = os.path.join(args.darknet, nm + '.bin'), os.path.join(args.ref, nm + '.bin')
        if nm != 'input' and not os.path.exists(pd):
            break
        if not (os.path.exists(pd) and os.path.exists(pr)):
            print('%-10s missing file' % nm)
            bad += 1
            continue
        a, b = np.fromfile(pd, np.float32), np.fromfile(pr, np.float32)
        if a.size != b.size:
            print('%-10s SIZE MISMATCH darknet=%d ref=%d' % (nm, a.size, b.size))
            bad += 1
            continue
        e = float(np.abs(a - b).max())
        rel = e / max(float(np.abs(a).max()), 1e-12)
        ok = rel < tol
        bad += not ok
        print('%-10s %10d %12.3e %12.3e  %s' % (nm, a.size, e, rel, 'ok' if ok else 'FAIL'))
    print('\nALL MATCH' if not bad else '\n%d blob(s) differ' % bad)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--cfg', required=True)
    r.add_argument('--weights', required=True)
    r.add_argument('--image', required=True, help='test image (also gives original size for box output)')
    r.add_argument('--input-bin', help='darknet DUMP_DIR/input.bin: use darknet\'s exact resized tensor')
    r.add_argument('--out', required=True)
    r.add_argument('--names', help='e.g. darknet/data/coco.names')
    r.add_argument('--calib', nargs='*', help='extra calibration images')
    r.add_argument('--calib-dir', help='folder of calibration images')
    r.add_argument('--bits', type=int, default=8)
    r.add_argument('--act-pct', type=float, default=99.99, help='percentile of |activation| used for range')
    r.add_argument('--w-pct', type=float, default=100.0, help='percentile of |weight| used for range')
    r.add_argument('--thresh', type=float, default=0.25)
    r.add_argument('--nms', type=float, default=0.45)
    r.add_argument('--no-int8', action='store_true')
    q = sub.add_parser('infer-int8')
    q.add_argument('--cfg', required=True)
    q.add_argument('--weights', required=True)
    q.add_argument('--model', required=True)
    q.add_argument('--image', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--names')
    q.add_argument('--thresh', type=float, default=0.25)
    q.add_argument('--nms', type=float, default=0.45)
    c = sub.add_parser('compare')
    c.add_argument('--darknet', required=True, help='DUMP_DIR written by patched darknet')
    c.add_argument('--ref', required=True, help='<out>/float from the run command')
    c.add_argument('--tol', type=float, default=1e-3, help='max error relative to layer max |value|')
    a = ap.parse_args()
    if a.cmd == 'compare':
        sys.exit(cmd_compare(a))
    elif a.cmd == 'infer-int8':
        sys.exit(cmd_infer_int8(a))
    else:
        sys.exit(cmd_run(a))


if __name__ == '__main__':
    main()
