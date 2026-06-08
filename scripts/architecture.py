#!/usr/bin/env python3
"""Generate architecture diagram for sf_seg.  Saves to docs/architecture.png/svg."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = '#0D1117'
C_IN    = '#2563EB'
C_HEAD  = '#7C3AED'
C_ATTN  = '#9D174D'
C_BLEND = '#059669'
C_FUSE  = '#0D9488'
C_OUT   = '#D97706'
C_SHAPE = '#FCD34D'
WHITE   = '#F1F5F9'
GRAY    = '#64748B'
LGRAY   = '#94A3B8'
ARROW   = '#94A3B8'

FW, FH = 24, 15


def setup():
    fig, ax = plt.subplots(figsize=(FW, FH))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis('off')
    return fig, ax


# ── Primitives ────────────────────────────────────────────────────────────────

def block(ax, cx, cy, w, h, color, title, sub='', tsz=8.5, ssz=6.5, alpha=0.93):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle="round,pad=0.1",
                       facecolor=color, edgecolor='white', linewidth=1.3,
                       alpha=alpha, zorder=3)
    ax.add_patch(r)
    dy = 0.15 if sub else 0
    ax.text(cx, cy+dy, title, ha='center', va='center',
            fontsize=tsz, fontweight='bold', color=WHITE, zorder=4)
    if sub:
        ax.text(cx, cy-0.22, sub, ha='center', va='center',
                fontsize=ssz, color=C_SHAPE, zorder=4, style='italic')


def arr(ax, x1, y1, x2, y2, lbl='', lside='right', loff=(0.12, 0), lsz=6.0, clr=ARROW, lw=1.4):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=clr, lw=lw,
                                connectionstyle='arc3,rad=0.0'), zorder=2)
    if lbl:
        mx, my = (x1+x2)/2 + loff[0], (y1+y2)/2 + loff[1]
        ha = 'left' if lside == 'right' else 'right'
        ax.text(mx, my, lbl, ha=ha, va='center', fontsize=lsz, color=C_SHAPE, zorder=5)


def cat_node(ax, cx, cy, r=0.22):
    """Circle for concat operation."""
    circ = plt.Circle((cx, cy), r, facecolor='#1E293B', edgecolor=WHITE, lw=1.2, zorder=4)
    ax.add_patch(circ)
    ax.text(cx, cy, 'cat', ha='center', va='center', fontsize=6, color=WHITE,
            fontweight='bold', zorder=5)


def section_label(ax, x, y, text):
    ax.text(x, y, text, ha='center', va='center', fontsize=7, color=LGRAY,
            fontstyle='italic', zorder=5)


# ── Main draw ─────────────────────────────────────────────────────────────────

def draw(ax):

    # Title
    ax.text(FW/2, FH-0.35, 'sf_seg  —  Multi-Scale Sparse-Focus Segmentation',
            ha='center', va='top', fontsize=14, fontweight='bold', color=WHITE)
    ax.text(FW/2, FH-0.80,
            'num_channels=64 · focus_size=32 · num_classes=81 · params ≈ 482K',
            ha='center', va='top', fontsize=8, color=LGRAY)

    # ── Columns ───────────────────────────────────────────────────────────────
    X0 = 1.8    # input
    X1 = 4.5    # head blocks
    X2 = 7.5    # attention outputs
    X3 = 10.0   # cat nodes
    X4 = 12.5   # fuse blocks
    X5 = 15.5   # d_med / d_lg
    X6 = 18.0   # blend_up + final chain
    X7 = 20.8   # final output steps

    # ── Y positions ───────────────────────────────────────────────────────────
    YS = 11.2   # small head
    YM = 7.2    # medium head
    YL = 3.2    # large head
    YF1 = (YS+YM)/2   # fuse1  9.2
    YF2 = (YM+YL)/2   # fuse2  5.2

    BW, BH = 2.3, 0.9    # standard block
    HW, HH = 2.4, 2.0    # head block (taller for internals)

    # ── Input ─────────────────────────────────────────────────────────────────
    YIN = (YS + YM) / 2
    block(ax, X0, YIN, BW, BH, C_IN, 'Input', '(B, 3, 224, 224)')

    # Resize arrows to each head
    for y_dst, resize, size in [(YS,'H/16','14²'), (YM,'H/4','56²'), (YL,'H','224²')]:
        ax.annotate('', xy=(X1-HW/2, y_dst), xytext=(X0+BW/2, YIN),
                    arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.2,
                                    connectionstyle='arc3,rad=0.0'), zorder=2)
        # resize label
        mx = (X0+BW/2 + X1-HW/2)/2 - 0.05
        my = (YIN + y_dst)/2
        ax.text(mx, my+0.1, f'{resize}', ha='right', va='bottom',
                fontsize=6, color=LGRAY, style='italic')
        ax.text(mx, my-0.05, f'({size})', ha='right', va='top',
                fontsize=5.5, color=GRAY, style='italic')

    # ── Attention heads ───────────────────────────────────────────────────────
    head_defs = [
        (YS, 'head_small',  '7×7',   'k=16 (33%)'),
        (YM, 'head_medium', '28×28', 'k=256 (33%)'),
        (YL, 'head_large',  '112×112','k=1024 (8%)'),
    ]
    internals = 'Conv(3→C, 3×3, ×2)\nConv(C→2C, 3×3)\nclamped_softmax\nchannel_mix (1×1)'

    for y, name, attn_sz, budget in head_defs:
        # Head block (taller)
        r = FancyBboxPatch((X1-HW/2, y-HH/2), HW, HH,
                           boxstyle="round,pad=0.1",
                           facecolor=C_HEAD, edgecolor='white',
                           linewidth=1.3, alpha=0.93, zorder=3)
        ax.add_patch(r)
        ax.text(X1, y+HH/2-0.25, name,
                ha='center', va='top', fontsize=8.5, fontweight='bold',
                color=WHITE, zorder=4)
        ax.text(X1, y-0.10, internals,
                ha='center', va='center', fontsize=5.8, color='#C4B5FD',
                linespacing=1.7, zorder=4)

        # Attention output
        short = name.split('_')[1]
        block(ax, X2, y, BW, BH, C_ATTN, f'a_{short}',
              f'(C, {attn_sz})\n{budget}', tsz=8, ssz=6)
        arr(ax, X1+HW/2, y, X2-BW/2, y)

    # ── Left bracket ──────────────────────────────────────────────────────────
    bx = 0.55
    ax.plot([bx,bx], [YL-HH/2, YS+HH/2], color=GRAY, lw=1.0, zorder=1)
    ax.plot([bx,bx+0.18], [YL-HH/2, YL-HH/2], color=GRAY, lw=1.0, zorder=1)
    ax.plot([bx,bx+0.18], [YS+HH/2, YS+HH/2], color=GRAY, lw=1.0, zorder=1)
    ax.text(bx-0.1, (YS+YL)/2, '3 independent\nattention heads\n(no shared weights)',
            ha='center', va='center', fontsize=6.2, color=LGRAY,
            rotation=90, zorder=5)

    # ── Decoder: fuse 1 (small + medium) ─────────────────────────────────────
    # a_small down to cat1
    arr(ax, X2, YS-BH/2, X2, YF1+0.22, lbl='upsample ×4', lside='right', loff=(0.12,0))
    # a_medium down to cat1
    arr(ax, X2, YM+BH/2, X2, YF1-0.22)
    ax.plot([X2, X3], [YF1+0.22, YF1+0.22], color=C_BLEND, lw=1.1, ls='--', zorder=2)
    ax.plot([X2, X3], [YF1-0.22, YF1-0.22], color=C_ATTN,  lw=1.1, ls='--', zorder=2)

    # blend_up_sm box (inline in the top dashed line)
    block(ax, (X2+X3)/2, YF1+0.22, 2.0, 0.55, C_BLEND,
          'blend_up_sm', 'Conv(C→C, 3×3)', tsz=7.5, ssz=5.8)

    cat_node(ax, X3, YF1)
    ax.plot([X3, X3], [YF1-0.22, YF1-0.22+0.22-0.22], color=ARROW, lw=1.2, zorder=2)
    ax.annotate('', xy=(X4-BW/2, YF1), xytext=(X3+0.22, YF1),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.4), zorder=2)
    section_label(ax, (X3+X4)/2, YF1+0.28, '(2C channels)')

    block(ax, X4, YF1, BW+0.3, BH+0.3, C_FUSE,
          'fuse_sm_med', 'Conv(2C→C, 3×3) × 2')
    block(ax, X5, YF1, BW+0.1, BH, C_FUSE, 'd_med', '(C, 28, 28)')
    arr(ax, X4+0.65+BW/2-0.65, YF1, X5-BW/2-0.05, YF1)

    # ── Decoder: fuse 2 (medium/d_med + large) ───────────────────────────────
    arr(ax, X5, YF1-BH/2, X5, YF2+0.22, lbl='upsample ×4', lside='right', loff=(0.12,0))
    arr(ax, X2, YL+BH/2, X2, YF2-0.22)
    ax.plot([X5, X3], [YF2+0.22, YF2+0.22], color=C_BLEND, lw=1.1, ls='--', zorder=2)
    ax.plot([X2, X3], [YF2-0.22, YF2-0.22], color=C_ATTN,  lw=1.1, ls='--', zorder=2)

    block(ax, (X5+X3)/2, YF2+0.22, 2.0, 0.55, C_BLEND,
          'blend_up_med', 'Conv(C→C, 3×3)', tsz=7.5, ssz=5.8)

    cat_node(ax, X3, YF2)
    ax.annotate('', xy=(X4-BW/2, YF2), xytext=(X3+0.22, YF2),
                arrowprops=dict(arrowstyle='->', color=ARROW, lw=1.4), zorder=2)
    section_label(ax, (X3+X4)/2, YF2+0.28, '(2C channels)')

    block(ax, X4, YF2, BW+0.3, BH+0.3, C_FUSE,
          'fuse_med_lg', 'Conv(2C→C//2, 3×3) × 2')
    block(ax, X5, YF2, BW+0.1, BH, C_FUSE, 'd_lg', '(C//2, 112, 112)')
    arr(ax, X4+0.65+BW/2-0.65, YF2, X5-BW/2-0.05, YF2)

    # ── Final chain ───────────────────────────────────────────────────────────
    Yp = YF2 - 1.7
    Ym = Yp  - 1.3
    Yo = Ym  - 1.4

    arr(ax, X5, YF2-BH/2, X5, Yp+BH/2, lbl='upsample ×2\n→ (H, W)', lside='right', loff=(0.12,0))
    block(ax, X5, Yp, BW+0.5, BH, C_BLEND, 'pre_masks', 'Conv(C//2→C//2, 3×3)')
    arr(ax, X5, Yp-BH/2, X5, Ym+BH/2)
    block(ax, X5, Ym, BW+0.5, BH, C_BLEND, 'masks', 'Conv(C//2→81, 1×1)')
    arr(ax, X5, Ym-BH/2, X5, Yo+BH/2+0.05)
    block(ax, X5, Yo, BW+0.5, BH+0.2, C_OUT, 'Output Logits',
          '(B, 81, 224, 224)', tsz=9.5)

    # ── Section dividers ──────────────────────────────────────────────────────
    for xd in [X1+HW/2+0.2, X2+BW/2+0.3, X4-BW/2-0.3]:
        ax.axvline(xd, color=GRAY, lw=0.5, ls=':', alpha=0.3, zorder=1)
    ax.text(X1, 0.4, 'Attention\nHeads', ha='center', fontsize=6.5, color=LGRAY)
    ax.text(X2, 0.4, 'Attended\nFeatures', ha='center', fontsize=6.5, color=LGRAY)
    ax.text((X3+X5)/2, 0.4, 'Bottom-Up Decoder', ha='center', fontsize=6.5, color=LGRAY)

    # ── Legend ────────────────────────────────────────────────────────────────
    items = [
        (C_IN,    'Input image'),
        (C_HEAD,  'Attention head  (enc→score/feat→attn→mix)'),
        (C_ATTN,  'Attended features  (attn × features)'),
        (C_BLEND, 'Blend / upsample convolution'),
        (C_FUSE,  'Fusion block  (concat + 2× conv)'),
        (C_OUT,   'Output logits'),
    ]
    lx, ly = 18.2, 12.8
    ax.text(lx, ly+0.25, 'Layer types', fontsize=8, fontweight='bold', color=WHITE)
    for i, (c, t) in enumerate(items):
        ry = ly - 0.65 - i * 0.65
        r = FancyBboxPatch((lx, ry-0.16), 0.45, 0.38,
                           boxstyle="round,pad=0.04",
                           facecolor=c, edgecolor='white', lw=0.9, alpha=0.93)
        ax.add_patch(r)
        ax.text(lx+0.58, ry+0.03, t, fontsize=6.5, color=WHITE, va='center')

    # Coverage note
    note = (
        'Attention budget k per channel:\n'
        '  head_small  (7×7=49)      k =  16  →  33%  [global context]\n'
        '  head_medium (28×28=784)   k = 256  →  33%  [mid-range]\n'
        '  head_large  (112×112=12544)k=1024  →   8%  [local selection]'
    )
    ax.text(18.2, 7.5, note, fontsize=6, color=LGRAY, va='top',
            family='monospace', linespacing=1.6)

    # Blending note
    blend_note = (
        'Feature blending points:\n'
        '  [1] channel_mix (1x1) - after attn x feat in each head\n'
        '  [2] blend_up_sm/med (3x3) - after each upsample\n'
        '  [3] fuse_sm_med / fuse_med_lg (3x3 x2) - concat fusion\n'
        '  [4] pre_masks (3x3) - before final 1x1 prediction'
    )
    ax.text(18.2, 5.6, blend_note, fontsize=6, color=LGRAY, va='top',
            family='monospace', linespacing=1.6)


def main():
    Path('docs').mkdir(exist_ok=True)
    fig, ax = setup()
    draw(ax)
    plt.tight_layout(pad=0.1)
    for ext in ('png', 'svg'):
        path = f'docs/architecture.{ext}'
        fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=BG)
        print(f'Saved: {path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
