#!/usr/bin/env python3
"""Sinh diagram kiến trúc hiện tại (SAED) → docs/saed_*.png.

3 hình: (1) forward flow end-to-end, (2) sparse attention head,
(3) ensemble branch. Style dark theme khớp docs/ sẵn có.

    python -m scripts.diagram_saed
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG='#0D1117'; WHITE='#F1F5F9'; SH='#FCD34D'; GRAY='#64748B'; LGRAY='#94A3B8'; ARROW='#94A3B8'
C_IN='#2563EB'; C_BB='#1D4ED8'; C_HEAD='#7C3AED'; C_ATTN='#9D174D'; C_DEC='#0D9488'
C_ENS='#BE185D'; C_PRES='#059669'; C_OUT='#D97706'; C_LOSS='#B45309'
DOCS = Path(__file__).resolve().parent.parent / 'docs'


def setup(fw, fh):
    fig, ax = plt.subplots(figsize=(fw, fh)); fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.set_xlim(0, fw); ax.set_ylim(0, fh); ax.axis('off')
    return fig, ax


def box(ax, cx, cy, w, h, color, title, sub='', tsz=8.5, ssz=6.2):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.08",
                facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.93, zorder=3))
    dy = 0.16*h if sub else 0
    ax.text(cx, cy+dy, title, ha='center', va='center', fontsize=tsz,
            fontweight='bold', color=WHITE, zorder=4)
    if sub:
        for i, s in enumerate(sub.split('\n')):
            ax.text(cx, cy-0.12-0.22*i, s, ha='center', va='center', fontsize=ssz,
                    color=SH, zorder=4, style='italic')


def arr(ax, x1, y1, x2, y2, lbl='', clr=ARROW, lw=1.5, rad=0.0, lsz=6.2, loff=(0,0.14)):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=clr, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'), zorder=2)
    if lbl:
        ax.text((x1+x2)/2+loff[0], (y1+y2)/2+loff[1], lbl, ha='center', va='center',
                fontsize=lsz, color=LGRAY, zorder=5, style='italic')


def oplus(ax, cx, cy, r=0.18, sym='+'):
    ax.add_patch(plt.Circle((cx, cy), r, facecolor='#1E293B', edgecolor=WHITE, lw=1.2, zorder=5))
    ax.text(cx, cy, sym, ha='center', va='center', fontsize=9, color=WHITE, fontweight='bold', zorder=6)


def title(ax, fw, fh, t, s):
    ax.text(fw/2, fh-0.3, t, ha='center', va='top', fontsize=14, fontweight='bold', color=WHITE)
    ax.text(fw/2, fh-0.78, s, ha='center', va='top', fontsize=8, color=LGRAY)


# ── 1. Forward flow end-to-end ──────────────────────────────────────────────
def forward_flow():
    FW, FH = 22, 13
    fig, ax = setup(FW, FH)
    title(ax, FW, FH, 'SF-Seg V2 + SAED  —  Forward Flow',
          'micro C=32 · 32/8/8 masks · ensemble · ~5.2M params · loss tính @H/2, eval full-res')
    XIN, XBB, XHD, XAT, XDEC, XENS, XCOR, XOUT = 1.6, 4.0, 6.7, 9.3, 12.3, 12.3, 16.2, 19.4
    # input
    box(ax, XIN, 6.5, 2.0, 0.9, C_IN, 'Input', '(B,3,H,W)')
    # backbone — 5 features
    feats = [('f4','C4,H/32',10.6),('f3','C3,H/16',9.0),('f2','C2,H/8',7.4),
             ('f1','C1,H/4',5.0),('f_detail','32,H/2',2.6)]
    box(ax, XBB, 6.5, 1.9, 9.2, C_BB, '', '')
    ax.text(XBB, 11.7, 'SFBackbone', ha='center', fontsize=9, fontweight='bold', color=WHITE)
    ax.text(XBB, 11.35, 'ConvNeXt, IN-1k pretrain', ha='center', fontsize=6, color=SH, style='italic')
    arr(ax, XIN+1.0, 6.5, XBB-0.95, 6.5)
    for name, sh, y in feats:
        ax.text(XBB, y, name, ha='center', fontsize=7.5, color=WHITE, fontweight='bold')
        ax.text(XBB, y-0.32, sh, ha='center', fontsize=5.6, color=SH, style='italic')
    # heads
    heads = [('head_tiny','32 masks · self',10.6,C_ATTN),('head_small','8 masks · self',9.0,C_ATTN),
             ('head_medium','8 masks · cross←f4',7.4,C_ATTN),('head_large','4 masks · cross←f4',5.0,C_ATTN)]
    for nm, sub, y, col in heads:
        box(ax, XHD, y, 2.3, 1.05, col, nm, sub, tsz=7.5, ssz=5.6)
        arr(ax, XBB+0.95, y, XHD-1.15, y, lw=1.2)
    # presence from f4
    box(ax, XHD, 12.0, 2.3, 0.8, C_PRES, 'GAP+GMP → presence', 'BCE multi-label', tsz=6.8, ssz=5.4)
    arr(ax, XBB+0.4, 11.0, XHD-1.0, 12.0, rad=0.2, lw=1.1)
    # attention outputs → converge
    ax.text(XAT, 11.5, 'a_tiny/small/medium/large', ha='center', fontsize=6.5, color=LGRAY, style='italic')
    for y in [10.6,9.0,7.4,5.0]:
        arr(ax, XHD+1.2, y, XAT-0.2, y, lw=1.1)
    # DECODER BASE
    box(ax, XDEC, 8.0, 2.5, 2.4, C_DEC, 'DECODER BASE', 'fuse bottom-up\ntiny→small→medium→large\n+ ctx (mid-fusion)\ndecoder_dim=256 @H/4', tsz=8, ssz=5.6)
    for y in [10.6,9.0,7.4,5.0]:
        arr(ax, XAT+0.1, y, XDEC-1.3, 8.0, rad=0.05, lw=1.0)
    arr(ax, XHD, 11.6, XDEC-0.6, 9.25, rad=-0.2, lw=1.0, clr=C_PRES)   # ctx
    # detail branch
    arr(ax, XBB+0.4, 2.6, XDEC, 6.7, rad=-0.25, lw=1.0)
    ax.text(XDEC-1.4, 4.0, 'f_detail (biên @H/2)', ha='center', fontsize=5.6, color=LGRAY, style='italic', rotation=90)
    # ENSEMBLE BRANCH
    box(ax, XENS, 3.0, 2.5, 2.0, C_ENS, 'ENSEMBLE BRANCH', 'mỗi mask = weak predictor\nclassifier · gate vùng attend\nΣ(pred·gate) = ens_logit', tsz=8, ssz=5.5)
    for y in [10.6,9.0,7.4]:
        arr(ax, XAT+0.1, y, XENS-1.3, 3.0, rad=-0.15, lw=0.9, clr=C_ENS)
    # decoder logit + presence late
    box(ax, XCOR, 8.0, 1.0, 0.8, C_DEC, 'logit', '@H/2', tsz=7, ssz=5.4)
    arr(ax, XDEC+1.25, 8.0, XCOR-0.5, 8.0)
    oplus(ax, XCOR, 9.4, sym='+'); ax.text(XCOR+0.55, 9.4, 'late_ctx(presence)', fontsize=5.6, color=C_PRES, va='center')
    arr(ax, XHD, 12.0, XCOR-0.1, 9.6, rad=-0.3, lw=0.9, clr=C_PRES)
    arr(ax, XCOR, 9.2, XCOR, 8.45)
    # correction
    box(ax, XCOR, 5.6, 1.7, 1.0, C_OUT, 'ens_correct', 'concat[dec,ens]\nzero-init', tsz=7, ssz=5.4)
    arr(ax, XCOR, 7.5, XCOR, 6.15, lw=1.1)
    arr(ax, XENS+1.25, 3.0, XCOR-0.85, 5.3, rad=0.1, lw=1.0, clr=C_ENS)
    oplus(ax, XCOR, 4.0, sym='+')
    arr(ax, XCOR, 5.1, XCOR, 4.2, lw=1.1, lbl='', loff=(0.5,0))
    arr(ax, XCOR-0.85, 7.6, XCOR-0.3, 4.05, rad=-0.3, lw=0.9)   # decoder residual skip
    ax.text(XCOR-1.2, 5.9, 'final = dec + correct', fontsize=5.6, color=LGRAY, style='italic', rotation=90)
    # output
    box(ax, XOUT, 4.0, 1.9, 1.0, C_OUT, 'Output', 'bilinear↑\n(B,150,H,W)', tsz=8, ssz=5.6)
    arr(ax, XCOR+0.2, 4.0, XOUT-1.0, 4.0)
    fig.savefig(DOCS/'saed_01_forward.png', dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


# ── 2. Sparse attention head ─────────────────────────────────────────────────
def sparse_head():
    FW, FH = 18, 10
    fig, ax = setup(FW, FH)
    title(ax, FW, FH, 'SparseAttnHead  —  top-k sparse attention + per-mask weak predictor',
          'decoupled qk_dim=32 · temperature τ học được · budget ladder · top-k softmax (sparse thật)')
    box(ax, 1.6, 5.0, 1.7, 1.0, C_IN, 'feature', 'f (C,h,w)')
    # Q K V
    box(ax, 4.2, 7.5, 1.6, 0.8, C_HEAD, 'q_proj', 'M×32', tsz=7.5, ssz=5.8)
    box(ax, 4.2, 6.2, 1.6, 0.8, C_HEAD, 'k_proj', 'M×32', tsz=7.5, ssz=5.8)
    box(ax, 4.2, 4.0, 1.6, 0.8, C_HEAD, 'v_proj', 'C (chia M)', tsz=7.5, ssz=5.8)
    for y in [7.5,6.2,4.0]: arr(ax, 2.45,5.0, 3.4,y, lw=1.1)
    box(ax, 6.5, 9.0, 2.2, 0.7, C_PRES, '+ PE (sin-cos 2D)', 'vào Q/K, V sạch', tsz=6.8, ssz=5.4)
    arr(ax, 4.2,7.9, 5.8,9.0, rad=0.2, lw=0.9, clr=C_PRES)
    # sim
    box(ax, 6.9, 6.85, 1.8, 0.9, C_ATTN, 'Q·Kᵀ · scale / τ', 'sim (M,N,N_k)', tsz=7, ssz=5.6)
    arr(ax, 5.0,7.5, 6.1,7.0, lw=1.1); arr(ax, 5.0,6.2, 6.1,6.7, lw=1.1)
    # clamp
    box(ax, 9.5, 6.85, 2.0, 1.0, C_ATTN, 'top-k softmax', 'k/mask (ladder)', tsz=7.5, ssz=5.5)
    arr(ax, 7.8,6.85, 8.5,6.85, lw=1.3)
    ax.text(9.5, 5.55, 'giữ top-k → softmax(Σ=1)', ha='center', fontsize=5.8, color=LGRAY, style='italic')
    ax.text(9.5, 5.2, 'còn lại = 0 (focus đúng k điểm)', ha='center', fontsize=5.8, color='#34D399', style='italic')
    # attn @ V
    box(ax, 12.6, 5.6, 1.7, 0.9, C_DEC, 'attn @ V', 'out (M,dv,h,w)', tsz=7.5, ssz=5.6)
    arr(ax, 10.5,6.5, 11.75,5.9, lw=1.2)
    arr(ax, 4.2,3.6, 12.0,5.15, rad=-0.2, lw=1.0)
    # out_proj
    box(ax, 15.4, 5.6, 1.7, 0.9, C_DEC, 'out_proj', '→ a (C,h,w)', tsz=7.5, ssz=5.6)
    arr(ax, 13.45,5.6, 14.55,5.6, lw=1.2)
    # per-mask outputs
    box(ax, 12.6, 3.0, 2.0, 0.95, C_ENS, 'per_mask_feat', '(M,dv,h,w)\ngate=max-attn', tsz=7, ssz=5.4)
    arr(ax, 12.6,5.15, 12.6,3.5, lw=1.0, clr=C_ENS)
    ax.text(15.4, 3.0, '→ ENSEMBLE\nbranch', ha='center', fontsize=7, color=C_ENS, fontweight='bold')
    arr(ax, 13.6,3.0, 14.4,3.0, lw=1.0, clr=C_ENS)
    # diversity note
    ax.text(9.0, 8.0, 'diversity loss: phạt masks giống nhau → mỗi mask attend vùng khác',
            ha='center', fontsize=6.2, color=SH, style='italic')
    fig.savefig(DOCS/'saed_02_sparse_head.png', dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


# ── 3. Ensemble branch ───────────────────────────────────────────────────────
def ensemble_branch():
    FW, FH = 17, 10
    fig, ax = setup(FW, FH)
    title(ax, FW, FH, 'SAED Ensemble  —  region-gated weak predictors',
          'mỗi mask predict một phần (vùng nó attend) → gộp + correction sửa sai · đánh đuôi dài')
    # 3 masks ví dụ
    for i,(y,nm) in enumerate([(8.0,'mask A'),(6.3,'mask B'),(4.6,'mask C')]):
        box(ax, 2.0, y, 1.6, 0.85, C_ATTN, nm, 'feat+gate', tsz=7, ssz=5.4)
        box(ax, 4.6, y, 1.7, 0.85, C_HEAD, 'classifier', 'dv→C', tsz=7, ssz=5.4)
        arr(ax, 2.85,y, 3.7,y, lw=1.1)
        box(ax, 7.3, y, 1.9, 0.85, C_ENS, 'pred × gate', 'chỉ vùng attend', tsz=6.8, ssz=5.3)
        arr(ax, 5.5,y, 6.3,y, lw=1.1)
        arr(ax, 8.3,y, 9.4,6.3, rad=(0.12 if i!=1 else 0), lw=1.0, clr=C_ENS)
        # per-mask loss
        ax.text(7.3, y-0.62, 'CE gated → L_mask', ha='center', fontsize=5.4, color=SH, style='italic')
    ax.text(2.0, 9.0, '… 48 masks (tiny+small+medium)', ha='center', fontsize=6.5, color=LGRAY, style='italic')
    # combine
    box(ax, 10.0, 6.3, 2.0, 1.1, C_ENS, 'Σ(pred·gate)\n/ Σgate', 'ensemble_logit', tsz=7.5, ssz=5.6)
    # decoder logit
    box(ax, 10.0, 3.4, 2.0, 0.9, C_DEC, 'decoder_logit', '(đường base)', tsz=7, ssz=5.4)
    # correction
    box(ax, 13.2, 4.85, 2.0, 1.1, C_OUT, 'ens_correct', 'concat → refine\nzero-init', tsz=7.5, ssz=5.5)
    arr(ax, 11.0,6.3, 12.2,5.2, lw=1.2, clr=C_ENS)
    arr(ax, 11.0,3.4, 12.2,4.5, lw=1.2, clr=C_DEC)
    oplus(ax, 13.2, 3.0, sym='+')
    arr(ax, 13.2,4.3, 13.2,3.2, lw=1.1)
    arr(ax, 11.0,3.2, 12.95,3.0, rad=-0.2, lw=0.9, clr=C_DEC)
    box(ax, 15.6, 3.0, 1.6, 0.9, C_OUT, 'final', 'dec+correct', tsz=7.5, ssz=5.4)
    arr(ax, 13.4,3.0, 14.8,3.0, lw=1.2)
    ax.text(8.5, 1.4, 'Class hiếm: nhiều mask attend cùng vùng → vote nhiều lần + decoder base + correction → "tư duy ≥2 lần"',
            ha='center', fontsize=6.4, color=SH, style='italic')
    fig.savefig(DOCS/'saed_03_ensemble.png', dpi=140, bbox_inches='tight', facecolor=BG)
    plt.close(fig)


if __name__ == '__main__':
    forward_flow(); sparse_head(); ensemble_branch()
    print('saved: docs/saed_01_forward.png, saed_02_sparse_head.png, saed_03_ensemble.png')
