"""Custom CUDA ops for sf_seg.

First import triggers JIT compilation (~60 s); result is cached in
~/.cache/torch_extensions/ for subsequent runs.
Falls back to pure-PyTorch automatically when NVCC is unavailable.
"""
from __future__ import annotations

import os
import warnings

import torch
import torch.nn.functional as F

_here  = os.path.dirname(os.path.abspath(__file__))
_op    = None
_ready = False


def _build() -> None:
    global _op, _ready
    try:
        # Ensure venv binaries (ninja) are visible to subprocess
        import os
        _venv = os.path.join(_here, '..', '..', 'venv', 'bin')
        _venv = os.path.normpath(_venv)
        if os.path.isdir(_venv) and _venv not in os.environ.get('PATH', ''):
            os.environ['PATH'] = _venv + os.pathsep + os.environ.get('PATH', '')

        from torch.utils.cpp_extension import load as _jit
        _op = _jit(
            name='sf_clamped_softmax',
            sources=[os.path.join(_here, 'clamped_softmax_cuda.cu')],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
        _ready = True
    except Exception as exc:
        warnings.warn(
            f"[sf_ops] CUDA build failed ({type(exc).__name__}: {exc}). "
            "Falling back to PyTorch — correctness unchanged.",
            stacklevel=3,
        )


# ─── Pure-PyTorch reference (fallback when CUDA extension is unavailable) ─────

def _py_clamped_softmax(score: torch.Tensor, k: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (out, lam) — same interface as CUDA forward."""
    orig  = score.dtype
    s     = score.float()
    L, ik = s.shape[-1], int(k)
    p     = F.softmax(s, dim=-1) * k
    tv    = torch.topk(p, k=ik, dim=-1).values
    ts    = torch.sort(tv, dim=-1, descending=True).values
    cs    = ts.cumsum(dim=-1)
    j     = torch.arange(1, ik + 1, device=s.device, dtype=torch.float32)
    lj    = (j - cs) / (L - j).clamp(min=1e-6)
    jsat  = (ts - 1.0 >= lj).long().sum(dim=-1, keepdim=True)
    lam   = torch.gather(lj, -1, (jsat - 1).clamp(min=0))
    lam   = lam * (jsat > 0).float()          # (N, 1)
    out   = (p - lam).clamp(0.0, 1.0).to(orig)
    return out, lam.squeeze(-1)               # out: (N,L), lam: (N,)


# ─── autograd.Function: CUDA forward + analytical backward ────────────────────

class _ClampedSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, score: torch.Tensor, k: float, n_bisect: int):
        shape = score.shape
        flat  = score.reshape(-1, shape[-1])
        out, lam = _op.forward(flat, k, n_bisect)   # out: (N,L) fp16/fp32, lam: (N,) fp32
        ctx.save_for_backward(score, lam)
        ctx.k = k
        return out.reshape(shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        score, lam = ctx.saved_tensors   # lam: (N,) fp32
        k  = ctx.k
        M  = score.reshape(-1, score.shape[-1]).shape[0]
        L  = score.shape[-1]

        with torch.no_grad():
            s    = score.reshape(M, L).float()
            g    = grad.reshape(M, L).float()
            lam_ = lam.view(M, 1)   # (M, 1) for broadcasting

            # Recompute z = softmax(score)*k  (one softmax pass — cheaper than topk)
            z = F.softmax(s, dim=-1) * k   # (M, L)

            # Support mask: positions where 0 < z_i − λ* < 1
            eps  = 1e-5
            diff = z - lam_
            sm   = ((diff > eps) & (diff < 1.0 - eps)).float()  # (M, L)

            # Analytical Jacobian-vector product of clamped_softmax:
            #   dx_j = z_j · [ sm_j·(g_j − G_S/|S|)  +  (G_S·B_S − A_S·|S|)/(|S|·k) ]
            # where sums (G_S, A_S, B_S) are over the support set S.
            S    = sm.sum(dim=-1, keepdim=True).clamp_(min=1.0)
            mg   = g * sm                                        # g, masked to S
            G_S  = mg.sum(dim=-1, keepdim=True)
            A_S  = (mg * z).sum(dim=-1, keepdim=True)
            B_S  = (sm * z).sum(dim=-1, keepdim=True)
            c1   = G_S / S
            c2   = (G_S * B_S - A_S * S) / (S * k)
            dx   = z * (sm * (g - c1) + c2)

        return dx.reshape(score.shape).to(grad.dtype), None, None


# ─── Public API ───────────────────────────────────────────────────────────────

def clamped_softmax(
    score: torch.Tensor,
    k: float,
    n_bisect: int = 30,
) -> torch.Tensor:
    """Budget-k sparse attention: output ∈ [0,1]^L with Σ_i out_i = k.

    Uses fused CUDA bisection kernel (forward) + analytical gradient (backward).
    Falls back to pure PyTorch when CUDA kernel is unavailable.

    score   : (…, L)  raw scores, spatial dimension last (already flattened)
    k       : float   budget — typically focus_size²
    n_bisect: int     bisection iterations (30 → < 4e-6 absolute error)
    """
    if _ready and score.is_cuda:
        return _ClampedSoftmax.apply(score, float(k), n_bisect)
    # PyTorch fallback: return only out (lam not needed outside backward)
    out, _ = _py_clamped_softmax(score.reshape(-1, score.shape[-1]), k)
    return out.reshape(score.shape)


# Trigger build at import time (non-blocking on import errors)
_build()
