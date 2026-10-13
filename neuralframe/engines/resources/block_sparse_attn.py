"""PyTorch stand-in for ``block_sparse_attn.block_sparse_attn_func`` (mit-han-lab/Block-Sparse-Attention).

Used by NeuralFrame's FlashVSR environment when the CUDA kernels cannot be built (no CUDA compiler). It
computes the same attention - each 128-token query block attends only to the key blocks its mask allows -
by gathering those key blocks, so memory stays proportional to the allowed blocks. Much slower than the
kernels; the FlashVSR authors note that dense fallbacks can change results slightly.
"""

import torch
import torch.nn.functional as F

BLOCK = 128


def block_sparse_attn_func(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type, streaming_info, base_blockmask,
                           max_seqlen_q, max_seqlen_k, p_dropout=0.0, deterministic=False, softmax_scale=None,
                           is_causal=False, exact_streaming=False, return_attn_probs=False):
    """q: (total_q, heads, d), k/v: (total_k, heads, d); one sequence per batch entry given by cu_seqlens;
    base_blockmask: (batch, mask_heads, q_blocks, k_blocks) bool; head_mask_type[h] = m > 0 means head h
    uses mask m - 1, 0 means dense. Returns (total_q, heads, d)."""
    out = torch.empty_like(q)
    heads = q.shape[1]
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    batches = cu_seqlens_q.numel() - 1
    for b in range(batches):
        q0, q1 = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        k0, k1 = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
        qb, kb = q[q0:q1], k[k0:k1]
        vb = v[k0:k1]
        nq, nk = q1 - q0, k1 - k0
        q_blocks, k_blocks = -(-nq // BLOCK), -(-nk // BLOCK)
        for h in range(heads):
            kind = int(head_mask_type[h]) if head_mask_type is not None else 0
            qh = qb[:, h]
            kh, vh = kb[:, h], vb[:, h]
            if kind <= 0 or base_blockmask is None:
                o = F.scaled_dot_product_attention(qh[None, None], kh[None, None], vh[None, None], scale=scale)
                out[q0:q1, h] = o[0, 0]
                continue
            mask = base_blockmask[min(b, base_blockmask.shape[0] - 1), kind - 1, :q_blocks, :k_blocks].bool()
            for i in range(q_blocks):
                cols = torch.nonzero(mask[i], as_tuple=False).flatten()
                rows = slice(i * BLOCK, min((i + 1) * BLOCK, nq))
                if cols.numel() == 0:
                    out[q0 + rows.start:q0 + rows.stop, h] = 0
                    continue
                idx = (cols[:, None] * BLOCK + torch.arange(BLOCK, device=cols.device)[None]).flatten()
                idx = idx[idx < nk]
                o = F.scaled_dot_product_attention(qh[rows][None, None], kh[idx][None, None], vh[idx][None, None],
                                                   scale=scale)
                out[q0 + rows.start:q0 + rows.stop, h] = o[0, 0]
    return out
