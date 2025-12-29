import torch
import triton
import triton.language as tl
import time
import math
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4, num_stages=2),
        #trtion.Config({'BLOCK_M' :64, 'BLOCK_N':128}, num_warps=)
    ],
    key=['N_CTX'],
)
@triton.jit
def flash_attn_v1_fwd_kernel(
    Q, K, V, O, M, L,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_mz, stride_mh, stride_mm,
    stride_lz, stride_lh, stride_lm,
    Z, H, N_CTX,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_HEAD_DIM: tl.constexpr,
    IF_CAUSAL_MASK: tl.constexpr,
    softmax_scale:tl.constexpr
):
    start_m = tl.program_id(0)
    batch_head_id = tl.program_id(1)

    batch_id = batch_head_id // H
    head_id = batch_head_id % H

    offs_m = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_HEAD_DIM)

    q_ptrs = (Q + batch_id * stride_qz + head_id * stride_qh +
              offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize online softmax variables
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) + float('-inf')
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_HEAD_DIM], dtype=tl.float32)

    #qk_scale = tl.rsqrt(BLOCK_SIZE_HEAD_DIM) # Ensure float type for rsqrt argument


    end_n = N_CTX if not IF_CAUSAL_MASK else (start_m + 1) * BLOCK_SIZE_M

    for start_n in range(0, end_n, BLOCK_SIZE_N):
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)

        # Corrected K pointers for transpose: (BLOCK_SIZE_HEAD_DIM, BLOCK_SIZE_N)
        # The strides correspond to the logical layout *after* transpose,
        # so we need to use stride_kn with offs_k and stride_kk with offs_n
        k_ptrs = (K + batch_id * stride_kz + head_id * stride_kh
                   + offs_n[:,None] * stride_kn + offs_k[None,:] * stride_kk )
        #k_ptrs = (K + batch_id * stride_kz + head_id * stride_kh
                   #+  offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn ) # Swapped strides and offsets

        # V pointers: (BLOCK_SIZE_N, BLOCK_SIZE_HEAD_DIM)
        v_ptrs = (V + batch_id * stride_vz + head_id * stride_vh +
                  offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

        # Mask for K and V loading
        k_mask = offs_n[:, None] < N_CTX
        #k_mask=  offs_n[None,:] <N_CTX # Mask applies to the dimension varying with 'n'
        # V mask
        v_mask = offs_n[:, None] < N_CTX

        # Load K tile (shape will be BLOCK_SIZE_HEAD_DIM x BLOCK_SIZE_N due to pointer layout)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        # Load V tile (shape will be BLOCK_SIZE_N x BLOCK_SIZE_HEAD_DIM)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)


        # Compute attention scores: Q (M, K) * K^T (K, N) -> (M, N)
        # tl.dot(q, k, trans_b=True) expects k to have shape (K, N) already
        # Since we loaded k with shape (BLOCK_SIZE_HEAD_DIM, BLOCK_SIZE_N),
        # we need to use trans_b=False or simply tl.dot(q, k)
        #qk = tl.dot(q, tl.trans(k), allow_tf32=False) # Use tl.dot(q, k) if k is loaded with shape (K, N)
                          # Or use tl.dot(q, k, trans_b=True) if k is loaded with shape (N, K)
                          # Given our pointer logic for k_ptrs, k is loaded as (K, N)
                          # So, tl.dot(q, k) is correct for (M, K) * (K, N) -> (M, N)
        #tl.debug_barrier()
        #tl.print(“qk tile =”, qk)
        qk = tl.dot(q, tl.trans(k))
        qk*=  softmax_scale

        # Apply causal mask if needed
        if IF_CAUSAL_MASK:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float('-inf'))

        # Online softmax computation
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p_ij = tl.exp(qk - m_i_new[:, None])
        scale = tl.exp(m_i - m_i_new)

        # Update accumulator
        acc = acc * scale[:, None]
        # p_ij (M, N), v (N, K) -> dot (M, K)
        acc += tl.dot(p_ij.to(v.dtype), v) # Removed trans_b=True, dot(A, B) expects B with columns matching A rows

        # Update normalizing factors
        l_i_current = tl.sum(p_ij, axis=1)
        l_i = l_i * scale + l_i_current
        m_i = m_i_new

    # Store outputs
    O_ptrs = (O + batch_id * stride_oz + head_id * stride_oh +
              offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    M_ptrs = (M + batch_id * stride_mz + head_id * stride_mh + offs_m * stride_mm)
    L_ptrs = (L + batch_id * stride_lz + head_id * stride_lh + offs_m * stride_lm)

    acc_o = acc / l_i[:, None]

    o_mask = offs_m[:, None] < N_CTX
    m_mask = offs_m < N_CTX

    tl.store(O_ptrs, acc_o, mask=o_mask)
    tl.store(M_ptrs, m_i, mask=m_mask)
    tl.store(L_ptrs, l_i, mask=m_mask)

#----BACKWARD KERNEL STARTS HERE------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 256}, num_warps=8),
    ],
    key=['N_CTX'],
)
@triton.jit
def flash_attn_v1_dO_O_bwd_kernel(
    dO, O, Delta,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_db,stride_dh,stride_dm,
    L,
    B, H, N_CTX,
    BLOCK_M:tl.constexpr,
    BLOCK_HEAD_DIM:tl.constexpr
):
    start_m = tl.program_id(0)
    batch_head_id = tl.program_id(1)

    batch_id = batch_head_id // H
    head_id = batch_head_id % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_HEAD_DIM)

    dO_ptrs = tl.load(dO + batch_id * stride_dob + head_id * stride_doh +
                      offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok,
                      mask=(offs_m[:, None] < N_CTX),
                      other=0.0).to(tl.float32)

    O_ptrs = tl.load(O + batch_id * stride_ob + head_id * stride_oh +
                     offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok,
                     mask=(offs_m[:, None] < N_CTX),
                     other=0.0).to(tl.float32)

    L_vals = tl.load(L + batch_id*stride_db + head_id*stride_dh + offs_m*stride_dm, mask=offs_m < N_CTX, other=1e-12)

    dO_ptrs_normal = dO_ptrs / L_vals[:, None]

    beta_vals = tl.sum((dO_ptrs_normal * O_ptrs), axis=-1)

    tl.store(dO + batch_id * stride_dob + head_id * stride_doh +
             offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok, dO_ptrs_normal)
    tl.store(Delta + batch_id * stride_db + head_id * stride_dh + offs_m * stride_dm , beta_vals)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=3)
    ],
    key=['N_CTX'],
)
@triton.jit
def flash_attn_bwd_dq_kernel(
    Q, K, V, DO, DQ, L, M, D, softmax_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    Z, H, N_CTX, HEAD_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,BLOCK_HEAD_DIM:tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_h = off_hz % H
    off_z = off_hz // H

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEAD_DIM)

    # Load Q, DO for this block
    q_ptrs = (Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    do_ptrs = (DO + off_z * stride_doz + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dok)



    q = tl.load(q_ptrs, mask=(offs_m[:, None]<N_CTX) & (offs_d[None ,:]< HEAD_DIM), other=0.0)
    do = tl.load(do_ptrs, mask=(offs_m[:, None]<N_CTX) & (offs_d[None ,:]< HEAD_DIM), other=0.0)

    # Load statistics
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    d_ptrs = D + off_hz * N_CTX + offs_m

    l_i = tl.load(l_ptrs, mask=offs_m < N_CTX, other=1.0)
    m_i = tl.load(m_ptrs, mask=offs_m < N_CTX, other=float('-inf'))
    d_i = tl.load(d_ptrs, mask=offs_m < N_CTX, other=0.0)

    # Initialize dQ accumulator
    dq = tl.zeros([BLOCK_M,BLOCK_HEAD_DIM], dtype=tl.float32)

    # Loop over blocks of K, V
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_curr = start_n + offs_n

        # Load K, V
        k_ptrs = (K + off_z * stride_kz + off_h * stride_kh + offs_n_curr[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v_ptrs = (V + off_z * stride_vz + off_h * stride_vh + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vk)


        k = tl.load(k_ptrs, mask=(offs_n_curr[:, None]<N_CTX) & (offs_d[None, :] < HEAD_DIM) , other=0.0)
        v = tl.load(v_ptrs,  mask=(offs_n_curr[:, None]<N_CTX) & (offs_d[None, :] < HEAD_DIM) , other=0.0)

        # this is done so the accumulaton stays in 32bit
        #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k))
        qk*=  softmax_scale

        # Apply causal mask
        #causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
        #qk = tl.where(causal_mask, qk, float('-inf'))

        # Compute attention weights
        p = tl.exp(qk - m_i[:, None])


        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - d_i[:, None])).to(q.dtype)
        #ds = ds / l_i[:, None]

        ds = ds  * softmax_scale
        # Accumulate dQ
        dq += tl.dot(ds.to(k.dtype), k)

    # Store dQ
    dq_ptrs = (DQ + off_z * stride_dqz + off_h * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqk)
    tl.store(dq_ptrs, dq, mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=3)
    ],
    key=['N_CTX'],
)
@triton.jit
def flash_attn_bwd_dk_dv_kernel(
    Q, K, V, DO, DK, DV, L, M, D, softmax_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    Z, H, N_CTX, HEAD_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,BLOCK_HEAD_DIM:tl.constexpr
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_h = off_hz % H
    off_z = off_hz // H

    # Initialize offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEAD_DIM)


    # Load K, V for this block
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk


    mask_kv_load = (offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM)

    k = tl.load(k_ptrs, mask=mask_kv_load, other=0.0)
    v = tl.load(v_ptrs, mask=mask_kv_load, other=0.0)

    # Initialize dK, dV accumulators
    dk = tl.zeros([BLOCK_N,BLOCK_HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N,BLOCK_HEAD_DIM], dtype=tl.float32)

    # Loop over blocks of Q, DO
    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m_curr = start_m + offs_m

        # Load Q, DO
        q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = DO + off_z * stride_doz + off_h * stride_doh + offs_m_curr[:, None] * stride_dom + offs_d[None, :] * stride_dok


        q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0.0)
        do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX, other=0.0)

        # Load statistics
        l_ptrs = L + off_hz * N_CTX + offs_m_curr
        m_ptrs = M + off_hz * N_CTX + offs_m_curr
        d_ptrs = D + off_hz * N_CTX + offs_m_curr



        l_i = tl.load(l_ptrs, mask=offs_m_curr < N_CTX, other=0.0)
        m_i = tl.load(m_ptrs, mask=offs_m_curr < N_CTX, other=float('-inf'))
        d_i = tl.load(d_ptrs, mask=offs_m_curr < N_CTX, other=0.0)


        # Compute attention scores
        #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k))
        qk *= softmax_scale

        # Apply causal mask
        #causal_mask = offs_m_curr[:, None] >= offs_n[None, :]
        #qk = tl.where(causal_mask, qk, float('-inf'))

        # Compute attention weights
        p = tl.exp(qk - m_i[:, None])
        #p = p * l_i[:, None]

        # Accumulate dV
        dv += tl.dot(tl.trans(p).to(q.dtype), do)

        # Compute dS and accumulate dK
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - d_i[:, None])).to(q.dtype)
        ds *=  softmax_scale
        dk += tl.dot(tl.trans(ds).to(q.dtype), q)

    # Store dK, dV
    dk_ptrs = DK + off_z * stride_dkz + off_h * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    dv_ptrs = DV + off_z * stride_dvz + off_h * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk

    tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))
    tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM))


class Myattention1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False):

        batch_size, num_heads, seq_len, head_dim = q.shape
        softmax_scale = 1.0 / math.sqrt(head_dim)

        # 2. Create output tensors
        output = torch.empty_like(q)
        # m and l are for the log-sum-exp for backward pass stability
        m = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)
        l = torch.empty((batch_size, num_heads, seq_len), device=q.device, dtype=torch.float32)


        grid = lambda meta:(triton.cdiv(seq_len, meta['BLOCK_SIZE_M']) ,batch_size * num_heads)

        #Launch Forward Kernel

        flash_attn_v1_fwd_kernel[grid](
            q, k, v, output, m, l,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            m.stride(0), m.stride(1), m.stride(2),
            l.stride(0), l.stride(1), l.stride(2),
            batch_size, num_heads, seq_len,
            #BLOCK_SIZE_M=block_m,
            #BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_HEAD_DIM=head_dim,
            IF_CAUSAL_MASK=causal,
            softmax_scale=softmax_scale
        )

        ctx.save_for_backward(output, m, l, q, k, v)
        ctx.sm_scale = softmax_scale
        ctx.head_dim = head_dim
        return output

    @staticmethod
    def backward(ctx, do):
        # 1. Retrieve saved tensors and constants
        O, m, l, Q, K, V = ctx.saved_tensors

        softmax_scale = ctx.sm_scale
        #BLOCK_M = ctx.BLOCK_M
        #BLOCK_N = ctx.BLOCK_N
        head_dim = ctx.head_dim
        # causal = ctx.causal # If needed in kernels

        batch, heads, seq_len, _ = Q.shape

        DO = do.contiguous()
        DQ = torch.empty_like(Q)
        DK = torch.empty_like(K)
        DV = torch.empty_like(V)

        Delta = torch.empty_like(l)

        #Preprocess - compute Delta (D)
        grid_preprocess = lambda meta :(triton.cdiv(seq_len, meta['BLOCK_M']), batch * heads)

        flash_attn_v1_dO_O_bwd_kernel[grid_preprocess](
            DO,
            O,
            Delta,
            DO.stride(0), DO.stride(1), DO.stride(2), DO.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            l, # Often passed to help scaling or masking
            batch, heads, seq_len,
            #BLOCK_M=BLOCK_M,
            BLOCK_HEAD_DIM=head_dim
        )


        # Grid often matches the forward grid structure
        grid_dq = lambda meta :(triton.cdiv(seq_len, meta['BLOCK_M']),
                                batch * heads)

        flash_attn_bwd_dq_kernel[grid_dq](
            Q, K, V, DO, DQ, l, m, Delta, softmax_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            DO.stride(0), DO.stride(1), DO.stride(2), DO.stride(3),
            DQ.stride(0), DQ.stride(1), DQ.stride(2), DQ.stride(3),
            batch, heads, seq_len, head_dim,
            #BLOCK_M=BLOCK_M,
            #BLOCK_N=BLOCK_N,
            BLOCK_HEAD_DIM=head_dim
        )

        #grid_dkv = (triton.cdiv(seq_len, BLOCK_N), batch * heads)
        grid_dkv = lambda meta :(triton.cdiv(seq_len, meta['BLOCK_N']),
                                batch * heads)
        flash_attn_bwd_dk_dv_kernel[grid_dkv](
            Q, K, V, DO, DK, DV, l, m, Delta, softmax_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            DO.stride(0), DO.stride(1), DO.stride(2), DO.stride(3),
            DK.stride(0), DK.stride(1), DK.stride(2), DK.stride(3),
            DV.stride(0), DV.stride(1), DV.stride(2), DV.stride(3),
            batch, heads, seq_len, head_dim,
            #BLOCK_M=BLOCK_M,
            #BLOCK_N=BLOCK_N,
            BLOCK_HEAD_DIM=head_dim
        )
        return DQ, DK, DV, None, None, None



