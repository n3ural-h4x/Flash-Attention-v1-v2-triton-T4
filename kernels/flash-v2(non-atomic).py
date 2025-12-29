import triton
import triton.language as tl
import torch
import math

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=3),
    ],
    key=['N_CTX'],
)
# Some basic thing stride are basically how many values to shift to get to the next dim since in CUDA the matrix assumes as row-major format
# That the matrix is laid out in the memory in row-column major so for example for a tensor of shape(3,4) its stride(0) would be 4 and stride(1) would be 1
@triton.jit
def flash_attn_v2_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lse_z, stride_lse_h, stride_lse_m,
    Z, H, N_CTX, D_HEAD,        # z here represents the batch_id and N_CTX is nothing but the seq_len
    SOFTMAX_SCALE,
    BLOCK_SIZE_M: tl.constexpr, #tl.constexpr means that the triton compiler will get the values at compile time
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    IF_CAUSAL_MASK: tl.constexpr,
):

    # Loading the indcies of batch and head and axis
    start_m = tl.program_id(axis=0)
    batch_head_id = tl.program_id(axis=1) # Batch and Head index combined since b*h can easily be covered by unique p_id

    # Loadin the indcies of batch and head
    batch_id = batch_head_id // H
    head_id = batch_head_id % H

    # Row offsets for the Q block (BLOCK_SIZE_M rows)
    offs_m = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Head dimension offsets (BLOCK_SIZE_K columns, which is D_HEAD)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Q pointers: Shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    q_ptrs = (Q + batch_id * stride_qz + head_id * stride_qh +
              offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # K/V base pointers for the current batch and head
    k_base_ptr = K + batch_id * stride_kz + head_id * stride_kh
    v_base_ptr = V + batch_id * stride_vz + head_id * stride_vh


    # Accumulator for the output O = Softmax(QK^T)V
    # Needs to be float32 for precision
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # Max value encountered so far per row: m_i
    m_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) - float('inf') # Initialize max to -inf
    # Sum of exp(x - max) encountered so far per row: l_i
    l_i = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) # Initialize sum to 0

    # Scale factor for QK^T (usually 1 / sqrt(D_HEAD)) This not working so better to
    #qk_scale = tl.rsqrt(D_HEAD.to(tl.float32))

    # --- Load Q Tile ---
    # Load Q for the current block row (offs_m)
    # Boundary check for Q rows (sequence length N_CTX)
    q_mask = offs_m[:, None] < N_CTX

    # Note the q_ptrs will be stored in sram for the full loop duration or for one p_id
    # Note the we could also have added offs_k[None, :]<D_HEAD but didnt since we want it to be fixed length
    q = tl.load(q_ptrs, mask=q_mask, other=0.0) # Shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)



    # We need all K/V blocks
    end_n = N_CTX
    if IF_CAUSAL_MASK:
      # K/V blocks should end where Q block ends
      # For token i (row in Q), we only attend to tokens j <= i (columns in K/V)
      end_n = (start_m + 1) * BLOCK_SIZE_M

    # --- Loop over K/V Blocks (Columns of QK^T matrix) ---
    for start_n in range(0, end_n, BLOCK_SIZE_N):
        # Loadin the K/V tiles since we are looping them over for one p_id
        # Column offsets for the current K/V block (BLOCK_SIZE_N columns)
        offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)

        # K pointers: Shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        #Since S=QK^T so we already loaded the K in the abovesaid dimension
        k_ptrs = (k_base_ptr +
                  offs_k[:, None] * stride_kk + offs_n[None ,:] * stride_kn )
        # V pointers: Shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
        v_ptrs = (v_base_ptr +
                  offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)

        # Boundary check masks for K and V tiles
        # K mask depends on N_CTX for columns (offs_n)
        k_mask = (offs_n[None ,:] < N_CTX)
        # V mask depends on N_CTX for rows (offs_n)
        v_mask = (offs_n[:, None] < N_CTX)

        # Load K tile (transposed layout for dot product)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0) # Shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        # Load V tile
        v = tl.load(v_ptrs, mask=v_mask, other=0.0) # Shape: (BLOCK_SIZE_N, BLOCK_SIZE_K)

        # --- Compute QK^T Score Block ---
        # q shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        # k shape: (BLOCK_SIZE_K, BLOCK_SIZE_N)
        #qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k) # Shape: (BLOCK_SIZE_M, BLOCK_SIZE_N)


        # --- Apply Causal Mask (if enabled) ---
        if IF_CAUSAL_MASK:
            # Create mask where q_row_index >= k_col_index
            # offs_m are row indices, offs_n are column indices
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            # Apply mask: Set scores for future tokens to negative infinity
            qk = tl.where(causal_mask, qk, -float('inf'))

        qk_scale= 1.44269504089  * SOFTMAX_SCALE  # 1/ln(2)=1.44269...  used for exp2 as we have written since on triton 2^x is much faster
        qk *= qk_scale # Apply scaling factor

        # --- Online Softmax Calculation ---
        # Since the whole Softmax matrix cant be stored as we havent seen the whole columns for K
        # 1. Find the new maximum across the block's scores and the old maximum
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1)) # Shape: (BLOCK_SIZE_M,)

        # 2. Calculate probabilities P_ij = exp(qk_ij - m_i_new)
        # IMP: Subtracting the new max prevents overflow
        p_ij = tl.exp2(qk - m_i_new[:, None]) # Shape: (BLOCK_SIZE_M, BLOCK_SIZE_N)

        # 3. Calculate scaling factor for previous accumulator and l_i
        # scale = exp(m_i_old - m_i_new)
        scale = tl.exp2(m_i - m_i_new) # Shape: (BLOCK_SIZE_M,)

        # 4. Rescale previous accumulator: acc = acc * scale
        acc = acc * scale[:, None] # Shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)

        # 5. Update accumulator: acc = acc + P_ij @ V_j
        # p_ij shape: (BLOCK_SIZE_M, BLOCK_SIZE_N)
        # v shape: (BLOCK_SIZE_N, BLOCK_SIZE_K)
        # Ensure matching types for dot product
        acc += tl.dot(p_ij.to(v.dtype), v)

        # 6. Update the sum denominator: l_i = l_i * scale + sum(P_ij, axis=1)
        # Calculate block sum: l_i_current = sum(P_ij, axis=1)
        l_i_current = tl.sum(p_ij, axis=1) # Shape: (BLOCK_SIZE_M,)
        # Rescale previous l_i and add current block's contribution
        l_i = l_i * scale + l_i_current # Shape: (BLOCK_SIZE_M,)
        # Why axis 1 because we want it be stored per column
        # 7. Update running max for next iteration: m_i = m_i_new
        m_i = m_i_new
       #Repeat till we havent seen the entire col matrix



    # LSE = m_i + log2(l_i)
    # We dont store the whole m_i and l_i as we will do in the v1 version we just use LSE:log_sum_exponential
    log2_l_i = tl.log2(l_i) # why log2 since its faster
    lse_final = m_i + log2_l_i # Shape: (BLOCK_SIZE_M,)


    #Divide it by the sum we encountered
    acc_o = acc * 1/l_i[:, None] # Shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)

    # 3. Store Output O
    # Output pointers: Shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
    o_ptrs = (O + batch_id * stride_oz + head_id * stride_oh +
              offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok)
    # Boundary check mask for O (same as Q mask for rows, implicit for cols)
    o_mask = offs_m[:, None] < N_CTX
    # Store the result (convert to output dtype if necessary)
    tl.store(o_ptrs, acc_o.to(Q.dtype.element_ty), mask=o_mask)

    # 4. Store LogSumExp (LSE)
    # LSE pointers: Shape (BLOCK_SIZE_M,)
    lse_ptrs = (LSE + batch_id * stride_lse_z + head_id * stride_lse_h +
                offs_m * stride_lse_m)
    # Boundary check mask for LSE rows
    lse_mask = offs_m < N_CTX
    # Store LSE
    tl.store(lse_ptrs, lse_final, mask=lse_mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256}, num_warps=8),
    ],
    key=['N_CTX'],
)
@triton.jit
def _backward_single_kernel(
    Do_ptr, O_ptr, Delta_ptr, H,

    N_CTX, HEAD_DIM: tl.constexpr,
    stride_dob, stride_doh, stride_don, stride_do_head,
    stride_ob, stride_oh, stride_on, stride_o_head,
    BLOCK_SIZE_M: tl.constexpr
):

    start_m = tl.program_id(0)
    batch_row_head = tl.program_id(1)
    offs_m = start_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    batch_row = batch_row_head // H
    row_head  = batch_row_head % H

    O_ptrs = O_ptr + stride_ob*batch_row + stride_oh*row_head
    Do_ptrs = Do_ptr + stride_dob*batch_row + stride_doh*row_head

    p_o = tl.make_block_ptr(
        base=O_ptrs,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_on, stride_o_head),
        offsets=(start_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0)
    )

    p_do = tl.make_block_ptr(
        base=Do_ptrs,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_don, stride_do_head),
        offsets=(start_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, HEAD_DIM),
        order=(1, 0)
    )


    b_o = tl.load(p_o, boundary_check=(0, 1), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")

    delta = tl.sum(b_o.to(tl.float32) * b_do.to(tl.float32), axis=1)

    delta_offs = Delta_ptr + batch_row_head * N_CTX + offs_m
    tl.store(delta_offs, delta, mask= offs_m< N_CTX)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=3)
    ],
    key=['N_CTX'],
)
@triton.jit
def _backward_for_dq(
    q_ptr, k_ptr, v_ptr, do_ptr, H,
    dq_ptr,
    lse_ptr, delta_ptr,
    # Strides (Input)
    stride_qb, stride_qh, stride_qn, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dob, stride_doh, stride_don, stride_dok,
    # Strides (Output)
    stride_dqb, stride_dqh, stride_dqn, stride_dqk,
    #stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    #stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    # Constants
    N_CTX, HEAD_DIM: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    LOG2_E: tl.constexpr
):

    start_m = tl.program_id(0)
    batch_row_head = tl.program_id(1)
    offs_m = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_d = tl.arange(0, HEAD_DIM)

    batch_row = batch_row_head // H
    row_head  = batch_row_head % H

    q_ptrs = (q_ptr + batch_row * stride_qb + row_head * stride_qh)
    k_ptrs = (k_ptr + batch_row * stride_kb + row_head * stride_kh)
    v_ptrs = (v_ptr + batch_row * stride_vb + row_head * stride_vh)
    do_ptrs = (do_ptr + batch_row * stride_dob + row_head * stride_doh)
    dq_ptrs = (dq_ptr + batch_row * stride_dqb + row_head * stride_dqh )
    # pointers which have been computed previously
    l_ptrs = lse_ptr + batch_row_head * N_CTX+ offs_m
    d_ptrs = delta_ptr + batch_row_head * N_CTX+ offs_m
    #contiguous
    dq = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)


    Q_ptr = tl.make_block_ptr(
            base=q_ptrs,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qn, stride_qk),
            offsets=(start_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, HEAD_DIM),
            order=(1, 0)
        )
    dO_ptr = tl.make_block_ptr(
            base=do_ptrs,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_don, stride_dok),
            offsets=(start_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, HEAD_DIM),
            order=(1, 0)
        )
    K_ptr = tl.make_block_ptr(
             base = k_ptrs,
             shape = (N_CTX, HEAD_DIM),
             strides = (stride_kn, stride_kk),
             offsets = (0, 0),
             block_shape = (BLOCK_SIZE_N, HEAD_DIM),
             order = (1, 0)
    )
    V_ptr = tl.make_block_ptr(
        base = v_ptrs,
        shape = (N_CTX, HEAD_DIM),
        strides = (stride_vn, stride_vk),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_N, HEAD_DIM),
        order = (1, 0)
    )
    q = tl.load(Q_ptr, boundary_check=(0, 1), padding_option="zero")
    dO = tl.load(dO_ptr, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(l_ptrs, mask = offs_m  < N_CTX, other = 0.0)
    D_i = tl.load(d_ptrs, mask = offs_m < N_CTX, other = 0.0)
    #tl.load(lse, mask = (offs_m[:, None] < N_ctx), others = 1e-6)

    for start_n in range(0, N_CTX, BLOCK_SIZE_N):

       #offs_curr_n = start_n + offs_n

       #l_ptrs = lse_ptr + batch_row_head * N_CTX + offs_curr_m
       #d_ptrs = delta_ptr + batch_row_head * N_CTX + offs_curr_m

       #q = tl.load(Q_ptr, boundary_check=(0, 1),padding_option="zero" )
       #dO = tl.load(dO_ptr, boundary_check=(0, 1), padding_option="zero")
       k = tl.load(K_ptr, boundary_check=(0, 1), padding_option="zero")
       v = tl.load(V_ptr, boundary_check=(0, 1), padding_option="zero")
       #lse = tl.load(l_ptrs, mask = offs_curr_m  < N_CTX, other = 0.0)
       #D_i = tl.load(d_ptrs, mask = offs_curr_m < N_CTX, other = 0.0)

       #Note you will generate if seq_len isnt power of just mask the k column and q column then -inf
       qk = tl.dot(q, tl.trans(k))
       #mask = offs_curr_m < N_ctx
       #qk = tl.where(mask[None, :], qk, float("-inf"))

       p = tl.exp2((qk * sm_scale * LOG2_E ) - lse[:, None]) # since in the lse is stored in log base 2 we will muplity it with scale factor ln(a) = ln(2)*log2(a)
       #Since tis is a dq kernel so not needed
       #dv = tl.dot(tl.trans(p).to(tl.float16), dO)
       dp = tl.dot(dO, tl.trans(v))
       ds = (p * (dp - D_i[:, None]) * sm_scale ).to(q.dtype)
       #ds = (p * dp * sm_scale).to(q.dtype)
       #ds *= sm_scale
       #tl.debug_barrier()
       dq += tl.dot(ds, k)

       #dq_curr = offs_curr_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqk


       #tl.debug_barrier()
       #tl.atomic_add(dq_ptrs + dq_curr, dq, mask=(offs_curr_m[:, None] < N_CTX))
       #tl.debug_barrier()
       #dk += tl.dot(tl.trans(ds), q)

       K_ptr = tl.advance(K_ptr, offsets=(BLOCK_SIZE_N, 0))
       V_ptr = tl.advance(V_ptr, offsets=(BLOCK_SIZE_N, 0))

    #dk_ptrs = dk_ptr + batch_row * stride_dkb + row_head * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    #dv_ptrs = dv_ptr + batch_row * stride_dvb + row_head * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk
    dq_ptrs = dq_ptr + batch_row * stride_dqb + row_head * stride_dqh + offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqk
    #tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < N_CTX))
    #tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < N_CTX))
    tl.store(dq_ptrs, dq, mask=(offs_m[:, None] < N_CTX))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=3)
    ],
    key=['N_CTX'],
)
@triton.jit
def _backward_for_dk_dv(
    q_ptr, k_ptr, v_ptr, do_ptr, H,
     dk_ptr, dv_ptr,
    lse_ptr, delta_ptr,
    # Strides (Input)
    stride_qb, stride_qh, stride_qn, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_dob, stride_doh, stride_don, stride_dok,
    # Strides (Output)
    #stride_dqb, stride_dqh, stride_dqn, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    # Constants
    N_CTX, HEAD_DIM: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    LOG2_E: tl.constexpr
):

    start_n = tl.program_id(0)
    batch_row_head = tl.program_id(1)
    offs_n = start_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, HEAD_DIM)

    batch_row = batch_row_head // H
    row_head  = batch_row_head % H

    q_ptrs = (q_ptr + batch_row * stride_qb + row_head * stride_qh)
    k_ptrs = (k_ptr + batch_row * stride_kb + row_head * stride_kh)
    v_ptrs = (v_ptr + batch_row * stride_vb + row_head * stride_vh)
    do_ptrs = (do_ptr + batch_row * stride_dob + row_head * stride_doh)
    #dq_ptrs = (dq_ptr + batch_row * stride_dqb + row_head * stride_dqh )

    #l_ptrs = lse_ptr + batch_row_head * N_ctx
    #d_ptrs = delta_ptr + batch_row_head * N_ctx
    #contiguous

    #dq_ptr = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_HEAD], dtype=tl.float32)
    dv  = tl.zeros([BLOCK_SIZE_N, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_SIZE_N, HEAD_DIM], dtype=tl.float32)

    Q_ptr = tl.make_block_ptr(
            base=q_ptrs,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qn, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, HEAD_DIM),
            order=(1, 0)
        )
    dO_ptr = tl.make_block_ptr(
            base=do_ptrs,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_don, stride_dok),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_M, HEAD_DIM),
            order=(1, 0)
        )
    K_ptr = tl.make_block_ptr(
             base = k_ptrs,
             shape = (N_CTX, HEAD_DIM),
             strides = (stride_kn, stride_kk),
             offsets = (start_n * BLOCK_SIZE_N, 0),
             block_shape = (BLOCK_SIZE_N, HEAD_DIM),
             order = (1, 0)
    )
    V_ptr = tl.make_block_ptr(
        base = v_ptrs,
        shape = (N_CTX, HEAD_DIM),
        strides = (stride_vn, stride_vk),
        offsets = (start_n * BLOCK_SIZE_N, 0),
        block_shape = (BLOCK_SIZE_N, HEAD_DIM),
        order = (1, 0)
    )
    k = tl.load(K_ptr, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(V_ptr, boundary_check=(0, 1), padding_option="zero")
    #tl.load(lse, mask = (offs_m[:, None] < N_ctx), others = 1e-6)

    for start_m in range(0, N_CTX, BLOCK_SIZE_M):

       offs_curr_m = start_m + offs_m

       l_ptrs = lse_ptr + batch_row_head * N_CTX + offs_curr_m
       d_ptrs = delta_ptr + batch_row_head * N_CTX + offs_curr_m

       q = tl.load(Q_ptr, boundary_check=(0, 1),padding_option="zero" )
       dO = tl.load(dO_ptr, boundary_check=(0, 1), padding_option="zero")
       lse = tl.load(l_ptrs, mask = offs_curr_m  < N_CTX, other = 0.0)
       D_i = tl.load(d_ptrs, mask = offs_curr_m < N_CTX, other = 0.0)

        #Note you will generate if seq_len isnt power of just mask the k column and q column then -inf
       qk = tl.dot(q, tl.trans(k))
       #mask = offs_curr_m < N_ctx
       #qk = tl.where(mask[None, :], qk, float("-inf"))

       p = tl.exp2((qk * sm_scale * LOG2_E ) - lse[:, None]) # since in the lse is stored in log base 2 we will muplity it with scale factor ln(a) = ln(2)*log2(a)

       dv += tl.dot(tl.trans(p).to(tl.float16), dO)
       dp = tl.dot(dO, tl.trans(v))
       ds = (p * (dp - D_i[:, None]) * sm_scale ).to(q.dtype)
       #ds = (p * dp * sm_scale).to(q.dtype)
       #ds *= sm_scale
       #tl.debug_barrier()
       dq = tl.dot(ds, k)

       #dq_curr = offs_curr_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqk


       #tl.debug_barrier()
       #tl.atomic_add(dq_ptrs + dq_curr, dq, mask=(offs_curr_m[:, None] < N_CTX))
       #tl.debug_barrier()
       dk += tl.dot(tl.trans(ds), q)

       Q_ptr = tl.advance(Q_ptr, offsets=(BLOCK_SIZE_M, 0))
       dO_ptr = tl.advance(dO_ptr, offsets=(BLOCK_SIZE_M, 0))

    dk_ptrs = dk_ptr + batch_row * stride_dkb + row_head * stride_dkh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkk
    dv_ptrs = dv_ptr + batch_row * stride_dvb + row_head * stride_dvh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvk

    tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < N_CTX))
    tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < N_CTX))

class MyattentionV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False):

        BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

        #Create output tensor and lse
        o = torch.empty_like(q)
        lse = torch.empty((BATCH, N_HEADS, N_CTX), device=q.device, dtype=torch.float32)
        grid = lambda meta: (
            triton.cdiv(N_CTX, meta['BLOCK_SIZE_M']),
            BATCH * N_HEADS
        )

        #Define the kernel
        flash_attn_v2_fwd_kernel[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            BATCH, N_HEADS, N_CTX, HEAD_DIM,
            SOFTMAX_SCALE=sm_scale,
            BLOCK_SIZE_K=HEAD_DIM,
            IF_CAUSAL_MASK=causal,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
        sm_scale = ctx.sm_scale

        # Ensure contiguous
        do = do.contiguous()

        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(lse)

        # Compute the do*o
        grid_delta = lambda meta: (
            triton.cdiv(N_CTX, meta['BLOCK_SIZE_M']),
            BATCH * N_HEADS
        )

        _backward_single_kernel[grid_delta](
            do, o, delta,
            N_HEADS, N_CTX, HEAD_DIM,
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3)
        )

        grid_dq = lambda meta: (
            triton.cdiv(N_CTX, meta['BLOCK_SIZE_M']),
            BATCH * N_HEADS
        )

        _backward_for_dq[grid_dq](
            q_ptr=q, k_ptr=k, v_ptr=v, do_ptr=do, H=N_HEADS,
            dq_ptr=dq,
            lse_ptr=lse, delta_ptr=delta,
            stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qn=q.stride(2), stride_qk=q.stride(3),
            stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kn=k.stride(2), stride_kk=k.stride(3),
            stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vn=v.stride(2), stride_vk=v.stride(3),
            stride_dob=do.stride(0), stride_doh=do.stride(1), stride_don=do.stride(2), stride_dok=do.stride(3),
            stride_dqb=dq.stride(0), stride_dqh=dq.stride(1), stride_dqn=dq.stride(2), stride_dqk=dq.stride(3),
            N_CTX=N_CTX, HEAD_DIM=HEAD_DIM, sm_scale=sm_scale,
            LOG2_E=1.44269504089
        )

        grid_dk_dv = lambda meta: (
            triton.cdiv(N_CTX, meta['BLOCK_SIZE_N']),
            BATCH * N_HEADS
        )

        _backward_for_dk_dv[grid_dk_dv](
            q_ptr=q, k_ptr=k, v_ptr=v, do_ptr=do, H=N_HEADS,
            dk_ptr=dk, dv_ptr=dv,
            lse_ptr=lse, delta_ptr=delta,
            stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qn=q.stride(2), stride_qk=q.stride(3),
            stride_kb=k.stride(0), stride_kh=k.stride(1), stride_kn=k.stride(2), stride_kk=k.stride(3),
            stride_vb=v.stride(0), stride_vh=v.stride(1), stride_vn=v.stride(2), stride_vk=v.stride(3),
            stride_dob=do.stride(0), stride_doh=do.stride(1), stride_don=do.stride(2), stride_dok=do.stride(3),
            stride_dkb=dk.stride(0), stride_dkh=dk.stride(1), stride_dkn=dk.stride(2), stride_dkk=dk.stride(3),
            stride_dvb=dv.stride(0), stride_dvh=dv.stride(1), stride_dvn=dv.stride(2), stride_dvk=dv.stride(3),
            N_CTX=N_CTX, HEAD_DIM=HEAD_DIM, sm_scale=sm_scale,
            LOG2_E=1.44269504089
        )


        return dq, dk, dv, None

