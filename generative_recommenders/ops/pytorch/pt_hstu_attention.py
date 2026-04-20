# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


def _get_chunk_size(default_chunk_size: int, device: torch.device, max_seq_len: int) -> int:
    chunk_size = default_chunk_size
    if chunk_size <= 0 and device.type == "npu":
        chunk_size = 64
    if chunk_size <= 0:
        return chunk_size

    # If the runtime exposes memory stats, use them to bias toward smaller chunks
    # when the device is nearly full. This keeps the fallback path conservative.
    if device.type == "npu" and hasattr(torch, "npu"):
        try:
            free_mem, total_mem = torch.npu.mem_get_info(device)  # pyre-ignore [16]
            if total_mem > 0:
                free_ratio = float(free_mem) / float(total_mem)
                if free_ratio < 0.10:
                    chunk_size = min(chunk_size, 16)
                elif free_ratio < 0.20:
                    chunk_size = min(chunk_size, 32)
                elif free_ratio < 0.30:
                    chunk_size = min(chunk_size, 48)
        except Exception:
            pass

    return max(1, min(chunk_size, max_seq_len))


@torch.fx.wrap
def _get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    row_start: int = 0,
    row_end: Optional[int] = None,
    col_start: int = 0,
    col_end: Optional[int] = None,
) -> torch.Tensor:
    if row_end is None:
        row_end = N
    if col_end is None:
        col_end = N

    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1)
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - contextual_seq_len + 1

    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1)
        ids = torch.clamp(ids.expand(max_ids.size(0), -1), max=max_ids)
    else:
        ids = ids

    row_ids = ids[:, row_start:row_end].unsqueeze(-1)
    col_ids = ids[:, col_start:col_end].unsqueeze(1)
    row_col_dist = row_ids - col_ids
    valid_attn_mask = row_ids == col_ids
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            max_ids_3d = max_ids.unsqueeze(1)
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids_3d - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask, row_col_dist <= max_attn_len
            )
    if contextual_seq_len > 0:
        max_ids_3d = max_ids.unsqueeze(1)
        valid_attn_mask = torch.logical_or(
            valid_attn_mask,
            torch.logical_and(row_ids == 0, col_ids < max_ids_3d),
        )
    return valid_attn_mask


def _pad_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, H, D = q.shape
    V = v.shape[2]
    padded_q = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=q.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(L, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V)
        .transpose(1, 2)
    )  # [B, H, N, D]
    return padded_q, padded_k, padded_v


def _pad_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    L, H, D = k.shape
    V = v.shape[2]
    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )
    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(L, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V)
        .transpose(1, 2)
    )
    return padded_k, padded_v


def _pad_q_chunk(
    q: torch.Tensor,
    seq_offsets: torch.Tensor,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    B = seq_offsets.numel() - 1
    chunk_size = row_end - row_start
    _, H, D = q.shape
    padded_q_chunk = torch.zeros(
        (B, H, chunk_size, D),
        device=q.device,
        dtype=q.dtype,
    )

    for batch_idx in range(B):
        seq_start = int(seq_offsets[batch_idx].item())
        seq_end = int(seq_offsets[batch_idx + 1].item())
        seq_len = seq_end - seq_start
        if row_start >= seq_len:
            continue
        local_end = min(row_end, seq_len)
        valid_rows = local_end - row_start
        q_slice = q[seq_start + row_start : seq_start + local_end]
        padded_q_chunk[batch_idx, :, :valid_rows, :] = q_slice.transpose(0, 1)

    return padded_q_chunk


def _get_chunk_column_range(
    start: int,
    end: int,
    max_seq_len: int,
    max_attn_len: int,
    contextual_seq_len: int,
) -> Tuple[int, int]:
    if max_attn_len <= 0:
        return 0, max_seq_len
    col_start = max(0, start - max_attn_len)
    if contextual_seq_len > 0:
        col_start = 0
    col_end = min(max_seq_len, end)
    return col_start, col_end


@torch.fx.wrap
def pytorch_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    L, H, _ = q.shape
    V = v.shape[2]
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]

    prepared_attn_scale = attn_scale
    if prepared_attn_scale is not None and prepared_attn_scale.ndim > 0:
        prepared_attn_scale = (
            torch.ops.fbgemm.jagged_to_padded_dense(
                values=prepared_attn_scale.unsqueeze(-1),
                offsets=[seq_offsets],
                max_lengths=[max_seq_len],
                padding_value=0.0,
            )
            .unsqueeze(1)
            .to(q.dtype)
        )

    # Chunked attention avoids materializing [B, H, N, N] in one shot.
    chunk_size = _get_chunk_size(
        default_chunk_size=int(os.environ.get("HSTU_ATTN_CHUNK_SIZE", "0")),
        device=q.device,
        max_seq_len=max_seq_len,
    )

    if chunk_size > 0 and chunk_size < max_seq_len:
        k, v = _pad_kv(k, v, seq_offsets, max_seq_len)
        chunks = []
        for start in range(0, max_seq_len, chunk_size):
            end = min(start + chunk_size, max_seq_len)
            col_start, col_end = _get_chunk_column_range(
                start=start,
                end=end,
                max_seq_len=max_seq_len,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
            )
            valid_attn_mask = _get_valid_attn_mask(
                device=q.device,
                causal=causal,
                N=max_seq_len,
                seq_lengths=seq_lengths,
                num_targets=num_targets,
                max_attn_len=max_attn_len,
                contextual_seq_len=contextual_seq_len,
                min_full_attn_seq_len=min_full_attn_seq_len,
                row_start=start,
                row_end=end,
                col_start=col_start,
                col_end=col_end,
            )
            q_chunk = _pad_q_chunk(q, seq_offsets, start, end)
            k_chunk = k[:, :, col_start:col_end, :]
            v_chunk = v[:, :, col_start:col_end, :]
            qk_attn_chunk = torch.einsum("bhxa,bhya->bhxy", q_chunk, k_chunk) * alpha
            if prepared_attn_scale is not None:
                qk_attn_chunk = F.silu(qk_attn_chunk) * prepared_attn_scale[
                    :, :, start:end, :
                ]
            else:
                qk_attn_chunk = F.silu(qk_attn_chunk) / max_seq_len

            qk_attn_chunk = qk_attn_chunk * valid_attn_mask.unsqueeze(1)
            if dropout_pr > 0.0:
                qk_attn_chunk = F.dropout(
                    qk_attn_chunk,
                    p=dropout_pr,
                    training=training,
                )
            chunks.append(torch.einsum("bhxy,bhyv->bhxv", qk_attn_chunk, v_chunk))
        attn_dense = torch.cat(chunks, dim=2)
    else:
        q, k, v = _pad_qkv(
            q, k, v, seq_offsets, max_seq_len
        )  # [B, H, N, D) and [B, H, N, V]
        valid_attn_mask = _get_valid_attn_mask(
            device=q.device,
            causal=causal,
            N=max_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            min_full_attn_seq_len=min_full_attn_seq_len,
        )
        qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
        if prepared_attn_scale is not None:
            qk_attn = F.silu(qk_attn) * prepared_attn_scale
        else:
            qk_attn = F.silu(qk_attn) / max_seq_len
        qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
        if dropout_pr > 0.0:
            qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)
        attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N, V]

    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N, H, V]->[B, N, H * V]
        [seq_offsets],
        L,
    )[0].view(L, H, V)


@torch.fx.wrap
def pytorch_cached_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    L, H, D = delta_q.shape
    _, _, V = v.shape
    B = seq_offsets.size(0) - 1
    delta_size = L // B
    delta_q = delta_q.view(B, -1, H, D).transpose(1, 2)
    full_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(-1, H * D),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, D)
        .transpose(1, 2)
    )
    full_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(-1, H * V),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, V)
        .transpose(1, 2)
    )
    qk_attn = torch.einsum("bhxa,bhya->bhxy", delta_q, full_k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    full_valid_attn_mask = _get_valid_attn_mask(
        device=delta_q.device,
        causal=True,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    mask = torch.arange(max_seq_len, device=delta_q.device).view(1, -1)
    mask = torch.logical_and(
        mask >= (seq_lengths - delta_size).view(-1, 1),
        mask < seq_lengths.view(-1, 1),
    )
    valid_attn_mask = (
        full_valid_attn_mask.expand(B, -1, -1)
        .flatten(0, 1)[mask.view(-1), :]
        .view(-1, delta_size, max_seq_len)
    )
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    attn_output = torch.einsum("bhxd,bhdv->bhxv", qk_attn, full_v)
    return attn_output.transpose(1, 2).reshape(-1, H, V)
