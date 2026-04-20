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

# pyre-unsafe

import abc

import torch
from generative_recommenders.research.modeling.initialization import truncated_normal


# Modification log:
# - 2026-04-16: Added GeoAwareEmbeddingModule for recall-stage item embedding
#   adaptation with geo side-information (region/cell-l5/cell-l7) and additive
#   fusion on top of item-id embedding.


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class GeoAwareEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_geo_region_ids: torch.Tensor,
        item_geo_cell_l5_ids: torch.Tensor,
        item_geo_cell_l7_ids: torch.Tensor,
        num_geo_regions: int,
        num_geo_cells_l5: int,
        num_geo_cells_l7: int,
        geo_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1,
            item_embedding_dim,
            padding_idx=0,
        )

        self._geo_region_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_geo_regions + 1,
            geo_embedding_dim,
            padding_idx=0,
        )
        self._geo_cell_l5_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_geo_cells_l5 + 1,
            geo_embedding_dim,
            padding_idx=0,
        )
        self._geo_cell_l7_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_geo_cells_l7 + 1,
            geo_embedding_dim,
            padding_idx=0,
        )

        # Keep geo lookup tensors with the module for DDP/device moves.
        self.register_buffer("_item_geo_region_ids", item_geo_region_ids.long())
        self.register_buffer("_item_geo_cell_l5_ids", item_geo_cell_l5_ids.long())
        self.register_buffer("_item_geo_cell_l7_ids", item_geo_cell_l7_ids.long())

        # Additive residual fusion: item_id_emb + proj([region, l5, l7]).
        # No bias in projection so zero-input stays zero for padding paths.
        self._geo_proj: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=geo_embedding_dim * 3,
                out_features=item_embedding_dim,
                bias=False,
            ),
            torch.nn.LayerNorm(item_embedding_dim),
        )

        self.reset_params()

    def debug_str(self) -> str:
        return f"geo_aware_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        truncated_normal(self._item_emb.weight, mean=0.0, std=0.02)
        truncated_normal(self._geo_region_emb.weight, mean=0.0, std=0.02)
        truncated_normal(self._geo_cell_l5_emb.weight, mean=0.0, std=0.02)
        truncated_normal(self._geo_cell_l7_emb.weight, mean=0.0, std=0.02)

        for module in self._geo_proj.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)

    def _safe_geo_lookup(self, item_ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        # Guard against out-of-range ids during warm-up/data mismatch.
        max_id = table.size(0) - 1
        safe_ids = item_ids.clamp(min=0, max=max_id)
        return table[safe_ids]

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_emb = self._item_emb(item_ids)

        region_ids = self._safe_geo_lookup(item_ids, self._item_geo_region_ids)
        cell_l5_ids = self._safe_geo_lookup(item_ids, self._item_geo_cell_l5_ids)
        cell_l7_ids = self._safe_geo_lookup(item_ids, self._item_geo_cell_l7_ids)

        region_ids = region_ids.clamp(min=0, max=self._geo_region_emb.num_embeddings - 1)
        cell_l5_ids = cell_l5_ids.clamp(min=0, max=self._geo_cell_l5_emb.num_embeddings - 1)
        cell_l7_ids = cell_l7_ids.clamp(min=0, max=self._geo_cell_l7_emb.num_embeddings - 1)

        region_emb = self._geo_region_emb(region_ids)
        cell_l5_emb = self._geo_cell_l5_emb(cell_l5_ids)
        cell_l7_emb = self._geo_cell_l7_emb(cell_l7_ids)

        geo_emb = torch.cat([region_emb, cell_l5_emb, cell_l7_emb], dim=-1)
        geo_delta = self._geo_proj(geo_emb)

        # Keep padding id=0 strictly on the item embedding path.
        geo_delta = geo_delta * (item_ids != 0).unsqueeze(-1).to(geo_delta.dtype)
        return item_emb + geo_delta

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class FourierGeoEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_geo_fourier_features: torch.Tensor,
        item_visit_time_features: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1,
            item_embedding_dim,
            padding_idx=0,
        )

        self.register_buffer(
            "_item_geo_fourier_features",
            item_geo_fourier_features.float(),
        )
        self.register_buffer("_item_visit_time_features", item_visit_time_features.float())
        self._geo_gate_max_scale: float = 0.2

        if item_geo_fourier_features.dim() != 2:
            raise ValueError("item_geo_fourier_features must be rank-2 [num_items+1, dim]")
        fourier_dim = item_geo_fourier_features.size(1)

        self._geo_proj: torch.nn.Module = torch.nn.Linear(
            in_features=fourier_dim + 24,
            out_features=item_embedding_dim,
            bias=False,
        )
        # Minimal adaptive gate: initialized around 0.05 and bounded by max scale.
        self._geo_gate: torch.nn.Module = torch.nn.Linear(
            in_features=item_embedding_dim * 2,
            out_features=1,
            bias=True,
        )
        self.register_buffer("_last_gate_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_last_gate_min", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_last_gate_max", torch.tensor(0.0, dtype=torch.float32))

        self.reset_params()

    def debug_str(self) -> str:
        return f"fourier_geo_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        truncated_normal(self._item_emb.weight, mean=0.0, std=0.02)
        for module in self._geo_proj.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
        if isinstance(self._geo_gate, torch.nn.Linear):
            torch.nn.init.zeros_(self._geo_gate.weight)
            # 0.2 * sigmoid(bias) ~= 0.05  => sigmoid(bias)=0.25
            self._geo_gate.bias.data.fill_(-1.0986123)

    def _safe_lookup(self, item_ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        max_id = table.size(0) - 1
        safe_ids = item_ids.clamp(min=0, max=max_id)
        return table[safe_ids]

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_emb = self._item_emb(item_ids)

        fourier_feat = self._safe_lookup(item_ids, self._item_geo_fourier_features)
        visit_time_feat = self._safe_lookup(item_ids, self._item_visit_time_features)
        full_geo_feat = torch.cat([fourier_feat, visit_time_feat], dim=-1)
        geo_delta = self._geo_proj(full_geo_feat)

        valid_mask = (item_ids != 0).unsqueeze(-1).to(geo_delta.dtype)
        geo_delta = geo_delta * valid_mask

        gate_input = torch.cat([item_emb, geo_delta], dim=-1)
        gate = self._geo_gate_max_scale * torch.sigmoid(self._geo_gate(gate_input))
        gate = gate.to(geo_delta.dtype) * valid_mask
        gate_detached = gate.detach()
        self._last_gate_mean.copy_(gate_detached.mean().to(torch.float32))
        self._last_gate_min.copy_(gate_detached.min().to(torch.float32))
        self._last_gate_max.copy_(gate_detached.max().to(torch.float32))
        return item_emb + gate * geo_delta

    def get_gate_diagnostics(self) -> dict:
        gate_bias = float(self._geo_gate.bias.detach().view(-1)[0].item())
        gate_weight_norm = float(self._geo_gate.weight.detach().norm().item())
        return {
            "gate_bias": gate_bias,
            "gate_weight_norm": gate_weight_norm,
            "gate_last_mean": float(self._last_gate_mean.detach().item()),
            "gate_last_min": float(self._last_gate_min.detach().item()),
            "gate_last_max": float(self._last_gate_max.detach().item()),
        }

    def get_gate_scalar(self) -> float:
        # A compact global proxy for gate strength.
        gate_bias = self._geo_gate.bias.detach().view(-1)[0]
        return float((self._geo_gate_max_scale * torch.sigmoid(gate_bias)).item())

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class FourierGeoConcatEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_geo_fourier_features: torch.Tensor,
        item_visit_time_features: torch.Tensor,
        branch_dropout_rate: float = 0.1,
        use_item_residual_anchor: bool = False,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._use_item_residual_anchor: bool = use_item_residual_anchor
        self._residual_scale: float = residual_scale

        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1,
            item_embedding_dim,
            padding_idx=0,
        )
        self.register_buffer(
            "_item_geo_fourier_features",
            item_geo_fourier_features.float(),
        )
        self.register_buffer("_item_visit_time_features", item_visit_time_features.float())

        if item_geo_fourier_features.dim() != 2:
            raise ValueError("item_geo_fourier_features must be rank-2 [num_items+1, dim]")
        fourier_dim = item_geo_fourier_features.size(1)

        self._item_branch: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(item_embedding_dim, item_embedding_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Dropout(branch_dropout_rate),
        )
        self._geo_branch: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(fourier_dim, item_embedding_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Dropout(branch_dropout_rate),
        )
        self._visit_time_branch: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(24, item_embedding_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Dropout(branch_dropout_rate),
        )
        self._fusion_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(item_embedding_dim * 3, item_embedding_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.LayerNorm(item_embedding_dim),
        )

        self.reset_params()

    def debug_str(self) -> str:
        variant = "b" if self._use_item_residual_anchor else "a"
        return f"fourier_geo_concat_{variant}_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        truncated_normal(self._item_emb.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def _safe_lookup(self, item_ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        max_id = table.size(0) - 1
        safe_ids = item_ids.clamp(min=0, max=max_id)
        return table[safe_ids]

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_emb = self._item_emb(item_ids)

        fourier_feat = self._safe_lookup(item_ids, self._item_geo_fourier_features)
        visit_time_feat = self._safe_lookup(item_ids, self._item_visit_time_features)

        item_h = self._item_branch(item_emb)
        geo_h = self._geo_branch(fourier_feat)
        visit_h = self._visit_time_branch(visit_time_feat)

        fused_h = self._fusion_mlp(torch.cat([item_h, geo_h, visit_h], dim=-1))
        valid_mask = (item_ids != 0).unsqueeze(-1).to(fused_h.dtype)
        fused_h = fused_h * valid_mask

        if self._use_item_residual_anchor:
            return item_emb + self._residual_scale * fused_h
        return fused_h

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
