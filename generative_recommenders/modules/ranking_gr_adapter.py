# pyre-strict

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from generative_recommenders.modules.multitask_module import MultitaskTaskType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def _build_mlp(
    in_dim: int,
    layer_sizes: List[int],
    activation: str = "relu",
    bias: bool = True,
) -> torch.nn.Module:
    if activation == "relu":
        activation_cls = torch.nn.ReLU
    elif activation == "gelu":
        activation_cls = torch.nn.GELU
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    dims = [in_dim] + layer_sizes
    layers: List[torch.nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1], bias=bias))
        if i < len(dims) - 2:
            layers.append(activation_cls())
    return torch.nn.Sequential(*layers)


@dataclass
class RankingAdapterConfig:
    prediction_head_arch: Tuple[int, ...] = (512, 1)
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True


class RankingGRAdapter(torch.nn.Module):
    """
    Ranking adapter that keeps the DlrmHSTU feature pipeline but swaps the
    prediction head + loss logic to a RankingGR-style binary multitask loss.
    """

    def __init__(
        self,
        hstu_configs: DlrmHSTUConfig,
        embedding_tables: Dict[str, EmbeddingConfig],
        is_inference: bool,
        ranking_config: Optional[RankingAdapterConfig] = None,
        bf16_training: bool = True,
    ) -> None:
        super().__init__()
        self._is_inference = is_inference
        self._backbone = DlrmHSTU(
            hstu_configs=hstu_configs,
            embedding_tables=embedding_tables,
            is_inference=is_inference,
            bf16_training=bf16_training,
        )

        cfg = ranking_config or RankingAdapterConfig()
        self._task_configs = hstu_configs.multitask_configs
        self._num_tasks = len(self._task_configs)
        if self._num_tasks < 1:
            raise ValueError("RankingGRAdapter requires at least one task")
        if not all(
            t.task_type == MultitaskTaskType.BINARY_CLASSIFICATION
            for t in self._task_configs
        ):
            raise ValueError(
                "RankingGRAdapter currently supports binary classification tasks only"
            )

        head_dims = list(cfg.prediction_head_arch)
        if len(head_dims) == 0:
            raise ValueError("prediction_head_arch must be non-empty")
        if head_dims[-1] != self._num_tasks:
            head_dims[-1] = self._num_tasks

        self._ranking_head = _build_mlp(
            in_dim=hstu_configs.hstu_transducer_embedding_dim,
            layer_sizes=head_dims,
            activation=cfg.prediction_head_act_type,
            bias=cfg.prediction_head_bias,
        )

    def set_hammer_kernel(self, hammer_kernel: HammerKernel) -> None:
        self._backbone.set_hammer_kernel(hammer_kernel)

    def set_use_triton_cc(self, use_triton_cc: bool) -> None:
        self._backbone.set_use_triton_cc(use_triton_cc)

    def set_training_dtype(self, training_dtype: torch.dtype) -> None:
        self._backbone.set_training_dtype(training_dtype)

    def set_is_inference(self, is_inference: bool) -> None:
        self._is_inference = is_inference
        self._backbone.set_is_inference(is_inference)

    def forward(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        (
            candidates_user_embeddings,
            mt_target_labels,
            mt_target_weights,
        ) = self._backbone.forward_user_tower(
            uih_features=uih_features,
            candidates_features=candidates_features,
        )
        candidates_item_embeddings = torch.empty(
            0,
            device=candidates_user_embeddings.device,
            dtype=candidates_user_embeddings.dtype,
        )

        logits = self._ranking_head(candidates_user_embeddings).transpose(0, 1)
        predictions = torch.sigmoid(logits)

        aux_losses: Dict[str, torch.Tensor] = {}
        if not self._is_inference and mt_target_labels is not None:
            labels = mt_target_labels.to(dtype=logits.dtype)
            if mt_target_weights is None:
                weights = torch.ones_like(labels)
            else:
                weights = mt_target_weights.to(dtype=logits.dtype)

            per_entry = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
            per_task = (per_entry * weights).sum(-1) / weights.sum(-1).clamp(min=1.0)
            aux_losses["ranking_adapter_loss"] = per_task.mean()

        return (
            candidates_user_embeddings,
            candidates_item_embeddings,
            aux_losses,
            predictions,
            mt_target_labels,
            mt_target_weights,
        )
