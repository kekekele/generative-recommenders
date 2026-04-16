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

from dataclasses import dataclass
import os
from typing import List

import pandas as pd
import torch
from generative_recommenders.research.data.dataset import DatasetV2, MultiFileDatasetV2
from generative_recommenders.research.data.item_features import ItemFeatures
from generative_recommenders.research.data.preprocessor import get_common_preprocessors


# Modification log:
# - 2026-04-16: Added geo-side tensor support for recall-stage dataset assembly.
#   RecoDataset now includes item_geo_region_ids/item_geo_cell_l5_ids/
#   item_geo_cell_l7_ids and loads them from item_geo_features.csv when present.


@dataclass
class RecoDataset:
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset
    item_geo_region_ids: torch.Tensor
    item_geo_cell_l5_ids: torch.Tensor
    item_geo_cell_l7_ids: torch.Tensor
    item_geo_lat_norm: torch.Tensor
    item_geo_lon_norm: torch.Tensor


def _get_item_geo_features_csv_candidates(dataset_name: str) -> List[str]:
    # Keep both naming conventions to support existing and custom preprocess outputs.
    return [
        f"tmp/processed/{dataset_name}/item_geo_features.csv",
        f"tmp/processed/{dataset_name.replace('-', '_')}/item_geo_features.csv",
    ]


def _get_item_geo_fourier_features_csv_candidates(dataset_name: str) -> List[str]:
    return [
        f"tmp/processed/{dataset_name}/item_geo_fourier_features.csv",
        f"tmp/processed/{dataset_name.replace('-', '_')}/item_geo_fourier_features.csv",
    ]


def _load_item_geo_tensors(
    dataset_name: str,
    max_item_id: int,
) -> List[torch.Tensor]:
    item_geo_region_ids = torch.zeros((max_item_id + 1,), dtype=torch.int64)
    item_geo_cell_l5_ids = torch.zeros((max_item_id + 1,), dtype=torch.int64)
    item_geo_cell_l7_ids = torch.zeros((max_item_id + 1,), dtype=torch.int64)

    geo_csv = None
    for candidate in _get_item_geo_features_csv_candidates(dataset_name):
        if os.path.exists(candidate):
            geo_csv = candidate
            break

    if geo_csv is None:
        print(
            f"[reco_dataset] item_geo_features.csv not found for {dataset_name}; "
            "falling back to all-zero geo features"
        )
        return [item_geo_region_ids, item_geo_cell_l5_ids, item_geo_cell_l7_ids]

    geo_df = pd.read_csv(geo_csv)
    required_cols = ["item_id", "geo_region_id", "geo_cell_l5", "geo_cell_l7"]
    for col in required_cols:
        if col not in geo_df.columns:
            raise ValueError(
                f"{geo_csv} missing required column {col}; found {list(geo_df.columns)}"
            )

    for row in geo_df.itertuples(index=False):
        item_id = int(getattr(row, "item_id"))
        if item_id < 0 or item_id > max_item_id:
            continue
        item_geo_region_ids[item_id] = int(getattr(row, "geo_region_id"))
        item_geo_cell_l5_ids[item_id] = int(getattr(row, "geo_cell_l5"))
        item_geo_cell_l7_ids[item_id] = int(getattr(row, "geo_cell_l7"))

    return [item_geo_region_ids, item_geo_cell_l5_ids, item_geo_cell_l7_ids]


def _load_item_geo_fourier_tensors(
    dataset_name: str,
    max_item_id: int,
) -> List[torch.Tensor]:
    item_geo_lat_norm = torch.zeros((max_item_id + 1,), dtype=torch.float32)
    item_geo_lon_norm = torch.zeros((max_item_id + 1,), dtype=torch.float32)

    geo_csv = None
    for candidate in _get_item_geo_fourier_features_csv_candidates(dataset_name):
        if os.path.exists(candidate):
            geo_csv = candidate
            break

    if geo_csv is None:
        print(
            f"[reco_dataset] item_geo_fourier_features.csv not found for {dataset_name}; "
            "falling back to all-zero Fourier geo features"
        )
        return [item_geo_lat_norm, item_geo_lon_norm]

    geo_df = pd.read_csv(geo_csv)
    required_cols = ["item_id", "lat_norm", "lon_norm"]
    for col in required_cols:
        if col not in geo_df.columns:
            raise ValueError(
                f"{geo_csv} missing required column {col}; found {list(geo_df.columns)}"
            )

    for row in geo_df.itertuples(index=False):
        item_id = int(getattr(row, "item_id"))
        if item_id < 0 or item_id > max_item_id:
            continue
        item_geo_lat_norm[item_id] = float(getattr(row, "lat_norm"))
        item_geo_lon_norm[item_id] = float(getattr(row, "lon_norm"))

    return [item_geo_lat_norm, item_geo_lon_norm]


def _infer_item_stats_from_sequence_csv(ratings_file: str) -> List[int]:
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"Ratings file not found: {ratings_file}")

    frame = pd.read_csv(ratings_file, delimiter=",")
    if "sequence_item_ids" not in frame.columns:
        raise ValueError(
            f"{ratings_file} missing required column sequence_item_ids; "
            f"found {list(frame.columns)}"
        )

    all_item_ids: List[int] = []
    for seq_str in frame["sequence_item_ids"].tolist():
        seq_text = str(seq_str).strip()
        if seq_text == "":
            continue
        parsed = eval(seq_text)
        if isinstance(parsed, int):
            all_item_ids.append(int(parsed))
        else:
            all_item_ids.extend([int(x) for x in list(parsed)])

    if len(all_item_ids) == 0:
        raise ValueError(f"No item ids found in {ratings_file}")

    max_item_id = max(all_item_ids)
    num_unique_items = len(set(all_item_ids))
    return [num_unique_items, max_item_id]


def get_reco_dataset(
    dataset_name: str,
    max_sequence_length: int,
    chronological: bool,
    positional_sampling_ratio: float = 1.0,
) -> RecoDataset:
    num_unique_items = None
    if dataset_name == "ml-1m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
            sample_ratio=positional_sampling_ratio,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
            sample_ratio=1.0,  # do not sample
        )
    elif dataset_name == "ml-20m":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
        )
    elif dataset_name == "ml-3b":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = MultiFileDatasetV2(
            file_prefix="tmp/ml-3b/16x32",
            num_files=16,
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = MultiFileDatasetV2(
            file_prefix="tmp/ml-3b/16x32",
            num_files=16,
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
        )
    elif dataset_name == "amzn-books":
        dp = get_common_preprocessors()[dataset_name]
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            shift_id_by=1,  # [0..n-1] -> [1..n]
            chronological=chronological,
        )
    elif os.path.exists(f"tmp/{dataset_name}/sasrec_format.csv"):
        ratings_file = f"tmp/{dataset_name}/sasrec_format.csv"
        train_dataset = DatasetV2(
            ratings_file=ratings_file,
            padding_length=max_sequence_length + 1,
            ignore_last_n=1,
            chronological=chronological,
        )
        eval_dataset = DatasetV2(
            ratings_file=ratings_file,
            padding_length=max_sequence_length + 1,
            ignore_last_n=0,
            chronological=chronological,
        )

        [num_unique_items, inferred_max_item_id] = _infer_item_stats_from_sequence_csv(
            ratings_file
        )
        max_item_id = inferred_max_item_id
        all_item_ids = [x for x in range(1, max_item_id + 1)]
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    if dataset_name == "ml-1m" or dataset_name == "ml-20m":
        items = pd.read_csv(dp.processed_item_csv(), delimiter=",")
        max_jagged_dimension = 16
        expected_max_item_id = dp.expected_max_item_id()
        assert expected_max_item_id is not None
        item_features: ItemFeatures = ItemFeatures(
            max_ind_range=[63, 16383, 511],
            num_items=expected_max_item_id + 1,
            max_jagged_dimension=max_jagged_dimension,
            lengths=[
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
                torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
            ],
            values=[
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
                torch.zeros(
                    (expected_max_item_id + 1, max_jagged_dimension),
                    dtype=torch.int64,
                ),
            ],
        )
        all_item_ids = []
        for df_index, row in items.iterrows():
            # print(f"index {df_index}: {row}")
            movie_id = int(row["movie_id"])
            genres = row["genres"].split("|")
            titles = row["cleaned_title"].split(" ")
            # print(f"{index}: genres{genres}, title{titles}")
            genres_vector = [hash(x) % item_features.max_ind_range[0] for x in genres]
            titles_vector = [hash(x) % item_features.max_ind_range[1] for x in titles]
            years_vector = [hash(row["year"]) % item_features.max_ind_range[2]]
            item_features.lengths[0][movie_id] = min(
                len(genres_vector), max_jagged_dimension
            )
            item_features.lengths[1][movie_id] = min(
                len(titles_vector), max_jagged_dimension
            )
            item_features.lengths[2][movie_id] = min(
                len(years_vector), max_jagged_dimension
            )
            for f, f_values in enumerate([genres_vector, titles_vector, years_vector]):
                for j in range(min(len(f_values), max_jagged_dimension)):
                    item_features.values[f][movie_id][j] = f_values[j]
            all_item_ids.append(movie_id)
        max_item_id = dp.expected_max_item_id()
        for x in all_item_ids:
            assert x > 0, "x in all_item_ids should be positive"
    elif dataset_name in ["amzn-books", "ml-3b"]:
        # expected_max_item_id and item_features are not set for Amazon datasets.
        item_features = None
        max_item_id = dp.expected_num_unique_items()
        all_item_ids = [x + 1 for x in range(max_item_id)]  # pyre-ignore [6]
    else:
        # Custom datasets use inferred max_item_id/all_item_ids from sequence csv.
        item_features = None

    [
        item_geo_region_ids,
        item_geo_cell_l5_ids,
        item_geo_cell_l7_ids,
    ] = _load_item_geo_tensors(
        dataset_name=dataset_name,
        max_item_id=max_item_id,  # pyre-ignore [6]
    )

    [item_geo_lat_norm, item_geo_lon_norm] = _load_item_geo_fourier_tensors(
        dataset_name=dataset_name,
        max_item_id=max_item_id,  # pyre-ignore [6]
    )

    if len(all_item_ids) > 0:
        assert min(all_item_ids) >= 1, "all_item_ids must reserve 0 for padding"

    assert item_geo_region_ids.size(0) == max_item_id + 1  # pyre-ignore [6]
    assert item_geo_cell_l5_ids.size(0) == max_item_id + 1  # pyre-ignore [6]
    assert item_geo_cell_l7_ids.size(0) == max_item_id + 1  # pyre-ignore [6]
    assert item_geo_lat_norm.size(0) == max_item_id + 1  # pyre-ignore [6]
    assert item_geo_lon_norm.size(0) == max_item_id + 1  # pyre-ignore [6]

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=(
            dp.expected_num_unique_items() if num_unique_items is None else num_unique_items
        ),
        max_item_id=max_item_id,  # pyre-ignore [6]
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        item_geo_region_ids=item_geo_region_ids,
        item_geo_cell_l5_ids=item_geo_cell_l5_ids,
        item_geo_cell_l7_ids=item_geo_cell_l7_ids,
        item_geo_lat_norm=item_geo_lat_norm,
        item_geo_lon_norm=item_geo_lon_norm,
    )
