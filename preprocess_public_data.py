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

"""
Usage: mkdir -p tmp/ && python3 preprocess_public_data.py

Custom data usage example:
python3 preprocess_public_data.py \
  --custom-prefix ml-1m-custom \
  --custom-sequence-csv /path/to/sequence.csv \
  --custom-geo-csv /path/to/item_geo.csv \
  --custom-item-id-offset 1
"""

import argparse

from generative_recommenders.research.data.preprocessor import (
    CustomSequenceDataProcessor,
    get_common_preprocessors,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom-prefix",
        type=str,
        default=None,
        help="Dataset prefix used for outputs under tmp/<prefix>/",
    )
    parser.add_argument(
        "--custom-sequence-csv",
        type=str,
        default=None,
        help="Path to custom sequence csv (required for custom mode)",
    )
    parser.add_argument(
        "--custom-geo-csv",
        type=str,
        default=None,
        help="Path to custom geo csv (optional for custom mode)",
    )
    parser.add_argument(
        "--custom-item-id-offset",
        type=int,
        default=1,
        help="Item id offset for model ids, typically 1 to reserve 0 for padding",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.custom_sequence_csv is not None:
        prefix = args.custom_prefix or "custom-seq"
        processor = CustomSequenceDataProcessor(
            prefix=prefix,
            sequence_csv_path=args.custom_sequence_csv,
            geo_csv_path=args.custom_geo_csv,
            item_id_offset=args.custom_item_id_offset,
        )
        processor.preprocess_rating()
        return

    preprocessors = get_common_preprocessors()
    preprocessors["ml-1m"].preprocess_rating()
    preprocessors["ml-20m"].preprocess_rating()
    # preprocessors["ml-1b"].preprocess_rating()
    preprocessors["amzn-books"].preprocess_rating()


if __name__ == "__main__":
    main()
