# Recall Geo 伪补丁模板

说明：

- 这是函数级伪补丁，不会直接修改代码。
- 只覆盖召回链路 `research/*`，不包含 `dlrm_v3/*`。
- 目标是最小可跑版本：geo-aware item embedding + local negative sampling 对齐 + 全量召回评估复用。
- 默认基于你当前提供的数据样例：用户序列 csv + item geo csv。

---

## 文件 1: generative_recommenders/research/data/preprocessor.py

## 0) 先决策略：保证原始数据集不受影响

### 调整结论

需要调整。`preprocessor.py` 的 geo 相关逻辑应改为“可选且默认关闭”，避免影响仓库现有公开实验数据流。

### 兼容原则

1. 不改现有 `MovielensDataProcessor.preprocess_rating()` 的默认行为。
2. 不改现有 `AmazonDataProcessor.preprocess_rating()` 的默认行为。
3. geo 与自定义 schema 归一化只在“自定义入口”启用。
4. 默认参数必须保证原始配置不变（即不传 geo 文件也不启用额外归一化）。

### 推荐实现方式（伪补丁）

- 方案 A（推荐）：新增 `CustomSequenceDataProcessor`，仅处理你的外部序列文件与 geo 文件。
- 方案 B（次选）：在原有 `preprocess_rating()` 中加开关，默认 `False`。

若采用方案 B，建议新增参数：

```python
enable_custom_schema_normalization: bool = False
geo_csv_path: Optional[str] = None
```

并严格按开关执行：

```python
if enable_custom_schema_normalization:
    # 才执行 UserId/UTCTimeOffset/ratings 补齐/id 偏移
    ...

if geo_csv_path is not None and os.path.exists(geo_csv_path):
    # 才生成 item_geo_features.csv
    ...
```

---

## 1) 插入点总览（精确落位）

### Before

- 当前 `preprocessor.py` 没有 geo side-info 输出，也没有样例 schema 兼容逻辑。

### After (伪补丁)

文件 1 建议拆成 5 个插入点：

1. 在 `DataProcessor.output_format_csv()` 后新增通用路径方法。
2. 在 `DataProcessor.to_seq_data()` 后新增通用归一化辅助函数。
3. 在 `MovielensDataProcessor.preprocess_rating()` 的 `seq_ratings_data = self.to_seq_data(...)` 之前插入序列归一化。
4. 在 `MovielensDataProcessor.preprocess_rating()` 的序列构建阶段插入 geo 文件生成。
5. 在 `AmazonDataProcessor.preprocess_rating()` 按相同模式插入序列归一化与 geo 文件生成。

并补一个总开关要求：

- 只有在 `enable_custom_schema_normalization=True` 时才触发插入点 C/E。
- 只有在 `geo_csv_path` 有效时才触发插入点 D/E 的 geo 输出。

---

## 2) 插入点 A：`DataProcessor` 通用路径方法

### 位置

- 放在 `output_format_csv()` 之后、`to_seq_data()` 之前。

### After (伪补丁)

```python
def output_item_geo_features_csv(self) -> str:
    return f"tmp/processed/{self._prefix}/item_geo_features.csv"
```

说明：

- 统一 geo 输出路径，避免在各子类里写死。

---

## 3) 插入点 B：`DataProcessor` 通用辅助函数

### 位置

- 放在 `to_seq_data()` 之后、`file_exists()` 之前。

### After (伪补丁)

```python
def _normalize_custom_sequence_schema(self, seq_df: pd.DataFrame) -> pd.DataFrame:
    if "UserId" in seq_df.columns:
        seq_df = seq_df.rename(columns={"UserId": "user_id"})
    if "sequence_UTCTimeOffset" in seq_df.columns:
        seq_df = seq_df.rename(columns={"sequence_UTCTimeOffset": "sequence_timestamps"})

    # Current expected format:
    # - sequence_item_ids: "333, 286, 5093"
    # - sequence_timestamps: "1333845188,1333847015,1333847448"
    seq_df["sequence_item_ids"] = seq_df["sequence_item_ids"].apply(
        lambda s: [int(v.strip()) for v in str(s).split(",") if v.strip() != ""]
    )
    seq_df["sequence_timestamps"] = seq_df["sequence_timestamps"].apply(
        lambda s: [int(v.strip()) for v in str(s).split(",") if v.strip() != ""]
    )

    if "sequence_ratings" not in seq_df.columns:
        seq_df["sequence_ratings"] = seq_df["sequence_item_ids"].apply(
            lambda seq: [1 for _ in seq]
        )
    else:
        seq_df["sequence_ratings"] = seq_df["sequence_ratings"].apply(
            lambda s: [int(float(v.strip())) for v in str(s).split(",") if v.strip() != ""]
        )

    for row in seq_df.itertuples():
        assert len(row.sequence_item_ids) == len(row.sequence_ratings)
        assert len(row.sequence_item_ids) == len(row.sequence_timestamps)

    return seq_df


def _shift_item_ids_for_model(self, seq_df: pd.DataFrame) -> pd.DataFrame:
    # reserve 0 for padding/unknown
    seq_df["sequence_item_ids"] = seq_df["sequence_item_ids"].apply(
        lambda seq: [int(x) + 1 for x in seq]
    )
    return seq_df


def _build_item_geo_features_df(self, geo_df: pd.DataFrame) -> pd.DataFrame:
    geo_df = geo_df.copy()
    geo_df["item_id"] = geo_df["item_id"].astype(int) + 1

    geo_df["geo_cell_l5"] = geo_df.apply(
        lambda row: geohash_encode(row["Latitude"], row["Longitude"], precision=5)
        if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"])
        else "UNK",
        axis=1,
    )
    geo_df["geo_cell_l7"] = geo_df.apply(
        lambda row: geohash_encode(row["Latitude"], row["Longitude"], precision=7)
        if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"])
        else "UNK",
        axis=1,
    )

    # fallback without boundary service
    geo_df["geo_region_id"] = geo_df["geo_cell_l5"]

    for col in ["geo_region_id", "geo_cell_l5", "geo_cell_l7"]:
        geo_df[col] = pd.Categorical(geo_df[col]).codes + 1
        geo_df.loc[geo_df[col] < 1, col] = 0

    return geo_df[["item_id", "geo_region_id", "geo_cell_l5", "geo_cell_l7"]]


def _parse_sequence_column(self, x):
    # Current format only: comma-separated string, e.g. "333, 286, 5093"
    s = str(x).strip()
    if s == "":
        return []
    return [int(v.strip()) for v in s.split(",") if v.strip() != ""]
```

---

## 4) 插入点 C：`MovielensDataProcessor.preprocess_rating()` 序列归一化

### 位置

- 在 `seq_ratings_data = self.to_seq_data(seq_ratings_data, users)` 之前插入。

### After (伪补丁)

```python
seq_ratings_data = self.to_seq_data(seq_ratings_data, users)

if enable_custom_schema_normalization:
    # optional custom schema normalization for external sequence sources
    seq_ratings_data = self._normalize_custom_sequence_schema(seq_ratings_data)
    seq_ratings_data = self._shift_item_ids_for_model(seq_ratings_data)

seq_ratings_data.sample(frac=1).reset_index().to_csv(
    self.output_format_csv(), index=False, sep="," 
)
```

说明：

- 如果继续跑仓库原始公开数据，保持 `enable_custom_schema_normalization=False`。
- 如果用你的外部样例输入，再启用该开关。

---

## 5) 插入点 D：`MovielensDataProcessor.preprocess_rating()` geo 文件生成

### 位置

- 放在 seq 输出之前（建议在 `ratings_group = ...groupby(...)` 前后）。

### After (伪补丁)

```python
if geo_csv_path is not None and os.path.exists(geo_csv_path):
    geo_df = pd.read_csv(geo_csv_path)
    item_geo_df = self._build_item_geo_features_df(geo_df)
    item_geo_df.to_csv(self.output_item_geo_features_csv(), index=False)
```

说明：

- 这个输出和 `sasrec_format.csv` 同批次产出。

---

## 6) 插入点 E：`AmazonDataProcessor.preprocess_rating()` 对齐同样逻辑

### 位置

- 在 `seq_ratings_data = self.to_seq_data(seq_ratings_data)` 之前和之后，按 Movielens 同样结构插入。

### After (伪补丁)

```python
seq_ratings_data = self.to_seq_data(seq_ratings_data)

if enable_custom_schema_normalization:
    seq_ratings_data = self._normalize_custom_sequence_schema(seq_ratings_data)
    seq_ratings_data = self._shift_item_ids_for_model(seq_ratings_data)

seq_ratings_data.sample(frac=1).reset_index().to_csv(
    self.output_format_csv(), index=False, sep="," 
)

if geo_csv_path is not None and os.path.exists(geo_csv_path):
    geo_df = pd.read_csv(geo_csv_path)
    item_geo_df = self._build_item_geo_features_df(geo_df)
    item_geo_df.to_csv(self.output_item_geo_features_csv(), index=False)
```

---

## 7) 文件 1 最小验收

1. `sasrec_format.csv` 有且仅有训练所需列：
   - `user_id`
   - `sequence_item_ids`
   - `sequence_ratings`
   - `sequence_timestamps`
2. 所有 `sequence_item_ids` 的最小值 `>= 1`。
3. `item_geo_features.csv` 存在，字段为：
   - `item_id`
   - `geo_region_id`
   - `geo_cell_l5`
   - `geo_cell_l7`
4. `item_geo_features.csv` 中 `item_id` 最小值 `>= 1`。
5. 序列长度一致性断言不触发。

补充验收（避免影响原始数据集）：

6. 在默认参数下，原始 `ml-1m/ml-20m/amzn-books` 的输出文件与改动前保持同字段、同语义。
7. 在不提供 `geo_csv_path` 时，不生成或不覆盖原有实验关键文件。

---

## 8) 针对你当前已处理训练数据的特别说明

你当前序列样例是：

- `user_id,sequence_item_ids,sequence_timestamps`
- `sequence_item_ids` / `sequence_timestamps` 是逗号字符串（非方括号）
- 没有 `sequence_ratings`

这意味着：

1. 必须走 `enable_custom_schema_normalization=True`。
2. 只按逗号字符串格式解析，不再兼容方括号列表格式。
3. 必须补 `sequence_ratings`（默认全 1）才能兼容现有训练逻辑。
4. geo 文件中 `item_id` 也应执行与序列一致的 id 偏移策略（若训练保留 0 为 padding，则 geo 需同步 +1）。

---

## 文件 2: generative_recommenders/research/data/reco_dataset.py

## 5) `RecoDataset` 增加 geo tensor 字段

### Before

`RecoDataset` 只包含：

- `max_sequence_length`
- `num_unique_items`
- `max_item_id`
- `all_item_ids`
- `train_dataset`
- `eval_dataset`

### After (伪补丁)

在 `RecoDataset` dataclass 中新增：

- `item_geo_region_ids: torch.Tensor`
- `item_geo_cell_l5_ids: torch.Tensor`
- `item_geo_cell_l7_ids: torch.Tensor`

---

## 6) 读取 `item_geo_features.csv`

### Before

- `get_reco_dataset()` 不读取 geo 文件

### After (伪补丁)

在 `get_reco_dataset()` 末尾新增：

```python
geo_df = pd.read_csv(f"tmp/processed/{dataset_name}/item_geo_features.csv")

item_geo_region_ids = torch.zeros((max_item_id + 1,), dtype=torch.int64)
item_geo_cell_l5_ids = torch.zeros((max_item_id + 1,), dtype=torch.int64)
item_geo_cell_l7_ids = torch.zeros((max_item_id + 1,), dtype=torch.int64)

for row in geo_df.itertuples():
    item_id = int(row.item_id)
    if item_id > max_item_id:
        continue
    item_geo_region_ids[item_id] = int(row.geo_region_id)
    item_geo_cell_l5_ids[item_id] = int(row.geo_cell_l5)
    item_geo_cell_l7_ids[item_id] = int(row.geo_cell_l7)
```

返回时注入：

```python
return RecoDataset(
    ...,
    item_geo_region_ids=item_geo_region_ids,
    item_geo_cell_l5_ids=item_geo_cell_l5_ids,
    item_geo_cell_l7_ids=item_geo_cell_l7_ids,
)
```

---

## 7) 数据一致性断言

### Before

- 无 geo tensor 相关断言

### After (伪补丁)

新增：

```python
assert min(all_item_ids) >= 1
assert item_geo_region_ids.size(0) == max_item_id + 1
assert item_geo_cell_l5_ids.size(0) == max_item_id + 1
assert item_geo_cell_l7_ids.size(0) == max_item_id + 1
```

---

## 文件 3: generative_recommenders/research/modeling/sequential/embedding_modules.py

## 8) 新增 `GeoAwareEmbeddingModule`

### Before

- 只有：
  - `LocalEmbeddingModule`
  - `CategoricalEmbeddingModule`

### After (伪补丁)

新增：

```python
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
        ...
```

---

## 9) item embedding 与 geo embedding 融合

### Before

`LocalEmbeddingModule.get_item_embeddings()`：

```python
return self._item_emb(item_ids)
```

### After (伪补丁)

在 `GeoAwareEmbeddingModule.get_item_embeddings()` 中实现：

```python
item_emb = self._item_emb(item_ids)

region_ids = self._item_geo_region_ids[item_ids]
cell_l5_ids = self._item_geo_cell_l5_ids[item_ids]
cell_l7_ids = self._item_geo_cell_l7_ids[item_ids]

region_emb = self._region_emb(region_ids)
cell_l5_emb = self._cell_l5_emb(cell_l5_ids)
cell_l7_emb = self._cell_l7_emb(cell_l7_ids)

geo_emb = torch.cat([region_emb, cell_l5_emb, cell_l7_emb], dim=-1)
geo_delta = self._geo_proj(geo_emb)

return item_emb + geo_delta
```

说明：

- 保持输出形状与 `LocalEmbeddingModule` 一致
- 不改 SASRec/HSTU 主体

---

## 10) 参数初始化

### Before

- 只有 item embedding 初始化

### After (伪补丁)

新增 geo embedding 和投影层初始化：

```python
truncated_normal(self._item_emb.weight, mean=0.0, std=0.02)
truncated_normal(self._region_emb.weight, mean=0.0, std=0.02)
truncated_normal(self._cell_l5_emb.weight, mean=0.0, std=0.02)
truncated_normal(self._cell_l7_emb.weight, mean=0.0, std=0.02)

torch.nn.init.xavier_uniform_(self._geo_proj[0].weight)
```

---

## 文件 4: generative_recommenders/research/modeling/sequential/autoregressive_losses.py

## 11) `LocalNegativesSampler` 不再依赖裸 embedding table

### Before

构造函数：

```python
def __init__(self, num_items, item_emb, all_item_ids, ...)
```

forward 中：

```python
self._item_emb(sampled_ids)
```

### After (伪补丁)

构造函数改为：

```python
def __init__(self, num_items, embedding_module, all_item_ids, ...)
```

并保存：

```python
self._embedding_module = embedding_module
```

forward 改为：

```python
sampled_embeddings = self._embedding_module.get_item_embeddings(sampled_ids)
return sampled_ids, self.normalize_embeddings(sampled_embeddings)
```

说明：

- 这是 geo-aware 召回正确训练的必要条件
- 否则正样本和负样本不在同一 embedding 语义空间中

---

## 文件 5: generative_recommenders/research/trainer/train.py

## 12) `train_fn` 参数新增 geo embedding 配置

### Before

- 只有 `embedding_module_type: str = "local"`

### After (伪补丁)

在 `train_fn` 参数中新增：

- `geo_embedding_dim: int = 16`
- `num_geo_regions: int = 4096`
- `num_geo_cells_l5: int = 65536`
- `num_geo_cells_l7: int = 262144`

---

## 13) embedding module 构造分支

### Before

```python
if embedding_module_type == "local":
    embedding_module = LocalEmbeddingModule(...)
else:
    raise ValueError(...)
```

### After (伪补丁)

```python
if embedding_module_type == "local":
    embedding_module = LocalEmbeddingModule(
        num_items=dataset.max_item_id,
        item_embedding_dim=item_embedding_dim,
    )
elif embedding_module_type == "geo_aware":
    embedding_module = GeoAwareEmbeddingModule(
        num_items=dataset.max_item_id,
        item_embedding_dim=item_embedding_dim,
        item_geo_region_ids=dataset.item_geo_region_ids,
        item_geo_cell_l5_ids=dataset.item_geo_cell_l5_ids,
        item_geo_cell_l7_ids=dataset.item_geo_cell_l7_ids,
        num_geo_regions=num_geo_regions,
        num_geo_cells_l5=num_geo_cells_l5,
        num_geo_cells_l7=num_geo_cells_l7,
        geo_embedding_dim=geo_embedding_dim,
    )
else:
    raise ValueError(...)
```

---

## 14) `LocalNegativesSampler` 初始化参数更新

### Before

```python
negatives_sampler = LocalNegativesSampler(
    num_items=dataset.max_item_id,
    item_emb=model._embedding_module._item_emb,
    all_item_ids=dataset.all_item_ids,
    ...,
)
```

### After (伪补丁)

```python
negatives_sampler = LocalNegativesSampler(
    num_items=dataset.max_item_id,
    embedding_module=embedding_module,
    all_item_ids=dataset.all_item_ids,
    ...,
)
```

说明：

- 不再假设 embedding module 内部一定有 `_item_emb`

---

## 文件 6: configs/*.gin

## 15) geo-aware 召回实验配置

### Before

- 只有 local item embedding 配置

### After (伪补丁)

新增一份 geo-aware 配置，例如：

```python
train_fn.embedding_module_type = "geo_aware"
train_fn.item_embedding_dim = 64
train_fn.geo_embedding_dim = 16
train_fn.num_geo_regions = 4096
train_fn.num_geo_cells_l5 = 65536
train_fn.num_geo_cells_l7 = 262144
```

---

## 最小联调顺序

1. 先改 `preprocessor.py`，把输入 schema、timestamps、item id 偏移、geo 文件产出打通
2. 再改 `reco_dataset.py`，让 geo 特征进入 dataset 元信息
3. 再加 `GeoAwareEmbeddingModule`
4. 再改 `LocalNegativesSampler`
5. 最后改 `train.py` 与 gin 配置
6. 训练前先做一个 smoke test：
   - 给定一小批 item ids
   - 验证 `GeoAwareEmbeddingModule.get_item_embeddings()` 输出 finite
   - 验证 `LocalNegativesSampler` 可正常返回 negative embeddings

---

## 最小验收标准

1. 训练样本可被 `DatasetV2` 正常读取
2. `sequence_item_ids` 中不存在业务 `0`
3. geo-aware 模式首个训练 step 无 NaN/Inf
4. local negative sampling 与主模型 item embedding 语义一致
5. `eval.py` 可在不改主逻辑的情况下完成全量召回评估