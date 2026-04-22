# Recall-Only Item Geo Adaptation Plan

## Scope

本文件只讨论仓库中的召回链路，也就是 README 中公共实验使用的 `research/*` 路径：

- `main.py`
- `generative_recommenders/research/trainer/train.py`
- `generative_recommenders/research/data/*`
- `generative_recommenders/research/modeling/sequential/*`
- `generative_recommenders/research/data/eval.py`

不讨论 `dlrm_v3/*`。

## Current Retrieval Path

当前公共实验的召回训练路径是：

1. `preprocess_public_data.py` 生成 `tmp/<dataset>/sasrec_format.csv`
2. `research/data/dataset.py` 读入用户历史序列、评分、时间戳
3. `research/modeling/sequential/features.py` 把一行样本转换成 `SequentialFeatures`
4. `train.py` 里通过 `model.get_item_embeddings(seq_features.past_ids)` 生成历史 item embedding
5. SASRec 或 HSTU 对序列做编码
6. 训练、负采样、评估、全量召回都通过 `model.get_item_embeddings()` 获取 item 向量

这条链路有一个关键事实：

- 默认配置下，item 特征只有 item id embedding。

虽然 `research/data/reco_dataset.py` 里为 MovieLens 构造了 `genres/title/year` 的 `item_features`，但当前 `train_fn()` 实际只实例化了 `LocalEmbeddingModule`，并没有把这些 side information 接进模型。

## Compatibility Update For Your Sample

你提供的数据样例和当前召回代码的默认输入格式存在 4 个差异，需要在文档中明确修正：

1. 列名差异：样例是 `UserId`，当前代码读取 `user_id`。
2. 时间列差异：样例是 `sequence_UTCTimeOffset` 且元素是时间字符串，当前代码读取 `sequence_timestamps` 且按整型时间戳处理。
3. 评分列缺失：样例没有 `sequence_ratings`，而 `DatasetV2` 与 `movielens_seq_features_from_row` 默认需要该列。
4. item id 从 0 开始：样例和 geo 表都出现 `item_id=0`，但当前模型保留 `0` 作为 padding id。

因此在“只看召回链路”的实现中，必须先做 schema 对齐与 id 对齐，再接入 geo-aware embedding。

### Required schema normalization

训练输入文件建议统一成以下列：

- `user_id`
- `sequence_item_ids`
- `sequence_ratings`
- `sequence_timestamps`

样例到训练输入的转换规则：

- `UserId -> user_id`
- `sequence_UTCTimeOffset -> sequence_timestamps`
- 时间字符串列表统一转换为 Unix 秒级整型列表
- 若无显式评分，生成 `sequence_ratings`（默认值全 1）

### Required item id normalization

由于 `0` 在当前召回模型中是 padding id，业务 item id 不能继续使用 `0`。

建议规则：

- 训练序列中的所有业务 item id 统一 `+1`
- geo side-info 中 `item_id` 同步 `+1`
- 保留 `0` 仅用于 padding 与缺失

这样可以避免把真实 item 当成 padding，导致训练与评估偏差。

## Key Conclusion

如果只看召回部分，给 item 增加地理位置特征的最合适入口不是 dataset payload，而是 item embedding module。

原因：

- 历史序列的 item embedding 通过 `model.get_item_embeddings()` 获取
- 训练时正样本 embedding 通过 `model.get_item_embeddings()` 获取
- 全量评估时候选库 embedding 通过 `model.get_item_embeddings()` 获取
- 相似度模块也默认通过 `get_item_embeddings()` 拿 item 向量

因此，只要 geo 信息能稳定地进入 `EmbeddingModule.get_item_embeddings()`，就能自动覆盖训练、负采样、离线评估和召回索引构建。

## Recommended Design

### Core idea

新增一个 geo-aware 的 `EmbeddingModule`，让每个 item 的最终 embedding 由以下几部分组合得到：

- item id embedding
- geo region embedding
- geo cell coarse embedding
- geo cell fine embedding

推荐最终形式：

```text
item_embedding = MLP(concat(item_id_emb, geo_region_emb, geo_cell_l5_emb, geo_cell_l7_emb))
```

或者更轻量：

```text
item_embedding = item_id_emb + proj_geo(concat(geo_region_emb, geo_cell_l5_emb, geo_cell_l7_emb))
```

更推荐第二种，改动更小，也更符合当前实现风格。

## Why Not Put Geo Into Sequence Payload First

`SequentialFeatures.past_payloads` 目前主要承载：

- `timestamps`
- `ratings`

这些 payload 的用途是给 input preprocessor 和 encoder 增强序列建模。

但 item geo 是 item 静态属性，不应该只作用在“历史侧输入”，还必须作用在：

- 正样本 target item
- 负样本 item
- 全量候选库 item

如果只把 geo 放进 `past_payloads`，训练时历史序列和候选 item 的语义空间会不一致，评估和召回也无法自然共享这套特征。因此 geo 的主接入点应该是 item embedding module，而不是 payload。

## Data Changes

### 1. Add item geo side-info file

建议新增一份 item 地理特征表，例如：

```text
tmp/processed/<dataset>/item_geo_features.csv
```

最小字段：

```text
item_id, latitude, longitude, geo_region_id, geo_cell_l5, geo_cell_l7
```

建议：

- `geo_region_id` 是较稳定的区域桶
- `geo_cell_l5` 是 coarse geohash/H3
- `geo_cell_l7` 是 finer geohash/H3
- 缺失值统一映射到保留桶 `0`

### 2. Keep external ids, normalize model ids

建议区分两套 id：

- 外部数据 id：保留原始值，便于回溯和数据治理。
- 训练模型 id：为适配 padding 规则，统一执行 `model_item_id = raw_item_id + 1`。

对应要求：

- 序列数据中的 `sequence_item_ids` 使用 model id。
- `item_geo_features.csv` 中的 `item_id` 也使用 model id。
- `0` 仅保留为 padding/unknown。

这样既不破坏原始数据可追溯性，也保证召回模型的 id 语义正确。

## Code Changes

### A. Preprocessing layer

目标文件：

- `generative_recommenders/research/data/preprocessor.py`

建议增加：

- item geo side-info 读取与清洗逻辑
- 输出规范化后的 `item_geo_features.csv`
- 对用户序列样本做 schema 归一化（列名、时间、ratings、id 偏移）

这里不一定要把 geo 特征展开到 `sasrec_format.csv` 中。对召回链路来说，更干净的做法是单独维护 `item_id -> geo features` 映射文件。

针对你样例的预处理最小动作：

1. 将 `UserId` 重命名为 `user_id`。
2. 将 `sequence_UTCTimeOffset` 解析为 `sequence_timestamps`（int64 Unix 秒）。
3. 自动补 `sequence_ratings`，长度与 `sequence_item_ids` 一致，默认值 1。
4. 将 `sequence_item_ids` 全量 `+1`。
5. geo 文件中的 `item_id` 同步 `+1`。

### B. Dataset metadata layer

目标文件：

- `generative_recommenders/research/data/reco_dataset.py`

当前 `RecoDataset` 返回：

- `max_sequence_length`
- `num_unique_items`
- `max_item_id`
- `all_item_ids`
- `train_dataset`
- `eval_dataset`

建议扩展为同时返回 geo side tensors，例如：

- `item_geo_region_ids: torch.Tensor`
- `item_geo_cell_l5_ids: torch.Tensor`
- `item_geo_cell_l7_ids: torch.Tensor`

形状建议均为：

```text
[max_item_id + 1]
```

使其可以直接按 `item_ids` 索引。

额外校验建议：

- 断言 `min(all_item_ids) >= 1`。
- 断言 geo 映射张量在 `0..max_item_id` 范围内可索引。
- 断言 `sequence_item_ids` 与 `sequence_timestamps`、`sequence_ratings` 长度一致。

### C. Embedding module layer

目标文件：

- `generative_recommenders/research/modeling/sequential/embedding_modules.py`

建议新增：

- `GeoAwareEmbeddingModule`

建议接口：

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

核心逻辑：

1. 查 item id embedding
2. 通过 `item_ids` 索引 geo id buffer
3. 查 geo embeddings
4. 做 concat 或 add fusion
5. 输出仍为 `(B, N, item_embedding_dim)`

推荐实现：

- `item_id_emb`: 维持原维度
- geo 三路 embedding 维度可以较小，例如 `16` 或 `32`
- geo concat 后通过一层线性投影到 `item_embedding_dim`
- 最后与 `item_id_emb` 相加

即：

```text
final_item_emb = item_id_emb + geo_proj(concat(region_emb, l5_emb, l7_emb))
```

这种做法兼容当前所有 encoder，不需要改 SASRec/HSTU 主体结构。

针对样例数据，建议增加一个可选的 unknown geo bucket 处理：

- 当 item 没有 geo 记录时，region/l5/l7 全部回落到 0 桶。

### D. Training entry

目标文件：

- `generative_recommenders/research/trainer/train.py`

当前只支持：

- `embedding_module_type == "local"`

建议改为支持：

- `local`
- `geo_aware`

做法：

1. `get_reco_dataset()` 返回 geo mapping tensors
2. `train_fn()` 根据 `embedding_module_type` 构建 `GeoAwareEmbeddingModule`
3. 其余训练逻辑尽量不动

### E. Negative sampler

目标文件：

- `generative_recommenders/research/modeling/sequential/autoregressive_losses.py`

这是召回部分里最容易漏掉的点。

当前 `LocalNegativesSampler` 直接依赖底层的 `torch.nn.Embedding`：

```python
self._item_emb(sampled_ids)
```

一旦 item embedding 不再是单纯的 `_item_emb`，这个 sampler 就不正确了。

因此需要改造 `LocalNegativesSampler`，不要再依赖裸 embedding table，而应依赖统一接口，例如：

```python
embedding_module.get_item_embeddings(sampled_ids)
```

推荐改法：

- 将 `LocalNegativesSampler` 的构造参数从 `item_emb` 改为 `embedding_module`
- 在 `forward()` 中调用 `embedding_module.get_item_embeddings(sampled_ids)`

这是 geo-aware 召回能否正确训练的必要改动。

### F. Evaluation

目标文件：

- `generative_recommenders/research/data/eval.py`

好消息是，这部分几乎不需要改逻辑。

因为全量评估候选库本来就是通过：

```python
model.get_item_embeddings(eval_negatives_ids)
```

构建的。

只要 `GeoAwareEmbeddingModule` 已接好，评估自然会使用 geo-aware item embeddings。

## Recommended Geo Feature Encoding

### First milestone

先只做离散地理桶：

- `geo_region_id`
- `geo_cell_l5`
- `geo_cell_l7`

不建议第一版直接输入原始 `latitude/longitude`。

原因：

- 当前召回链路天然适配离散 sparse features
- 原始浮点经纬度需要额外 dense feature path
- 全量候选索引阶段更适合提前离散后的静态 item embedding

### Suggested embedding sizes

- region embedding dim: `8` or `16`
- l5 embedding dim: `16`
- l7 embedding dim: `16` or `32`

总 geo 拼接维度先控制在 `32` 到 `64` 比较稳。

## Evaluation Recommendation

只看全局 HR/NDCG 不够。建议至少补这些 geo 切片指标：

- 同区域用户样本 HR@10 / NDCG@10
- 跨区域样本 HR@10 / NDCG@10
- 长尾区域 item 样本 HR@10 / NDCG@10
- 新 item 样本 HR@10 / NDCG@10

如果你们有用户位置或请求位置，再补：

- 近距离桶指标
- 远距离桶指标

否则很容易只学到“热门区域 item 更热门”，而不是“地理相关性更强”。

## What Not To Do First

第一版不建议：

- 修改 SASRec 或 HSTU 主体结构
- 把 geo 直接塞进 `past_payloads` 作为唯一入口
- 只改预处理文件，不改 embedding module
- 只改训练历史侧，不改负采样和全量评估

这些做法都容易导致训练和召回语义空间不一致。

## Minimal Viable Implementation

如果只做一版最小可行实现，建议只动这 4 个文件：

- `generative_recommenders/research/data/preprocessor.py`
- `generative_recommenders/research/data/reco_dataset.py`
- `generative_recommenders/research/modeling/sequential/embedding_modules.py`
- `generative_recommenders/research/modeling/sequential/autoregressive_losses.py`

以及一处训练入口：

- `generative_recommenders/research/trainer/train.py`

## Recommended First Patch

第一版建议实现的能力：

1. 加载 `item_geo_features.csv`
2. 构建 `GeoAwareEmbeddingModule`
3. 改 `train_fn()` 支持 `embedding_module_type = "geo_aware"`
4. 改 `LocalNegativesSampler` 调用 `embedding_module.get_item_embeddings()`
5. 复用现有 `eval.py` 做全量召回评估

在你的样例上，第一版需要再补两项前置：

6. 将 `sequence_UTCTimeOffset` 转换为整型 `sequence_timestamps`
7. 将业务 item id 与 geo item id 统一执行 `+1` 偏移

这样改完后，geo 特征会统一进入：

- 序列历史 item
- 正样本 item
- 负样本 item
- 全量候选库 item

这才是召回阶段正确的一致性接入。

## File-By-File Execution Checklist

本节用于直接执行开发，按文件和函数拆分为可交付任务。

### 1) `generative_recommenders/research/data/preprocessor.py`

#### Task P1: 增加样例 schema 归一化

改动点：

- 在数据读取后增加列名兼容处理：`UserId -> user_id`。
- 增加 `sequence_UTCTimeOffset -> sequence_timestamps` 转换。
- 时间字符串列表统一转换为 Unix 秒级整型列表。
- 当 `sequence_ratings` 缺失时自动补全为同长度全 1 列表。

验收：

- 预处理输出中存在 `user_id/sequence_item_ids/sequence_ratings/sequence_timestamps` 四列。
- 四列均可被 `DatasetV2.load_item()` 正常解析。

#### Task P2: 增加 item id 偏移与 geo 映射对齐

改动点：

- 对序列中的业务 item id 执行 `+1` 偏移。
- 对 geo side-info 中的 `item_id` 同步执行 `+1` 偏移。
- 保留 `0` 作为 padding/unknown。

验收：

- 输出序列内 item id 最小值 `>= 1`。
- geo 映射文件中 item id 最小值 `>= 1`。

#### Task P3: 产出 geo 侧信息文件

改动点：

- 新增导出：`tmp/processed/<dataset>/item_geo_features.csv`。
- 至少包含：`item_id, geo_region_id, geo_cell_l5, geo_cell_l7`。

验收：

- 文件存在且可被 pandas 正常读取。
- 缺失 geo item 使用 0 桶。

### 2) `generative_recommenders/research/data/reco_dataset.py`

#### Task R1: 扩展 `RecoDataset` 数据结构

改动点：

- 在 `RecoDataset` dataclass 中新增：
    - `item_geo_region_ids`
    - `item_geo_cell_l5_ids`
    - `item_geo_cell_l7_ids`

验收：

- `get_reco_dataset()` 返回对象含以上字段。

#### Task R2: 在 `get_reco_dataset()` 加载 geo 映射

改动点：

- 读取 `item_geo_features.csv`。
- 构造 3 个 shape 为 `[max_item_id + 1]` 的 LongTensor。
- 对不存在映射的 item 填 0。

验收：

- `tensor[item_id]` 可直接索引，无越界。
- `all_item_ids` 全量可映射到 geo tensor。

#### Task R3: 增加一致性断言

改动点：

- 断言 `min(all_item_ids) >= 1`。
- 断言 geo tensor 长度为 `max_item_id + 1`。

验收：

- 数据加载时提前失败而非训练阶段崩溃。

### 3) `generative_recommenders/research/modeling/sequential/embedding_modules.py`

#### Task E1: 新增 `GeoAwareEmbeddingModule`

改动点：

- 继承 `EmbeddingModule`。
- 构造函数接收 geo 映射 tensor 与 geo embedding 表大小参数。
- 注册 geo 映射为 buffer。

验收：

- 模块可被实例化并迁移到 GPU。

#### Task E2: 实现 `get_item_embeddings()` 融合逻辑

改动点：

- 查 item id embedding。
- 按 item_ids 查 geo ids，再查 geo embedding。
- 采用 `item_id_emb + geo_proj(concat(...))` 输出。

验收：

- 输入 `[B, N]`，输出 `[B, N, D]`。
- 输出全 finite，且 padding id=0 可返回稳定值。

#### Task E3: 实现 `debug_str()`

改动点：

- 提供可区分的 debug 字符串，方便日志识别。

验收：

- 训练日志中可看到 geo-aware embedding 类型。

### 4) `generative_recommenders/research/modeling/sequential/autoregressive_losses.py`

#### Task N1: 改造 `LocalNegativesSampler`

改动点：

- 构造参数从 `item_emb` 改为 `embedding_module`。
- `forward()` 中通过 `embedding_module.get_item_embeddings(sampled_ids)` 取向量。

验收：

- `sampling_strategy=local` 时负样本 embedding 与主模型 item embedding 语义一致。

#### Task N2: 保持归一化路径不变

改动点：

- `normalize_embeddings()` 保持原逻辑。

验收：

- 旧配置（local embedding）指标回归不明显劣化。

### 5) `generative_recommenders/research/trainer/train.py`

#### Task T1: 扩展 `embedding_module_type`

改动点：

- 支持 `local` 和 `geo_aware`。
- 新增 geo 配置参数：
    - `geo_embedding_dim`
    - `num_geo_regions`
    - `num_geo_cells_l5`
    - `num_geo_cells_l7`

验收：

- 两种 embedding 模式都可单独启动训练。

#### Task T2: 接入 `GeoAwareEmbeddingModule` 初始化

改动点：

- 从 `dataset` 读取 geo tensors。
- 构建 `GeoAwareEmbeddingModule` 并传入模型。

验收：

- 首个 step 可前向、反向、优化器更新。

#### Task T3: 负采样器参数更新

改动点：

- `LocalNegativesSampler` 初始化改为传 `embedding_module`。

验收：

- 训练中 sampled negatives 不报维度/类型错误。

### 6) Gin Config

#### Task G1: 新增 geo-aware 召回配置

改动点：

- 基于现有配置复制一个 geo-aware 版本。
- 设置 `embedding_module_type = "geo_aware"` 与 geo 参数。

验收：

- 单命令可直接启动 geo-aware 训练。

## End-To-End Acceptance Gates

### Gate A: 数据可用性

- 归一化后序列文件可被 `DatasetV2` 读取。
- `item_id=0` 不再出现在业务样本中。

### Gate B: 训练可用性

- geo-aware 模式首个 epoch 无 NaN/Inf。
- local 模式不受影响。

### Gate C: 评估可用性

- 全量评估可跑通。
- HR/NDCG 指标可正常产出。

### Gate D: 对照结论

- 至少一组 baseline(local) vs geo_aware 对照结果。
- 给出 geo 切片分析结论，避免只看全局指标。

## Suggested Task Order

1. P1-P3（先保证输入数据完全可用）
2. R1-R3（把 geo mapping 串进 dataset 元信息）
3. E1-E3（实现 geo-aware item embedding）
4. N1-N2（确保 local negatives 与新 embedding 一致）
5. T1-T3（训练入口打通）
6. G1 + Gate A-D（跑实验并验收）

## Minimal Implementation Templates

本节给出“每个文件一段最小实现模板”。这些不是最终代码，但已经接近可直接翻译成实现。

### 1) `generative_recommenders/research/data/preprocessor.py`

目标：把原始序列 csv 和 geo csv 转成召回训练可消费的规范文件。

```python
def normalize_recall_sequence_schema(seq_df: pd.DataFrame) -> pd.DataFrame:
    if "UserId" in seq_df.columns:
        seq_df = seq_df.rename(columns={"UserId": "user_id"})

    if "sequence_UTCTimeOffset" in seq_df.columns:
        seq_df = seq_df.rename(
            columns={"sequence_UTCTimeOffset": "sequence_timestamps"}
        )

    seq_df["sequence_item_ids"] = seq_df["sequence_item_ids"].apply(eval)
    seq_df["sequence_timestamps"] = seq_df["sequence_timestamps"].apply(eval)

    if "sequence_ratings" not in seq_df.columns:
        seq_df["sequence_ratings"] = seq_df["sequence_item_ids"].apply(
            lambda seq: [1 for _ in seq]
        )
    else:
        seq_df["sequence_ratings"] = seq_df["sequence_ratings"].apply(eval)

    seq_df["sequence_timestamps"] = seq_df["sequence_timestamps"].apply(
        lambda seq: [int(pd.Timestamp(x).timestamp()) for x in seq]
    )

    for row in seq_df.itertuples():
        assert len(row.sequence_item_ids) == len(row.sequence_ratings)
        assert len(row.sequence_item_ids) == len(row.sequence_timestamps)

    # reserve 0 for padding
    seq_df["sequence_item_ids"] = seq_df["sequence_item_ids"].apply(
        lambda seq: [int(x) + 1 for x in seq]
    )

    return seq_df


def build_item_geo_features(geo_df: pd.DataFrame) -> pd.DataFrame:
    geo_df = geo_df.rename(columns={"item_id": "raw_item_id"})
    geo_df["item_id"] = geo_df["raw_item_id"].astype(int) + 1

    def _encode_cell(lat: float, lon: float, level: int):
        if pd.isna(lat) or pd.isna(lon):
            return "UNK"
        return geohash_encode(lat, lon, precision=level)

    geo_df["geo_cell_l5"] = geo_df.apply(
        lambda row: _encode_cell(row["Latitude"], row["Longitude"], 5),
        axis=1,
    )
    geo_df["geo_cell_l7"] = geo_df.apply(
        lambda row: _encode_cell(row["Latitude"], row["Longitude"], 7),
        axis=1,
    )

    # fallback region = coarse cell if no reverse-geocode system is available
    geo_df["geo_region_id"] = geo_df["geo_cell_l5"]

    for col in ["geo_region_id", "geo_cell_l5", "geo_cell_l7"]:
        geo_df[col] = pd.Categorical(geo_df[col]).codes + 1
        geo_df.loc[geo_df[col] < 1, col] = 0

    return geo_df[["item_id", "geo_region_id", "geo_cell_l5", "geo_cell_l7"]]


def write_recall_artifacts(sequence_csv: str, geo_csv: str, output_dir: str) -> None:
    seq_df = pd.read_csv(sequence_csv)
    geo_df = pd.read_csv(geo_csv)

    seq_df = normalize_recall_sequence_schema(seq_df)
    geo_features_df = build_item_geo_features(geo_df)

    seq_df.to_csv(f"{output_dir}/sasrec_format.csv", index=False)
    geo_features_df.to_csv(f"{output_dir}/item_geo_features.csv", index=False)
```

### 2) `generative_recommenders/research/data/reco_dataset.py`

目标：把 geo 文件转成按 item_id 直接索引的 tensor。

```python
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


def _build_geo_feature_tensors(item_geo_csv: str, max_item_id: int):
    geo_df = pd.read_csv(item_geo_csv)

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

    return item_geo_region_ids, item_geo_cell_l5_ids, item_geo_cell_l7_ids


def get_reco_dataset(...):
    # existing logic
    ...

    item_geo_csv = f"tmp/processed/{dataset_name}/item_geo_features.csv"
    item_geo_region_ids, item_geo_cell_l5_ids, item_geo_cell_l7_ids = (
        _build_geo_feature_tensors(item_geo_csv, max_item_id)
    )

    assert min(all_item_ids) >= 1

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=...,
        max_item_id=max_item_id,
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        item_geo_region_ids=item_geo_region_ids,
        item_geo_cell_l5_ids=item_geo_cell_l5_ids,
        item_geo_cell_l7_ids=item_geo_cell_l7_ids,
    )
```

### 3) `generative_recommenders/research/modeling/sequential/embedding_modules.py`

目标：新增 geo-aware item embedding。

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
        super().__init__()
        self._item_embedding_dim = item_embedding_dim

        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self._region_emb = torch.nn.Embedding(
            num_geo_regions + 1, geo_embedding_dim, padding_idx=0
        )
        self._cell_l5_emb = torch.nn.Embedding(
            num_geo_cells_l5 + 1, geo_embedding_dim, padding_idx=0
        )
        self._cell_l7_emb = torch.nn.Embedding(
            num_geo_cells_l7 + 1, geo_embedding_dim, padding_idx=0
        )

        self.register_buffer("_item_geo_region_ids", item_geo_region_ids)
        self.register_buffer("_item_geo_cell_l5_ids", item_geo_cell_l5_ids)
        self.register_buffer("_item_geo_cell_l7_ids", item_geo_cell_l7_ids)

        self._geo_proj = torch.nn.Sequential(
            torch.nn.Linear(geo_embedding_dim * 3, item_embedding_dim),
            torch.nn.LayerNorm(item_embedding_dim),
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"geo_aware_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        truncated_normal(self._item_emb.weight, mean=0.0, std=0.02)
        truncated_normal(self._region_emb.weight, mean=0.0, std=0.02)
        truncated_normal(self._cell_l5_emb.weight, mean=0.0, std=0.02)
        truncated_normal(self._cell_l7_emb.weight, mean=0.0, std=0.02)
        for module in self._geo_proj:
            if hasattr(module, "weight"):
                torch.nn.init.xavier_uniform_(module.weight)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_emb = self._item_emb(item_ids)

        region_ids = self._item_geo_region_ids[item_ids]
        cell_l5_ids = self._item_geo_cell_l5_ids[item_ids]
        cell_l7_ids = self._item_geo_cell_l7_ids[item_ids]

        region_emb = self._region_emb(region_ids)
        cell_l5_emb = self._cell_l5_emb(cell_l5_ids)
        cell_l7_emb = self._cell_l7_emb(cell_l7_ids)

        geo_emb = torch.cat([region_emb, cell_l5_emb, cell_l7_emb], dim=-1)
        return item_emb + self._geo_proj(geo_emb)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim
```

### 4) `generative_recommenders/research/modeling/sequential/autoregressive_losses.py`

目标：让 local negative sampling 也走 geo-aware embedding。

```python
class LocalNegativesSampler(NegativesSampler):
    def __init__(
        self,
        num_items: int,
        embedding_module: EmbeddingModule,
        all_item_ids: List[int],
        l2_norm: bool,
        l2_norm_eps: float,
    ) -> None:
        super().__init__(l2_norm=l2_norm, l2_norm_eps=l2_norm_eps)
        self._num_items = len(all_item_ids)
        self._embedding_module = embedding_module
        self.register_buffer("_all_item_ids", torch.tensor(all_item_ids))

    def forward(
        self,
        positive_ids: torch.Tensor,
        num_to_sample: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_shape = positive_ids.size() + (num_to_sample,)
        sampled_offsets = torch.randint(
            low=0,
            high=self._num_items,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
        sampled_embeddings = self._embedding_module.get_item_embeddings(sampled_ids)
        return sampled_ids, self.normalize_embeddings(sampled_embeddings)
```

### 5) `generative_recommenders/research/trainer/train.py`

目标：在训练入口切换 embedding module。

## FourierGeo Three Variants (Visualization)

下面汇总当前已落地并训练验证过的 `FourierGeo` 融合方案（含 New-A / New-B / budget-concat-residual）。

公共主干（所有方案共用）：

- item_id -> item_emb: D
- item_geo_fourier_features.csv (geo_fourier_0..127) -> Lookup 128-d
- item_visit_time_features.csv (visit_hour_0..23) -> Lookup 24-d

### 方案流程（文本流程图）

V1: Online Random Fourier（早期）

- lat/lon 运行时随机 Fourier 映射
- concat visit_time(24)
- Linear -> LayerNorm -> geo_delta(D)
- 输出: item_emb + geo_delta

V2: Offline Deterministic Fourier + Fixed Scale（早期稳态）

- 预计算 geo_fourier_0..127
- concat visit_time(24)
- Linear -> geo_delta(D)
- 输出: item_emb + 0.05 * geo_delta

V3: Offline Deterministic Fourier + Adaptive Gate（当前主线之一）

- 预计算 geo_fourier_0..127
- concat visit_time(24)
- Linear -> geo_delta(D)
- gate_input = concat(item_emb, geo_delta)
- gate = 0.2 * sigmoid(Linear)
- 输出: item_emb + gate * geo_delta

New-A: MLP(item)+MLP(geo)+MLP(visit_time) -> cat -> fusion（纯融合输出）

- item 分支: MLP(item_emb)
- geo 分支: MLP(geo_fourier)
- visit 分支: MLP(visit_time)
- cat 三分支后经 fusion MLP
- 输出: fused_emb（不保留 item 残差锚点）

New-B: New-A + item residual anchor（当前最优）

- 与 New-A 同分支结构
- 输出改为: item_emb + scale * fused_emb
- 当前对照实验中，`geo_fourier_concat_b` 在峰值指标上优于 New-A 和 budget-concat-residual

Budget-Concat-Residual: 预算维度 cat + residual（对照线）

- item 分支: MLP(item_emb) -> item_branch_dim
- geo 分支: MLP(geo_fourier) -> geo_branch_dim
- visit 分支: MLP(visit_time) -> visit_time_branch_dim
- 约束: item_branch_dim + geo_branch_dim + visit_time_branch_dim = D
- cat 三分支后直接对齐到 D（可加 LayerNorm）
- 输出: item_emb + scale * budget_fused_emb
- 典型配置示例（D=50）: 22/16/12 或 24/10/16
- 现阶段实验结论: 在当前数据上未超过 `geo_fourier_concat_b`，且部分配置未超过纯 item baseline

### Variant Summary

| Variant | Fusion form | Strength | Risk |
|---|---|---|---|
| V1 | `Linear -> LayerNorm -> add` | 表达强 | 训练不稳，易压过 item 主语义 |
| V2 | `Linear -> 0.05 * add` | 简单稳态 | 上限偏低 |
| V3 | `Linear -> gate -> add` | 自适应强 | 需监控 gate 漂移 |
| New-A | `MLP(3-branch) -> cat -> fusion` | 交互表达更强 | 无锚点，后程易过拟合 |
| New-B | `New-A + residual anchor` | 表达与稳定性平衡最好 | 仍需控制融合强度 |
| Budget-Concat-Residual | `MLP(3-branch)->budget cat->residual add` | 参数更可控，分支职责清晰 | 预算过强时易形成信息瓶颈 |

### Current Recommendation

1. 主线使用 `New-B`（`embedding_module_type = "geo_fourier_concat_b"`）。
2. 融合强度优先测试 `scale=0.10/0.08/0.06`；当前实验结论为 `0.10 > 0.08 > 0.06`。
3. 训练策略采用“峰值 checkpoint”而非最后 checkpoint：
   - 当前最佳点通常在 epoch 20 左右。
   - 30 epoch 后多方案进入平台并小范围震荡。
4. `New-A` 和 budget-concat-residual 作为对照线保留，不建议作为默认生产训练配置。

```python
@gin.configurable
def train_fn(
    ...,
    embedding_module_type: str = "local",
    geo_embedding_dim: int = 16,
    num_geo_regions: int = 4096,
    num_geo_cells_l5: int = 65536,
    num_geo_cells_l7: int = 262144,
):
    dataset = get_reco_dataset(...)

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
        raise ValueError(f"Unknown embedding_module_type {embedding_module_type}")

    model = get_sequential_encoder(
        ...,
        embedding_module=embedding_module,
        ...,
    )

    if sampling_strategy == "local":
        negatives_sampler = LocalNegativesSampler(
            num_items=dataset.max_item_id,
            embedding_module=embedding_module,
            all_item_ids=dataset.all_item_ids,
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
        )
```

### 6) Geo-aware gin 配置模板

目标：最小可运行实验配置。

```python
train_fn.dataset_name = "<your_dataset>"
train_fn.embedding_module_type = "geo_aware"
train_fn.item_embedding_dim = 64
train_fn.geo_embedding_dim = 16
train_fn.num_geo_regions = 4096
train_fn.num_geo_cells_l5 = 65536
train_fn.num_geo_cells_l7 = 262144

train_fn.main_module = "HSTU"
train_fn.loss_module = "SampledSoftmaxLoss"
train_fn.sampling_strategy = "local"
train_fn.num_negatives = 128
```

### 7) 最小联调脚本模板

目标：在正式训练前做冒烟验证。

```python
def smoke_test_geo_aware_pipeline():
    dataset = get_reco_dataset(...)
    embedding_module = GeoAwareEmbeddingModule(...)

    sample_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.int64)
    sample_embs = embedding_module.get_item_embeddings(sample_ids)

    assert sample_embs.shape[-1] > 0
    assert torch.isfinite(sample_embs).all()

    negatives_sampler = LocalNegativesSampler(
        num_items=dataset.max_item_id,
        RECALL_ITEM_GEO_ADAPTATION_PLAN.md
        
        如果你愿意，我可以再给这一段补一个“更紧凑版”（单屏宽度更友好）的文本流程图样式。
        
        
        embedding_module=embedding_module,
        all_item_ids=dataset.all_item_ids,
        l2_norm=True,
        l2_norm_eps=1e-6,
    )

    neg_ids, neg_embs = negatives_sampler(
        positive_ids=torch.tensor([[1, 2]]),
        num_to_sample=4,
    )

    assert neg_ids.shape[-1] == 4
    assert torch.isfinite(neg_embs).all()
```