# HSTU + CombinedItemAndRatingInputFeaturesPreprocessor 适配清单

本文档只覆盖 HSTU 方案，不包含 SASRec。

目标是：

- 输入使用 `CombinedItemAndRatingInputFeaturesPreprocessor`
- token 序列为 `[item, rating, item, rating, ...]`
- HSTU 编码与相对时序偏置在 token 级长度上保持一致
- 训练目标仍可先保持为“预测下一个 item”

## 必须先固定的语义

在 combined 模式下，先固定以下规则：

1. 原始有效长度 `L` -> 预处理后有效长度 `2L`
2. 当前状态位置使用 `2L - 1`
3. 时间戳也按 token 级展开：`[ts0, ts1, ...] -> [ts0, ts0, ts1, ts1, ...]`
4. 第一阶段仍只预测 item，不把 rating token 并入候选集合

---

## 1) generative_recommenders/research/trainer/train.py

### 1.1 预处理器选择

#### 现在

- 默认固定实例化 `LearnablePositionalEmbeddingInputFeaturesPreprocessor`

#### 改成

- 增加 `input_preprocessor_type` 参数
- 增加 `num_ratings` 参数
- 当 `input_preprocessor_type == "combined_item_rating"` 时实例化 `CombinedItemAndRatingInputFeaturesPreprocessor`

### 1.2 HSTU encoder 长度入参

#### 现在

- `get_sequential_encoder(...)` 的 `max_sequence_length` / `max_output_length` 以原始 item 长度传入

#### 改成

- 传入 token 级长度
- combined 模式建议：
  - `encoder_max_sequence_length = dataset.max_sequence_length * 2`
  - `encoder_max_output_length = (gr_output_length + 1) * 2`

原因：HSTU 的注意力 mask 和 relative bias 都依赖长度参数，必须和 combined 输出的 token 长度一致。

### 1.3 训练 supervision 对齐（HSTU 同样需要）

#### 现在

- 训练沿用 item-only shift 对齐

#### 改成

- 使用 combined token 对齐
- query 取 rating token 位置
- label 取后续 item token 位置
- negatives sampler 仍只缓存 item label

说明：这部分和 SASRec 的 supervision 重排逻辑一致，区别只在 encoder 主干是 HSTU。

---

## 2) generative_recommenders/research/modeling/sequential/hstu.py

这是 HSTU 适配的核心文件。

### 2.1 `HSTU.__init__()` 中序列长度语义

#### 现在

- `self._max_sequence_length` 由构造参数直接赋值
- `_attn_mask` 大小由 `self._max_sequence_length + max_output_len` 决定

#### 改成

- 保持代码结构不变，但确保从 `train.py` 传入的 `max_sequence_len` / `max_output_len` 已是 token 级长度
- 即 `_attn_mask` 要覆盖 combined 后的真实 token 长度

### 2.2 Relative attention bias 长度

#### 现在

- `RelativeBucketedTimeAndPositionBasedBias(max_seq_len=max_sequence_len + max_output_len, ...)`

#### 改成

- 这里的 `max_seq_len` 必须使用 token 级长度（combined 后长度）

否则会出现：

- bias 张量长度与 token 序列不一致
- 位置偏置和时间偏置索引错位

### 2.3 `generate_user_embeddings()` 的 `all_timestamps`

#### 现在

- 直接把 `past_payloads["timestamps"]` 传入 `_hstu(...)`

#### 改成

- combined 模式下传入“展开后的 timestamps”
- 语义：每个 item/rating token 对应同一个事件时间

推荐最小逻辑：

- 若 preprocessor 为 combined，则先将 `timestamps` 扩成 `[ts0, ts0, ts1, ts1, ...]`
- 再传给 `_hstu(...)`

### 2.4 `encode()` 取当前状态位置

#### 现在

- `get_current_embeddings(lengths=past_lengths, encoded_embeddings=...)`

#### 改成

- combined 模式下改用 `current_lengths = past_lengths * 2`
- 再传入 `get_current_embeddings(...)`

这是 HSTU 下最容易漏改的点，漏改后不会立刻报错，但 query 状态会取错。

### 2.5 cache / delta 路径

#### 现在

- `delta_x_offsets` 与 cache 路径默认按既有长度语义运行

#### 建议

- 第一阶段先不改 cache 路径
- 先保证常规训练和全量 encode 跑通
- 若后续需要增量推理，再按 token 级长度重审 `delta_x_offsets` 对齐

---

## 3) generative_recommenders/research/modeling/sequential/input_features_preprocessors.py

### 3.1 `CombinedItemAndRatingInputFeaturesPreprocessor` 现状

已经具备：

- `forward()` 返回 `past_lengths * 2`
- `get_preprocessed_ids()`
- `get_preprocessed_masks()`

### 3.2 HSTU 需要新增的最小能力

建议增加一个方法（或等价逻辑），用于 timestamps 展开：

- `get_preprocessed_timestamps(...)`

语义：

- 输入 `[B, N]` timestamps
- 输出 `[B, 2N]` timestamps
- 每个时间点复制两次

### 3.3 可选增强

建议增加：

- `get_preprocessed_lengths(past_lengths)`

用于统一输出 `past_lengths * 2`，避免 HSTU 与训练入口散落写 `* 2`。

---

## 4) generative_recommenders/research/modeling/sequential/encoder_utils.py

### 4.1 是否必须改

- 通常不需要改函数结构
- 重点是调用方传入的长度语义

### 4.2 需保证

- `hstu_encoder(...)` 收到的 `max_sequence_length` / `max_output_length` 已是 token 级长度

---

## 5) generative_recommenders/research/data/eval.py

### 5.1 主评估链路

- 主要依赖 `model.encode()`
- 只要 HSTU `encode()` 已按 `past_lengths * 2` 取最后状态，主链路通常可复用

### 5.2 需关注

- 某些辅助评估函数默认把 `past_ids[:, -1]` 当 target
- combined 模式下该假设可能不稳妥，若实际用到再单独修

---

## 6) HSTU 最小实现顺序

建议严格按下面顺序推进：

1. `train.py` 支持 combined preprocessor + `num_ratings`
2. `train.py` 把 HSTU encoder 长度改成 token 级
3. `input_features_preprocessors.py` 增加 timestamps 展开能力
4. `hstu.py` 在 `generate_user_embeddings()` 里接入展开后的 timestamps
5. `hstu.py` 在 `encode()` 用 `past_lengths * 2` 取当前状态
6. `train.py` 重排 supervision（rating 位置预测下一 item）
7. 最后再看是否需要补 cache/delta 路径

---

## 7) HSTU 最小验收标准

达到以下条件可认为 HSTU 侧已切通：

1. HSTU forward 不再出现长度相关 shape 错误
2. relative bias 与 token 序列长度一致
3. timestamps 与 token 序列长度一致
4. `model.encode()` 返回的是最后有效 combined token 状态
5. 训练 supervision 只包含 item label
6. negatives sampler 未混入 rating token id
7. 评估 top-k 在 item 候选上可正常运行

---

## 8) 第一阶段不要做的事

为控制风险，HSTU 第一阶段不建议同时做：

1. 不要把 rating token 加入 sampled softmax 候选
2. 不要同时改 BCE 与 SampledSoftmax 两条分支
3. 不要先改 cache / delta 推理路径
4. 不要同时抽象大规模统一 preprocessor 接口

先完成最小闭环，再做扩展。