# HSTU + Combined 伪补丁模板

说明：

- 这是函数级伪补丁，不会直接修改代码。
- 只覆盖 HSTU 路线，不包含 SASRec。
- 目标是最小可跑版本：combined 输入 + HSTU 编码 + item-only 监督。

---

## 文件 1: generative_recommenders/research/trainer/train.py

## 1) train_fn 参数

### Before

- 无 input_preprocessor_type
- 无 num_ratings

### After (伪补丁)

在 train_fn 参数列表新增：

- input_preprocessor_type: str = "learnable_positional"
- num_ratings: int = 6

---

## 2) 预处理器导入

### Before

from ...input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)

### After (伪补丁)

from ...input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
    CombinedItemAndRatingInputFeaturesPreprocessor,
)

---

## 3) input_preproc_module 构造

### Before

input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
    max_sequence_len=dataset.max_sequence_length + gr_output_length + 1,
    embedding_dim=item_embedding_dim,
    dropout_rate=dropout_rate,
)

### After (伪补丁)

if input_preprocessor_type == "learnable_positional":
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=dataset.max_sequence_length + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=dropout_rate,
    )
elif input_preprocessor_type == "combined_item_rating":
    input_preproc_module = CombinedItemAndRatingInputFeaturesPreprocessor(
        max_sequence_len=dataset.max_sequence_length + gr_output_length + 1,
        item_embedding_dim=item_embedding_dim,
        dropout_rate=dropout_rate,
        num_ratings=num_ratings,
    )
else:
    raise ValueError(...)

---

## 4) HSTU encoder 长度传参

### Before

model = get_sequential_encoder(
    ...
    max_sequence_length=dataset.max_sequence_length,
    max_output_length=gr_output_length + 1,
    ...
)

### After (伪补丁)

if input_preprocessor_type == "combined_item_rating":
    encoder_max_sequence_length = dataset.max_sequence_length * 2
    encoder_max_output_length = (gr_output_length + 1) * 2
else:
    encoder_max_sequence_length = dataset.max_sequence_length
    encoder_max_output_length = gr_output_length + 1

model = get_sequential_encoder(
    ...
    max_sequence_length=encoder_max_sequence_length,
    max_output_length=encoder_max_output_length,
    ...
)

---

## 5) 训练监督重排 (combined + HSTU)

### Before

supervision_ids = seq_features.past_ids
ar_mask = supervision_ids[:, 1:] != 0

loss, aux_losses = ar_loss(
    lengths=seq_features.past_lengths,
    output_embeddings=seq_embeddings[:, :-1, :],
    supervision_ids=supervision_ids[:, 1:],
    supervision_embeddings=input_embeddings[:, 1:, :],
    supervision_weights=ar_mask.float(),
    ...
)

### After (伪补丁)

if input_preprocessor_type == "combined_item_rating":
    combined_ids = input_preproc_module.get_preprocessed_ids(
        past_lengths=seq_features.past_lengths,
        past_ids=seq_features.past_ids,
        past_embeddings=input_embeddings,
        past_payloads=seq_features.past_payloads,
    )

    combined_mask = input_preproc_module.get_preprocessed_masks(
        past_lengths=seq_features.past_lengths,
        past_ids=seq_features.past_ids,
        past_embeddings=input_embeddings,
        past_payloads=seq_features.past_payloads,
    )

    # rating 位置作为 query: 1,3,5,...
    query_embeddings = seq_embeddings[:, 1:-1:2, :]

    # 下一个 item 作为 label: 2,4,6,...
    label_item_ids = combined_ids[:, 2::2]
    label_item_embeddings = model.module.get_item_embeddings(label_item_ids)

    # 有效监督: label item 非零（可再与 combined_mask 对齐）
    label_mask = (label_item_ids != 0)
    query_lengths = seq_features.past_lengths

    loss, aux_losses = ar_loss(
        lengths=query_lengths,
        output_embeddings=query_embeddings,
        supervision_ids=label_item_ids,
        supervision_embeddings=label_item_embeddings,
        supervision_weights=label_mask.float(),
        negatives_sampler=negatives_sampler,
        **seq_features.past_payloads,
    )
else:
    # 保留原来的 item-only shift 逻辑
    ...

---

## 6) in-batch negatives 缓存

### Before

in_batch_ids = supervision_ids.view(-1)
negatives_sampler.process_batch(
    ids=in_batch_ids,
    presences=(in_batch_ids != 0),
    embeddings=model.module.get_item_embeddings(in_batch_ids),
)

### After (伪补丁)

if input_preprocessor_type == "combined_item_rating":
    in_batch_ids = label_item_ids.reshape(-1)
else:
    in_batch_ids = supervision_ids.view(-1)

negatives_sampler.process_batch(
    ids=in_batch_ids,
    presences=(in_batch_ids != 0),
    embeddings=model.module.get_item_embeddings(in_batch_ids),
)

说明：

- `label_item_ids = combined_ids[:, 2::2]` 是带步长的切片，通常不是 contiguous tensor
- 因此这里不能安全使用 `.view(-1)`
- 最小修正就是改成 `.reshape(-1)`
- 如果你希望行为更显式，也可以写成 `label_item_ids.contiguous().view(-1)`

---

## 文件 2: generative_recommenders/research/modeling/sequential/input_features_preprocessors.py

## 7) 为 HSTU 增加 timestamps 展开辅助

### Before

- Combined 类没有 timestamps 展开函数

### After (伪补丁)

在 CombinedItemAndRatingInputFeaturesPreprocessor 类新增方法：

def get_preprocessed_timestamps(
    self,
    past_lengths,
    past_ids,
    past_embeddings,
    past_payloads,
):
    # 输入: [B, N]
    # 输出: [B, 2N], [ts0, ts0, ts1, ts1, ...]
    ts = past_payloads["timestamps"]
    return ts.unsqueeze(2).expand(-1, -1, 2).reshape(ts.size(0), ts.size(1) * 2)

可选再加：

def get_preprocessed_lengths(self, past_lengths):
    return past_lengths * 2

---

## 文件 3: generative_recommenders/research/modeling/sequential/hstu.py

## 8) generate_user_embeddings 中 timestamps 对齐

### Before

all_timestamps=(
    past_payloads[TIMESTAMPS_KEY]
    if TIMESTAMPS_KEY in past_payloads
    else None
)

### After (伪补丁)

all_timestamps = None
if TIMESTAMPS_KEY in past_payloads:
    if hasattr(self._input_features_preproc, "get_preprocessed_timestamps"):
        all_timestamps = self._input_features_preproc.get_preprocessed_timestamps(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
    else:
        all_timestamps = past_payloads[TIMESTAMPS_KEY]

user_embeddings, cached_states = self._hstu(
    ...
    all_timestamps=all_timestamps,
    ...
)

---

## 9) encode 中 current state 位置

### Before

current_embeddings = get_current_embeddings(
    lengths=past_lengths,
    encoded_embeddings=encoded_seq_embeddings,
)

### After (伪补丁)

current_lengths = past_lengths
if hasattr(self._input_features_preproc, "get_preprocessed_lengths"):
    current_lengths = self._input_features_preproc.get_preprocessed_lengths(
        past_lengths
    )
elif self._input_features_preproc.__class__.__name__ == "CombinedItemAndRatingInputFeaturesPreprocessor":
    current_lengths = past_lengths * 2

current_embeddings = get_current_embeddings(
    lengths=current_lengths,
    encoded_embeddings=encoded_seq_embeddings,
)

---

## 10) relative bias 长度语义检查点

这里通常不改 hstu.py 的代码结构，但要保证：

- 构造 HSTU 时传入的 max_sequence_len / max_output_len 已经是 token 级长度

否则下面两处会错位：

- _attn_mask 大小
- RelativeBucketedTimeAndPositionBasedBias 的 max_seq_len

---

## 文件 4: configs/*.gin

## 11) 配置最小新增

在 HSTU 实验配置中新增：

train_fn.input_preprocessor_type = "combined_item_rating"
train_fn.num_ratings = 6

---

## 最小联调顺序

1. 先改 train.py 的 preprocessor 选择 + encoder token 长度
2. 再改 input_features_preprocessors.py 的 timestamps 展开辅助
3. 再改 hstu.py 的 timestamps 接入 + encode current_lengths
4. 最后改 train.py 的 combined supervision 对齐
5. 验证损失能跑通后再看评估和 cache 路径

---

## 快速自检清单

1. HSTU forward 无 shape mismatch
2. all_timestamps 长度与 token 序列长度一致
3. encode 取到最后有效 combined token
4. supervision_ids 只含 item id
5. in-batch negatives 未混入 rating token id

---

## 12) 训练超参调整建议

切到 `CombinedItemAndRatingInputFeaturesPreprocessor` 且使用 HSTU 后，即使模型没有明显崩溃、loss 也能正常下降，原本用于 `learnable_positional` 的训练超参也不应默认视为最优。

原因不是实现细节，而是训练问题本身已经变了：

- 输入序列长度从 `L` 变成了 `2L`
- token 数量增加，注意力和 relative bias 的优化难度上升
- query / label 对齐方式也从 item-only shift 变成了 rating token 预测下一个 item

因此超参通常需要重新扫描。

### 12.1 最优先调整：学习率

#### 建议

- 第一优先尝试把学习率调低，而不是调高
- 可以先从原 baseline 的 `0.5x ~ 0.8x` 开始试

#### 示例

- 如果原来 `learning_rate = 1e-3`
- 可先尝试 `5e-4` 或 `8e-4`

#### 原因

- combined + HSTU 的优化更敏感
- 序列更长后，前期梯度波动通常会更明显

### 12.2 强烈建议重新打开或增加 warmup

#### 建议

- 如果原配置 `num_warmup_steps = 0`，建议至少尝试非零 warmup
- 一个稳妥起点可以是总训练 step 的 `1% ~ 5%`

#### 原因

- HSTU 在 combined 模式下前期更容易抖动
- warmup 往往比直接增大 epoch 更先带来稳定收益

### 12.3 epoch / 总训练步数可能需要增加

#### 建议

- 不必一开始就把 epoch 大幅翻倍
- 先观察训练曲线和验证指标是否收敛更慢
- 如果只是收敛变慢，而不是过拟合或训练异常，再增加 epoch 或总 step

#### 原因

- combined 输入通常会让模型在同样 epoch 数下更晚达到最佳点

### 12.4 dropout 建议重新扫描

#### 建议

- 以原值为中心做小范围扫描
- 例如原来 `dropout_rate = 0.2`，可试 `0.1 / 0.2 / 0.3`

#### 原因

- token 变多后，模型接收到的上下文复杂度更高
- 最优正则强度不一定与 baseline 相同

### 12.5 batch size 改变会联动学习率

#### 建议

- 如果因为显存压力降低了 batch size，不要沿用完全相同的学习率
- 需要把 batch size 和学习率一起看

#### 原因

- combined 序列更长，显存占用通常上升
- 一旦 batch size 被迫减小，训练动态通常也会变化

### 12.6 推荐的最小调参顺序

建议按下面顺序试，而不是一次改很多项：

1. 先只切换到 combined，其他参数不动，拿到第一版对照结果
2. 在此基础上先调低学习率
3. 再加 warmup
4. 若训练正常但收敛偏慢，再增加 epoch 或总 step
5. 最后再扫描 dropout

### 12.7 更合理的目标预期

第一版 combined + HSTU 的目标不应该直接设成“必然超过 learnable_positional baseline”，更现实的目标是：

1. 先跑通且训练稳定
2. 指标接近 baseline
3. 再通过调参和对齐细化尝试超过 baseline

### 12.8 一句话建议

如果只允许先动两个训练超参，优先顺序建议是：

1. `learning_rate`
2. `num_warmup_steps`

### 12.9 基于 ml-1m HSTU baseline 的实验组

下面这些实验组都以 [configs/ml-1m/hstu-sampled-softmax-n128-final.gin](configs/ml-1m/hstu-sampled-softmax-n128-final.gin) 为基线，只列需要修改的项和值。

#### 实验 A：最小替换对照组

用途：先看切到 combined 后，不调训练超参时的直接结果。

需要修改：

- `train_fn.input_preprocessor_type = "combined_item_rating"`
- `train_fn.num_ratings = 6`

#### 实验 B：保守稳定组

用途：优先验证“降学习率 + 加 warmup”是否能提升稳定性。

需要修改：

- `train_fn.input_preprocessor_type = "combined_item_rating"`
- `train_fn.num_ratings = 6`
- `train_fn.learning_rate = 5e-4`
- `train_fn.num_warmup_steps = 2000`

#### 实验 C：中等保守组

用途：验证学习率不降太多时，是否能获得更快收敛。

需要修改：

- `train_fn.input_preprocessor_type = "combined_item_rating"`
- `train_fn.num_ratings = 6`
- `train_fn.learning_rate = 8e-4`
- `train_fn.num_warmup_steps = 2000`

#### 实验 D：长训组

用途：判断指标偏低是不是因为收敛更慢，而不是方向错误。

需要修改：

- `train_fn.input_preprocessor_type = "combined_item_rating"`
- `train_fn.num_ratings = 6`
- `train_fn.learning_rate = 5e-4`
- `train_fn.num_warmup_steps = 2000`
- `train_fn.num_epochs = 150`

#### 实验 E：更强正则组

用途：如果训练 loss 下降快但验证指标不跟，检查是否需要更强正则。

需要修改：

- `train_fn.input_preprocessor_type = "combined_item_rating"`
- `train_fn.num_ratings = 6`
- `train_fn.learning_rate = 5e-4`
- `train_fn.num_warmup_steps = 2000`
- `train_fn.dropout_rate = 0.3`
- `hstu_encoder.linear_dropout_rate = 0.3`

#### 实验 F：更弱正则组

用途：如果训练和验证都偏慢，检查是否正则过强。

需要修改：

- `train_fn.input_preprocessor_type = "combined_item_rating"`
- `train_fn.num_ratings = 6`
- `train_fn.learning_rate = 5e-4`
- `train_fn.num_warmup_steps = 2000`
- `train_fn.dropout_rate = 0.1`
- `hstu_encoder.linear_dropout_rate = 0.1`

#### 推荐优先顺序

如果不想一次跑太多组，建议先跑：

1. 实验 A
2. 实验 B
3. 实验 D