# CombinedItemAndRatingInputFeaturesPreprocessor 适配清单

本文档只列出从默认的 `LearnablePositionalEmbeddingInputFeaturesPreprocessor` 切换到 `CombinedItemAndRatingInputFeaturesPreprocessor` 时需要修改的点，不直接修改代码。

## 目标范围

建议先实现最小闭环：

- 输入序列改为 `[item, rating, item, rating, ...]`
- 编码器联合建模 item 和 rating token
- 训练目标仍然只预测下一个 item
- loss 只对 item label 计算，不把 rating 当候选集合的一部分
- 先跑通 SASRec，再扩展到 HSTU

## 关键语义变化

切换到 `CombinedItemAndRatingInputFeaturesPreprocessor` 后，下面几个前提全部改变：

- 原来一个时间步只对应一个 item token
- 现在一个原始位置会展开成两个 token：`item_i` 和 `rating_i`
- 原始有效长度 `L` 会变成预处理后的有效长度 `2L`
- 编码输出的最后有效位置不再是 `L - 1`，而是 `2L - 1`
- 训练 supervision 不能继续直接按原始 item 序列做 `shift-one`

## 按文件拆分的修改清单

### 1. generative_recommenders/research/trainer/train.py

#### 必改

- 在 `train_fn()` 增加预处理器配置项。
- 增加类似 `input_preprocessor_type` 的参数，用于选择默认预处理器。
- 增加 `num_ratings` 参数，因为 `CombinedItemAndRatingInputFeaturesPreprocessor` 构造函数需要它。
- 在导入列表中加入 `CombinedItemAndRatingInputFeaturesPreprocessor`。
- 把当前硬编码的 `LearnablePositionalEmbeddingInputFeaturesPreprocessor(...)` 改成按类型分支实例化。

#### 具体位置

- 默认预处理器导入：`generative_recommenders/research/trainer/train.py` 中导入 `input_features_preprocessors` 的位置。
- 默认实例化点：`input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(...)`

#### 必须同步调整的逻辑

- 构建 encoder 时，传入的 `max_sequence_length` 不能继续表示原始 item 序列长度，而要表示预处理后的 token 长度。
- 如果原始长度为 `raw_max_len`，切到 combined 后建议 encoder 看到的最大长度为 `2 * raw_max_len`。
- `max_output_length` 也要遵循同样的 token 级别规则，否则注意力 mask 会不够长。

#### 训练 supervision 需要重写

当前训练逻辑默认是：

- 把 `target_ids` 填到 `past_ids[past_lengths]`
- 用 `seq_embeddings[:, :-1, :]` 预测 `past_ids[:, 1:]`

这套逻辑在 combined 模式下不再成立，因为：

- 编码输出长度会变成 `2N`
- 原始 `past_ids` 长度仍是 `N`
- rating token 没法直接作为 item label 参与现有 sampled softmax

#### 最小可行改法

训练时固定成下面这条路线：

- 输入 token 序列：`[item_0, rating_0, item_1, rating_1, ...]`
- query 位置：使用 rating token 的输出位置
- label：该 rating token 后面的下一个 item token
- loss：只对 item label 算 sampled softmax / BCE
- negatives sampler：仍然只采样 item

#### 对应实现含义

- `supervision_ids` 不能再直接取 `seq_features.past_ids`
- 要基于 combined 之后的 token 序列重新构造 supervision
- 只抽取“rating 后预测 item”的那些位置作为有效监督样本

#### in-batch negatives 注意点

- 不能把 rating token id 混入 `negatives_sampler.process_batch()`
- 缓存负样本时只能使用 item label id
- `get_item_embeddings()` 仍然只能喂 item id

### 2. generative_recommenders/research/modeling/sequential/input_features_preprocessors.py

#### 当前已有能力

`CombinedItemAndRatingInputFeaturesPreprocessor` 已经提供了：

- `forward()`：返回 `past_lengths * 2`
- `get_preprocessed_ids()`：把原始 `(item, rating)` 交错展开成 token id 序列
- `get_preprocessed_masks()`：生成展开后的有效 mask

#### 建议补充的能力

为了减少下游散落的特殊判断，建议在这个文件里补充以下辅助约定或方法：

- 预处理后的“当前有效位置”计算规则
- 如有需要，补充 `get_preprocessed_timestamps()`，用于 HSTU
- 如有需要，补充一个明确返回“预处理后长度”的辅助接口

#### 需要明确的语义

必须固定下面这条规则，否则训练、评估和 encode 取状态时会错位：

- 原始有效长度：`L`
- 预处理后有效长度：`2L`
- 最后有效 token 位置：`2L - 1`

#### target rating 的处理策略需要定死

训练前通常会把目标 item 注入到历史序列的尾部。切到 combined 后要明确：

- 是否同时注入目标 rating
- 或者目标 item 后的 rating 位置保持为 0 占位

建议最小闭环采用：

- 注入 `target_item`
- 不注入未来真实 `target_rating`
- 对应 rating 位置保持 0，占位即可

这样不会把未来标签泄漏进上下文。

### 3. generative_recommenders/research/modeling/sequential/sasrec.py

#### 必改

- `_attn_mask` 的长度必须覆盖预处理后的 token 长度，而不是原始 item 长度。
- `encode()` 里取最后状态的逻辑必须改成基于 combined 后长度。

#### 原因

当前 SASRec 假设：

- 输入长度就是原始 item 序列长度
- `get_current_embeddings(lengths=past_lengths, ...)` 取到的就是最后有效状态

combined 模式下这两个假设都不成立。

#### 最小改法

- 构造 SASRec 时让 `max_sequence_len + max_output_len` 表示 token 级长度
- `encode()` 中对 combined 预处理器使用预处理后的长度取当前 embedding

#### 不建议的做法

- 不建议只在 preprocessor 里把长度翻倍，而保持 SASRec mask 长度不变
- 不建议继续按原始 `past_lengths` 从 `encoded_embeddings` 取最后状态

### 4. generative_recommenders/research/modeling/sequential/utils.py

#### 影响点

- `get_current_embeddings()` 当前完全依赖传入的 `lengths`

#### 修改思路

二选一：

- 保持这个工具函数不动，在 `encode()` 调用前传入预处理后的长度
- 或者扩展一个新的 helper，用于 combined 模式下取最后有效 token

#### 推荐

- 优先保持工具函数通用不变
- 在 `SASRec.encode()` / `HSTU.encode()` 里根据 preprocessor 类型传入正确长度

### 5. generative_recommenders/research/data/eval.py

#### 核心影响

- `eval_metrics_v2_from_tensors()` 依赖 `model.encode()` 输出当前 query embedding
- 只要 `encode()` 取最后状态正确，这部分通常不需要大改

#### 需要检查的点

- `eval_recall_metrics_from_tensors()` 中有一段逻辑默认把 `past_ids[:, -1]` 当 target
- 如果这个函数会被实际使用，combined 模式下这条假设不再稳妥

#### 建议

- 主评估链路优先复用现有实现
- 对 `eval_recall_metrics_from_tensors()` 单独检查是否依赖“最后一列一定是目标 item”这个前提

### 6. generative_recommenders/research/modeling/sequential/losses/sampled_softmax.py

#### 原则

最小闭环下，这个文件尽量不改 loss 实现本身，只改传入它的数据。

#### 保持不变的约束

- `supervision_ids` 仍然只包含 item id
- `supervision_embeddings` 仍然只包含 item embedding
- sampled negatives 仍然来自 item 集合

#### 需要上游保证

- 上游训练逻辑已经只抽出了有效的 item 监督位置
- 不能把 rating token 直接送进 sampled softmax

### 7. generative_recommenders/research/modeling/sequential/hstu.py

这一部分不是最小闭环必须，但如果你后续要让 HSTU 也支持 combined，这里必须改。

#### 必改

- `_attn_mask` 的长度改成预处理后的 token 长度
- relative attention bias 初始化长度改成 token 级长度
- `encode()` 取当前状态的位置改成 combined 后长度

#### 额外必须补的数据适配

HSTU 会把 `past_payloads["timestamps"]` 传进相对时间 bias 计算，因此 timestamps 也必须展开。

#### 建议的展开方式

- 原始 timestamps: `[ts_0, ts_1, ts_2, ...]`
- combined timestamps: `[ts_0, ts_0, ts_1, ts_1, ts_2, ts_2, ...]`

#### 否则会出现的问题

- 时间戳长度和 token 长度不一致
- relative bias 对不上 token 位置
- 即使 shape 勉强对上，时间语义也会错位

### 8. generative_recommenders/research/modeling/sequential/features.py

#### 需要确认但不一定立即改

这个文件负责从 dataset row 组装 `SequentialFeatures`。

#### 当前行为

- 会给历史序列后面拼接 output slots
- 会把 `target_timestamps` scatter 到尾部位置
- 不会把 `target_ratings` scatter 到历史 ratings 尾部

#### 对 combined 的影响

- 如果你采用“目标 item 后的 rating 用 0 占位”的方案，这里可以先不改
- 如果你决定把目标 rating 也注入上下文，这里就必须同步修改

#### 推荐

- 最小闭环先不注入 `target_rating`
- 只保留 `target_item` 注入和 rating=0 占位

### 9. configs/*.gin

#### 必改

所有需要启用 combined 预处理器的实验配置，都要补新的 `train_fn` 参数绑定。

#### 至少新增

- `train_fn.input_preprocessor_type = "combined_item_rating"`
- `train_fn.num_ratings = ...`

#### 注意

- 仅修改 gin 文件不会生效，前提是 `train.py` 里已经把预处理器选择改成可配置

## 推荐实现顺序

1. 先改 `train.py`，把预处理器选择做成可配置，并补 `num_ratings`
2. 再改 SASRec 的 encoder 长度和 `encode()` 当前状态位置
3. 再改训练 supervision 对齐逻辑
4. 先只跑 `SASRec + SampledSoftmaxLoss`
5. 确认训练 shape 和评估正常后，再扩展到 HSTU
6. 最后再看是否需要把这些能力正式抽象进 preprocessor 接口

## 最小闭环完成标准

满足下面几点，说明 SASRec 侧基本切通：

- 预处理器可以通过配置切到 `CombinedItemAndRatingInputFeaturesPreprocessor`
- 编码器 attention mask 长度能覆盖 combined token 序列
- `model.encode()` 取得的是最后有效 combined token 状态
- 训练时 `supervision_ids` 只包含 item id，不包含 rating token id
- negatives sampler 只处理 item，不处理 rating
- loss 能稳定接收对齐后的 query / item label 对
- 评估 top-k 检索仍然以 item 为目标正常运行

## 不建议一起做的事情

为了避免改动面过大，第一阶段不建议同时做下面这些事情：

- 不要同时把 rating token 也纳入 sampled softmax 候选集合
- 不要一开始就改 BCE 和 SampledSoftmax 两条 loss 分支
- 不要先做 HSTU cache / delta inference 路径适配
- 不要先追求所有模型统一抽象，先把 SASRec 跑通更稳妥

## 一句话总结

这次切换的本质不是“把默认类名换成 `CombinedItemAndRatingInputFeaturesPreprocessor`”，而是把整个训练和编码流程从“item 序列建模”改成“item/rating 交错 token 序列建模，但监督目标仍是下一个 item”。

## 最小 Diff 方案

这一节只给最小闭环方案，目标是先跑通：

- `SASRec`
- `SampledSoftmaxLoss`
- `CombinedItemAndRatingInputFeaturesPreprocessor`
- 训练目标仍然是“预测下一个 item”

不在第一步里覆盖：

- HSTU
- rating token 也参与候选 softmax
- cache / 增量推理路径

---

### A. generative_recommenders/research/trainer/train.py

#### A1. `train_fn()` 参数列表

#### 现在

`train_fn()` 没有 preprocessor 类型参数，也没有 `num_ratings` 参数。

#### 改成

在 `train_fn()` 参数列表中新增至少两个参数：

- `input_preprocessor_type: str = "learnable_positional"`
- `num_ratings: int = 6`

`num_ratings` 建议先按 MovieLens 这类数据的 `0` padding + `1~5` rating 处理成 6。

#### A2. 预处理器导入

#### 现在

只导入了：

- `LearnablePositionalEmbeddingInputFeaturesPreprocessor`

#### 改成

同时导入：

- `CombinedItemAndRatingInputFeaturesPreprocessor`

#### A3. `input_preproc_module = ...` 这一段

#### 现在

当前逻辑是固定写死：

- 实例化 `LearnablePositionalEmbeddingInputFeaturesPreprocessor`
- `max_sequence_len=dataset.max_sequence_length + gr_output_length + 1`

#### 改成

把这段改成按 `input_preprocessor_type` 分支：

##### 分支 1. 原逻辑保留

- 当 `input_preprocessor_type == "learnable_positional"` 时，继续实例化 `LearnablePositionalEmbeddingInputFeaturesPreprocessor`

##### 分支 2. 新增 combined 分支

- 当 `input_preprocessor_type == "combined_item_rating"` 时，实例化 `CombinedItemAndRatingInputFeaturesPreprocessor`
- 传入：
	- `max_sequence_len=dataset.max_sequence_length + gr_output_length + 1`
	- `item_embedding_dim=item_embedding_dim`
	- `dropout_rate=dropout_rate`
	- `num_ratings=num_ratings`

##### 分支 3. 兜底

- 否则抛 `ValueError`

#### A4. `get_sequential_encoder(...)` 调用参数

#### 现在

当前传入的是原始 item 序列长度：

- `max_sequence_length=dataset.max_sequence_length`
- `max_output_length=gr_output_length + 1`

#### 改成

先在 `train.py` 中引入两个局部变量：

- `encoder_max_sequence_length`
- `encoder_max_output_length`

##### 当使用原始 learnable positional 时

- `encoder_max_sequence_length = dataset.max_sequence_length`
- `encoder_max_output_length = gr_output_length + 1`

##### 当使用 combined 时

- `encoder_max_sequence_length = dataset.max_sequence_length * 2`
- `encoder_max_output_length = (gr_output_length + 1) * 2`

然后把 `get_sequential_encoder(...)` 的入参改为这两个 token 级长度。

#### 为什么要这样改

因为 `CombinedItemAndRatingInputFeaturesPreprocessor.forward()` 返回的是 `past_lengths * 2`，attention 看到的真实序列长度已经是 token 长度，不再是 item 长度。

#### A5. target 注入逻辑

#### 现在

训练循环中有一段：

- `seq_features.past_ids.scatter_(...)`

只把 `target_ids` 写进 `past_ids`，不写 `ratings`。

#### 最小方案保留什么

这段先保留不动，只注入 `target_item`，不注入 `target_rating`。

#### 语义固定为

- 最后一个目标 item 被拼到历史末尾
- 对应的 rating token 位置保持 0 占位
- 不把未来真实 rating 泄漏给模型

#### A6. 构造 `supervision_ids` 的位置

#### 现在

当前是：

- `supervision_ids = seq_features.past_ids`
- `ar_mask = supervision_ids[:, 1:] != 0`
- `output_embeddings=seq_embeddings[:, :-1, :]`
- `supervision_ids=supervision_ids[:, 1:]`
- `supervision_embeddings=input_embeddings[:, 1:, :]`

这是一套标准 item-only shift 训练法。

#### 改成

这整段要替换成“combined token 对齐”的逻辑。

##### 第一步：拿到 combined token ids 和 mask

当 `input_preproc_module` 是 `CombinedItemAndRatingInputFeaturesPreprocessor` 时，先计算：

- `combined_ids = input_preproc_module.get_preprocessed_ids(...)`
- `combined_mask = input_preproc_module.get_preprocessed_masks(...)`

这里的 `combined_ids` 语义是：

- 第 0,2,4,... 列是 item token
- 第 1,3,5,... 列是 rating token

##### 第二步：只选 rating 位置作为 query

定义一个 rating query 的切片：

- query 位置：`1, 3, 5, ...`

最小写法可以直接用：

- `query_embeddings = seq_embeddings[:, 1:-1:2, :]`

这里故意去掉最后一个 rating 位置之后越界的情况。

##### 第三步：label 取每个 rating 后面的下一个 item

label item 对应 combined 序列中的：

- `2, 4, 6, ...`

最小写法可以直接用：

- `label_item_ids = combined_ids[:, 2::2]`

##### 第四步：label embedding 仍然走 item embedding 表

直接取：

- `label_item_embeddings = model.module.get_item_embeddings(label_item_ids)`

不要从 combined token embedding 里取，因为 rating token 不是 item embedding 表的一部分。

##### 第五步：构造 supervision weight

只保留“当前 rating 有效且下一 item 非 0”的位置。

最小语义：

- rating query 位置有效
- label item 位置有效

最简单的构造可以以 `label_item_ids != 0` 为主，因为 combined preprocessor 的 mask 已经把 padding 区扩展好了。

##### 第六步：loss 入参替换

把原先这组：

- `output_embeddings=seq_embeddings[:, :-1, :]`
- `supervision_ids=supervision_ids[:, 1:]`
- `supervision_embeddings=input_embeddings[:, 1:, :]`
- `supervision_weights=ar_mask.float()`
- `lengths=seq_features.past_lengths`

改成 token 对齐后的这一组：

- `output_embeddings=query_embeddings`
- `supervision_ids=label_item_ids`
- `supervision_embeddings=label_item_embeddings`
- `supervision_weights=label_mask.float()`
- `lengths=query_lengths`

其中 `query_lengths` 的语义是“每个 batch 里有效 query 数”，最小规则可定义为：

- `query_lengths = seq_features.past_lengths`

原因：每个原始有效 item/rating 对，都会贡献一个 rating query，用来预测下一个 item；对齐后每行 query 数仍然等于原始有效 item 数。

#### A7. `negatives_sampler.process_batch(...)`

#### 现在

当前 in-batch 分支缓存的是：

- `in_batch_ids = supervision_ids.view(-1)`

#### 改成

在 combined 模式下，只缓存：

- `label_item_ids.view(-1)`

对应 embeddings 也只取 item embedding：

- `model.module.get_item_embeddings(label_item_ids.view(-1))`

不要把 rating token id 传进去。

---

### B. generative_recommenders/research/modeling/sequential/sasrec.py

#### B1. `__init__()` 中 `_max_sequence_length`

#### 现在

- `self._max_sequence_length = max_sequence_len + max_output_len`

#### 改什么

这行本身不一定要改写法，但要保证从 `train.py` 传进来的 `max_sequence_len` / `max_output_len` 已经是 token 级长度。

也就是说：

- 这里的数值语义要从“item 序列长度”变成“进入 attention 的真实 token 序列长度”

#### B2. `_attn_mask`

#### 现在

`_attn_mask` 是按 `self._max_sequence_length` 建的。

#### 最小方案

这里只要 `train.py` 传入的是 token 级长度，代码本身通常不用改。

#### B3. `encode()`

#### 现在

当前是：

- 调 `generate_user_embeddings(...)`
- 再 `get_current_embeddings(lengths=past_lengths, encoded_embeddings=...)`

#### 改成

在 `encode()` 里区分 preprocessor 类型。

##### 当 preprocessor 不是 combined

- 保持原逻辑

##### 当 preprocessor 是 combined

- 使用 `past_lengths * 2` 作为 `get_current_embeddings(...)` 的 `lengths`

#### 最小实现语义

把：

- `lengths=past_lengths`

改成：

- `lengths=current_lengths`

其中：

- `current_lengths = past_lengths`，默认情况
- `current_lengths = past_lengths * 2`，combined 情况

这样 `encode()` 才会取到最后一个有效 combined token，而不是取到最后一个 item token 位置。

---

### C. generative_recommenders/research/modeling/sequential/input_features_preprocessors.py

#### C1. `CombinedItemAndRatingInputFeaturesPreprocessor`

#### 现在

这个类已经提供了：

- `get_preprocessed_ids(...)`
- `get_preprocessed_masks(...)`
- `forward()` 返回 `past_lengths * 2`

#### 最小方案是否必须改这个文件

严格说，SASRec 最小闭环不一定非要改这个文件。

#### 但建议加一个小辅助方法

建议新增一个方法，语义类似：

- `get_preprocessed_lengths(past_lengths) -> past_lengths * 2`

#### 作用

- 避免在 `SASRec.encode()`、训练 loss 对齐逻辑里到处散落 `* 2`
- 让 combined 的长度规则集中在 preprocessor 内部

#### C2. HSTU 预留能力

如果后面要支持 HSTU，建议这个类再补一个方法：

- `get_preprocessed_timestamps(...)`

最小语义为：

- `[ts0, ts1, ts2] -> [ts0, ts0, ts1, ts1, ts2, ts2]`

SASRec 最小闭环阶段可以先不实现。

---

### D. generative_recommenders/research/modeling/sequential/utils.py

#### `get_current_embeddings()`

#### 现在

完全依赖外部传入的 `lengths`。

#### 最小方案

这个工具函数不改。

#### 改动落点

只在 `SASRec.encode()` 中把传入它的 `lengths` 改对即可。

---

### E. generative_recommenders/research/data/eval.py

#### E1. `eval_metrics_v2_from_tensors()`

#### 最小方案

这部分先不改。

#### 前提

只要 `model.encode()` 在 combined 模式下能够取到正确的最后状态，主评估链路通常还能继续用。

#### E2. `eval_recall_metrics_from_tensors()`

#### 现在

这里隐含假设：

- `seq_features.past_ids[:, -1]` 是 target

#### 最小方案

如果你的实验流程没有用到这个函数，可以先不改。

如果会用到，后续要把“最后一列是目标 item”的假设改成基于原始 item 序列长度定位 target。

---

### F. generative_recommenders/research/modeling/sequential/losses/sampled_softmax.py

#### 最小方案

不改。

#### 条件

上游已经保证传进来的：

- `output_embeddings` 是 rating query 的输出
- `supervision_ids` 全是 item id
- `supervision_embeddings` 全是 item embedding
- `supervision_weights` 对应有效 query 数
- `lengths` 对应每行有效 query 数

---

### G. configs/*.gin

#### 最小新增项

在你要使用 combined 的 gin 配置里加：

```gin
train_fn.input_preprocessor_type = "combined_item_rating"
train_fn.num_ratings = 6
```

如果数据集 rating vocab 不是 `0~5`，这里改成真实值。

---

## 最小改动后的训练数据流

下面用一条样本说明最小方案下训练怎么对齐。

### 原始输入

- 历史 item: `[i0, i1, i2]`
- 历史 rating: `[r0, r1, r2]`
- 目标 item: `i3`

### 注入 target item 后

- item 序列: `[i0, i1, i2, i3]`
- rating 序列: `[r0, r1, r2, 0]`

### combined token 序列

- `[i0, r0, i1, r1, i2, r2, i3, 0]`

### query 位置

- `r0, r1, r2`

### label 位置

- `i1, i2, i3`

### 训练含义

- 看到 `(i0, r0)` 后，预测 `i1`
- 看到 `(i1, r1)` 后，预测 `i2`
- 看到 `(i2, r2)` 后，预测 `i3`

这就是最小闭环方案。

---

## 最终建议的最小文件改动集合

如果你只追求最小可跑版本，第一轮实际上只动下面 4 个地方就够：

1. `generative_recommenders/research/trainer/train.py`
2. `generative_recommenders/research/modeling/sequential/sasrec.py`
3. `configs/...` 对应 gin 文件
4. 可选：`generative_recommenders/research/modeling/sequential/input_features_preprocessors.py` 增加 `get_preprocessed_lengths()` 辅助方法

第一轮不要动：

- `sampled_softmax.py`
- `utils.py`
- `eval.py` 主评估路径
- `hstu.py`

## 一句话版本

最小 diff 的核心只有三件事：

1. 在 `train.py` 把 preprocessor 切换做成可配置，并把 encoder 长度改成 token 级长度。
2. 在 `train.py` 把原来的 item-only `shift-one` loss，改成“rating 位置预测下一个 item”。
3. 在 `sasrec.py` 的 `encode()` 里把最后状态位置从 `past_lengths` 改成 combined 下的 `past_lengths * 2`。