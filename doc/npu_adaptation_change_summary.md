# DLRM v3 NPU 适配修改总结

## 总览
本文汇总了 DLRM v3 在 Ascend NPU 上运行所做的关键适配与稳定性修改，以及指标系统和 train-eval 行为修复。

## 1) 运行时与入口配置
- 增加了 device/backend/kernel/model-parallel 的运行时切换参数。
- 主要目的：让训练入口可在 `npu` 上运行，并支持回退到纯 PyTorch kernel 路径。

位置：
- generative_recommenders/dlrm_v3/train/train_ranker.py

## 2) 分布式与优化器路径重构（Dense vs Model Parallel）
- 在优化器与分片流程中增加了 non-model-parallel 路径。
- 保持 embedding 参数与 dense 参数分开优化（CombinedOptimizer）。
- 分布式初始化支持 backend/device_type 配置（`nccl/gloo/hccl` 与 `cuda/npu/cpu`）。

位置：
- generative_recommenders/dlrm_v3/train/utils.py

## 3) 训练循环稳定性与语义保护
- 训练循环增加了 gradient accumulation 控制。
- 增加 strict semantics 校验辅助逻辑。
- 在 train/train-eval/streaming 路径中增加 non-finite 诊断（forward、loss、grad、首个异常参数）。
- 修复 train-eval 的 eval 循环行为：未设置 `num_eval_batches` 时，eval 跑完一轮 eval dataloader 后结束（不再无限 eval）。

位置：
- generative_recommenders/dlrm_v3/train/utils.py

## 4) NPU Embedding 稳定化
- 在 Dense+NPU 路径下规避直接不稳定的 embedding lookup 行为。
- 增加 CPU embedding fallback 控制与诊断：
  - device 迁移前后的 embedding 参数健康检查；
  - 可选将 embedding collection 固定在 CPU；
  - 按 feature 的 embedding lookup 诊断（non-finite 与越界检查）；
  - 在 NPU fallback 场景下使用按 feature 的 `torch.nn.functional.embedding` 手工路径。

位置：
- generative_recommenders/modules/dlrm_hstu.py
- generative_recommenders/dlrm_v3/train/utils.py

## 5) Autocast 设备泛化
- 将硬编码 `"cuda"` 的 autocast 调用改为动态 `device_type=...device.type`。
- 主要目的：避免在 NPU 上触发运行时错误。

位置：
- generative_recommenders/modules/dlrm_hstu.py
- generative_recommenders/modules/multitask_module.py
- generative_recommenders/modules/contextual_interleave_preprocessor.py

## 6) Attention fallback 内存优化
- 在 PyTorch fallback 路径中增加 chunk 化与低峰值累加策略。
- 增加基于环境变量的 chunk 控制，降低 OOM 风险。

位置：
- generative_recommenders/ops/pytorch/pt_hstu_attention.py

## 7) 指标系统修复（NPU + 分布式兼容）
- 增加 AUC 计算与缓存逻辑。
- NPU 场景支持将 metric 状态放在 CPU 上。
- 在 metric compute 路径增加 CPU/分布式兼容保护，避免 `No backend type associated with device type cpu`。
- 增加指标输入 non-finite 诊断。
- profiler 的 activities/sort key 改为按设备动态选择（去除 CUDA 硬编码假设）。

位置：
- generative_recommenders/dlrm_v3/utils.py

## 8) Train-Eval GIN 行为调优
- 增加显式 `train_eval_loop.num_eval_batches = 47`，使 movielens-1m 的每次 eval 有界且稳定。

位置：
- generative_recommenders/dlrm_v3/train/gin/movielens_1m.gin

## 9) 本地调试/缩放类改动（工作区特定）
- 另外存在一些工作区级调优修改（例如缩小 hash/embedding 尺度、debug gin 调整），便于 bring-up，但可能与正式复现配置不同。

位置（按当前工作区状态）：
- generative_recommenders/dlrm_v3/configs.py
- generative_recommenders/dlrm_v3/train/gin/debug.gin

## 备注
- 如果实际运行路径为 `/home/code/...`，测试前请同步当前仓库改动到运行环境。
- FBGEMM 在 NPU 上 unsupported-op 的 warning 可视为性能告警；正确性以指标是否有限、稳定及趋势是否正常为准。
