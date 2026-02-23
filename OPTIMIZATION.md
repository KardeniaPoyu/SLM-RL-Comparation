# GRPO 与 PPO 训练优化指南

## 📋 更新内容

### 新增功能

#### 1. **Response 日志记录**
- **GRPO**: 所有生成的 response 保存到 `logs/grpo_responses.txt`
- **PPO**: 所有生成的 response 保存到 `logs/ppo_responses.txt`
- 每个 response 都标有对应的 Step/Update 和 Sample 序号
- 使用清晰的分隔线便于人工查看

#### 2. **新增分析工具**
创建了 `analyze_responses.py` 脚本用于：
- 解析 response 日志文件
- 统计 response 长度、token 数等信息
- 分析 response 质量（如包含 `<think>` 标签的比例）
- 对比 GRPO 与 PPO 的输出差异
- 生成汇总报告

### 优化改动

#### 1. **内存管理优化**
- 确保每个 epoch 后及时释放缓存
- 重复使用 `torch.cuda.empty_cache()` 和 `gc.collect()`
- 减少显存峰值占用

#### 2. **日志输出改进**
- 保留原有的 CSV 指标记录
- 增加新的 Response 日志输出
- 训练完成时打印文件位置统计信息

#### 3. **代码结构优化**
- 统一两个训练脚本的日志初始化逻辑
- 改进文件关闭机制确保数据完整性
- 添加训练完成提示信息

## 🚀 使用方法

### 训练

```bash
# 运行 GRPO 训练
python train_grpo.py

# 运行 PPO 训练
python train_ppo.py
```

### 分析 Response 日志

```bash
# 自动分析两个日志并生成对比报告
python analyze_responses.py
```

输出会包括：
- GRPO 与 PPO 的 response 数量统计
- 字符长度统计 (平均、最大、最小)
- Token 数估计
- 样本质量分析（如是否包含思考标签）
- 具体样本对比

### 输出文件说明

```
logs/
├── grpo_metrics.csv          # GRPO 训练指标
├── grpo_responses.txt        # GRPO 所有生成的 response
├── ppo_metrics.csv           # PPO 训练指标
├── ppo_responses.txt         # PPO 所有生成的 response
└── response_summary.txt      # 分析工具生成的汇总报告
```

## 📊 Response 日志格式

```
=== GRPO Training Responses Log ===

Step 0 - Sample 0:
<生成的 response 内容>
--------------------------------------------------------------------------------
Step 0 - Sample 1:
<生成的 response 内容>
--------------------------------------------------------------------------------
...
```

## 💡 使用 Response 日志的场景

1. **调试模型输出**：查看原始生成内容是否合理
2. **对比分析**：比较两种方法在同一 step / update 的输出差异
3. **质量评估**：分析生成的思考过程和最终答案
4. **后续改进**：根据具体样本优化提示词或模型架构
5. **论文撰写**：提供具体的输出示例支撑论文论述

## ⚙️ 性能影响

- **I/O 开销**：每个 response 都会被写入文件，约增加 5-10% 的训练时间
- **磁盘空间**：取决于模型生成的 response 长度和总数量
  - 单步 8 个 response（256 tokens）≈ ~2-3 KB
  - 10 个 update = ~20-30 KB

## 🔍 常见问题

### Q: 如何避免日志文件过大？
- 设置 `max_samples` 限制数据集大小
- 使用测试集而不是全量训练集
- 定期清理旧日志

### Q: 可以修改日志格式吗？
- 可以编辑 `train_grpo.py` 和 `train_ppo.py` 中的日志写入部分
- 或自定义 `analyze_responses.py` 中的解析逻辑

### Q: Response 日志会影响训练速度吗？
- 有轻微影响，但对整体训练时间占比不超过 10%
- 如要完全禁用，注释掉 `response_file.write()` 部分即可

## 📝 下一步改进建议

- [ ] 添加 response 去重功能
- [ ] 按成功率筛选高质量 response
- [ ] 与 metrics CSV 关联，按成功与失败分开存储
- [ ] 生成可视化的 response 对比图表
- [ ] 添加自动答案抽取和正确率统计
