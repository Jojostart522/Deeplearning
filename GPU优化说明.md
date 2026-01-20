# GPU优化说明 - 大幅提升训练速度

## 已完成的优化

### 1. 混合精度训练（AMP）⭐⭐⭐
**最重要的优化，可提升2-3倍速度**

- 使用 `torch.cuda.amp.autocast()` 自动混合精度
- 使用 `GradScaler` 处理梯度缩放
- 训练和推理都启用混合精度
- **预期效果：GPU利用率从1%提升到60-80%**

### 2. 数据加载优化⭐⭐
**提升数据读取速度，避免GPU等待**

- `num_workers=8`（从4增加到8）
- `persistent_workers=True`（避免每个epoch重新创建worker）
- `prefetch_factor=4`（预取4个批次）
- `pin_memory=True`（已有）
- `non_blocking=True`（异步数据传输）

### 3. 批处理优化⭐⭐
**更大的batch size充分利用GPU并行能力**

- 训练batch_size：512（从256增加）
- 预测batch_size：1024（从逐个样本改为批量）
- **预期效果：训练速度提升1.5-2倍**

### 4. 预测函数优化⭐⭐⭐
**最大的瓶颈之一**

- 之前：逐个样本预测（180000次循环）
- 现在：批量预测（180000/1024 ≈ 176次循环）
- **预期效果：预测速度提升100倍以上**

### 5. 其他优化⭐
- `torch.backends.cudnn.benchmark=True`（自动寻找最优卷积算法）
- `optimizer.zero_grad(set_to_none=True)`（更快的梯度清零）

## 优化效果预测

### 训练阶段
- **之前**：GPU利用率1%，每个epoch可能需要几分钟
- **现在**：GPU利用率60-80%，每个epoch预计10-30秒

### 预测阶段
- **之前**：180000个样本逐个预测，可能需要几分钟
- **现在**：批量预测，预计只需几秒钟

### 总体提升
- **训练速度**：提升5-10倍
- **预测速度**：提升100倍以上
- **GPU利用率**：从1%提升到60-80%

## 如何验证优化效果

### 1. 查看GPU利用率
打开任务管理器（Ctrl+Shift+Esc）→ 性能 → GPU
- 应该看到GPU利用率在60-80%
- GPU内存使用应该明显增加

### 2. 查看训练速度
运行程序后，观察每个epoch的时间：
```
Epoch [5/300], Train Loss: 0.123456, Val Loss: 0.234567, Patience: 0/50
```
- 之前：每5个epoch可能需要几分钟
- 现在：每5个epoch应该只需要1-2分钟

### 3. 查看预测速度
观察预测阶段的输出：
```
low 频段预测完成，信号长度: 180000
```
- 之前：可能需要几分钟
- 现在：应该只需要几秒钟

## 进一步优化建议

### 如果显存充足（16GB+）
可以进一步增大batch_size：
```python
batch_size = 1024  # 训练
batch_size = 2048  # 预测
```

### 如果显存不足
可以减小batch_size：
```python
batch_size = 256  # 训练
batch_size = 512   # 预测
```

### 如果CPU性能不足
可以减少num_workers：
```python
num_workers=4  # 从8减少到4
```

## 代码修改位置

### 1. `dl_anc_model.py`
- **第354-457行**：`train_model()` 函数（添加AMP支持）
- **第459-522行**：`predict_full_signal()` 函数（批量预测）

### 2. `dl_anc_multiband.py`
- **第138-143行**：DataLoader配置（增加workers和预取）
- **第203行**：batch_size增加到512
- **第338-339行**：predict_full_signal调用（添加batch_size参数）

## 常见问题

### Q1: 显存不足怎么办？
**A:** 减小batch_size：
```python
batch_size = 256  # 或更小
```

### Q2: 训练速度还是慢？
**A:** 检查：
1. GPU是否被其他程序占用
2. 是否真的在使用GPU（检查device是否为cuda）
3. 数据是否在GPU上（检查tensor.device）

### Q3: 出现CUDA out of memory错误？
**A:** 减小batch_size或减小模型hidden_size：
```python
batch_size = 128
hidden_size = 256  # 从384或640减小
```

### Q4: num_workers报错？
**A:** Windows上可能需要在main函数中添加：
```python
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main_multiband()
```

## 性能监控

### 使用nvidia-smi监控GPU
打开命令行，运行：
```bash
nvidia-smi -l 1
```
可以实时查看：
- GPU利用率
- 显存使用
- 温度
- 功耗

### 使用PyTorch Profiler（高级）
如果需要详细分析性能瓶颈：
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # 训练代码
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 总结

这次优化主要解决了三个问题：
1. **混合精度训练**：大幅提升GPU计算效率
2. **数据加载优化**：避免GPU等待数据
3. **批量预测**：消除最大的性能瓶颈

预期整体训练速度提升**5-10倍**，GPU利用率从**1%提升到60-80%**。

现在可以运行程序测试效果了！
