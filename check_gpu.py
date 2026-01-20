"""
检查GPU使用情况并提供优化建议
"""
import torch
import os

print("="*70)
print("GPU使用情况诊断")
print("="*70)

# 检查CUDA是否可用
print(f"\nCUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
    # 检查GPU内存
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU总内存: {total_memory:.2f} GB")
    
    # 当前内存使用
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"已分配内存: {allocated:.2f} GB")
    print(f"已保留内存: {reserved:.2f} GB")

print("\n" + "="*70)
print("优化建议")
print("="*70)

print("""
GPU利用率低（1%）的原因分析：

1. **Batch Size太小** ⚠️
   当前: batch_size=32
   建议: 增加到 128-256（根据GPU内存）
   
2. **DataLoader未多线程** ⚠️
   当前: num_workers=0（默认）
   建议: num_workers=4-8（根据CPU核心数）
   
3. **数据预处理在CPU** ⚠️
   滤波、归一化都在CPU上
   建议: 使用pin_memory=True加速数据传输

4. **模型计算量相对较小**
   模型参数: 2.3M
   对于现代GPU，这个模型很小
   正常现象: GPU利用率可能不会很高

优化后预期：
- GPU利用率: 30-60%
- 训练速度: 提升2-3倍
""")

print("\n建议的优化参数：")
print("-" * 70)
print("batch_size = 128  # 增大batch size")
print("num_workers = 4   # 多线程数据加载")
print("pin_memory = True # 加速CPU->GPU传输")
print("=" * 70)
