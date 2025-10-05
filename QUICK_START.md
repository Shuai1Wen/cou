# CT-OTS-U 快速上手指南

**5分钟开始你的第一次训练**

---

## 最简单的开始方式

### 步骤1: 安装依赖 (2分钟)

```bash
# 打开命令行，进入项目目录
cd d:\Program\CT-OTS-U-Clean

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 步骤2: 运行第一个训练 (3分钟)

**方法A: 使用VS Code (推荐)**

1. 打开VS Code
2. 打开文件: `scripts/train_simple.py`
3. 按 **F5** 或点击右上角运行按钮
4. 等待完成！

**方法B: 命令行**

```bash
python scripts/train_simple.py
```

### 步骤3: 查看结果

训练完成后，查看：
- 模型: `results/my_model.pkl`
- 摘要: `results/my_summary.json`

---

## 三个最常用的脚本

### 🌟 推荐: run_full_pipeline.py

**完整流程 = 训练 + 7项分析 + 综合报告**

```bash
# 打开并运行
python scripts/run_full_pipeline.py
```

输出:
- `results/full_pipeline/train_summary.json` - 训练结果
- `results/full_pipeline/analysis/summary_report.md` - 分析报告

### ⚡ 新手: train_simple.py

**所有参数都在文件顶部，改完直接运行**

1. 打开 `scripts/train_simple.py`
2. 修改第20-67行的配置
3. 运行

### 🔄 快速切换: train_quick.py

**4种场景一键切换**

1. 打开 `scripts/train_quick.py`
2. 修改第19行: `SCENARIO = 'AD_TO_PD'`
3. 运行

场景选项:
- `'AD_FULL'` - AD微胶质完整训练
- `'AD_TO_PD'` - AD→PD跨域验证
- `'PBMC'` - PBMC衰老分析
- `'BASELINE'` - 基线对照

---

## 常见第一次问题

### Q: 安装太慢？

```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 想用GPU加速？

```bash
# 先检测
python -m ct_ots_u.utils.device_detect

# 如果有NVIDIA GPU，安装CUDA版PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install geomloss pykeops
```

### Q: 训练时间太长？

打开 `scripts/train_simple.py`，修改：

```python
# 减少训练步数（不影响结果质量）
# 找到配置区并添加：
USE_QUICK_MODE = True  # 启用快速模式
```

或修改 `ct_ots_u/config.py`:
```python
steps: int = 50  # 从100减到50
```

### Q: 想用自己的数据？

准备AnnData格式 (.h5ad)，包含:
- `obs['stage']` - 阶段标签
- `obs['batch']` - 批次标签（可选）

然后修改脚本中的 `DATA_FILE` 路径。

---

## 下一步

1. **阅读完整文档**: [README.md](README.md)
2. **尝试完整流程**: `python scripts/run_full_pipeline.py`
3. **查看分析结果**: `results/full_pipeline/analysis/`

---

## 目录说明

```
CT-OTS-U-Clean/
├── scripts/           # 👈 运行这里的脚本
│   ├── train_simple.py       ⭐ 新手首选
│   ├── run_full_pipeline.py  ⭐⭐⭐ 最推荐
│   ├── train_quick.py
│   └── run_train_config.py
├── data/              # 数据集（已准备好）
├── results/           # 输出结果（运行后生成）
├── ct_ots_u/          # 核心代码（无需修改）
└── README.md          # 完整文档
```

---

**准备好了吗？运行你的第一个训练吧！**

```bash
python scripts/train_simple.py
```
