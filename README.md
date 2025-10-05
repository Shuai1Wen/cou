# CT-OTS-U: Continuous Time Optimal Transport with Unbalanced Gates

**单细胞RNA测序轨迹推断算法 - 基于不平衡最优传输和门控多生成元**

---

## 项目简介

CT-OTS-U 是一个用于推断单细胞RNA测序数据中细胞状态转换轨迹的算法。基于不平衡最优传输（UOT）和门控多分支建模，能够处理细胞增殖/凋亡过程，并保证数学稳定性和时间可组合性。

### 核心特性

- **不平衡最优传输（UOT）**: 处理细胞增殖/凋亡导致的质量变化
- **门控多分支建模**: Bayesian GMM自适应发现细胞亚群分支
- **稳定性保证**: 软-硬约束确保谱半径<1的数学性质
- **半群一致性**: 保证时间可组合性（T_0→2 = T_1→2 ∘ T_0→1）
- **GPU加速**: 自动检测GPU并启用加速（可选）
- **10项优化**: BGMM门控、UOT平台、温度缩放、批效应决策等

---

## 快速开始

### 1. 环境要求

- **Python**: >= 3.11
- **操作系统**: Windows / Linux / macOS
- **内存**: >= 16GB RAM（推荐32GB用于大数据集）
- **GPU**: 可选（NVIDIA GPU with CUDA可加速2-5倍）

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装核心依赖
pip install -r requirements.txt
```

### 3. GPU加速（可选）

如果有NVIDIA GPU，可安装以下依赖获得更快速度：

```bash
# 根据CUDA版本选择（查看 nvidia-smi）
# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装GeomLoss（GPU加速OT计算）
pip install geomloss pykeops
```

**检查GPU是否可用**:
```bash
python -m ct_ots_u.utils.device_detect
```

### 4. 一键运行（推荐）

最简单的方式 - 在VS Code中打开并运行：

#### 方案A: 简单训练（5-10分钟）

```bash
# 1. 打开 scripts/train_simple.py
# 2. 修改配置区参数（如数据路径）
# 3. 按 F5 或点击运行按钮
```

#### 方案B: 完整流程（30-60分钟，推荐）

```bash
# 1. 打开 scripts/run_full_pipeline.py
# 2. 修改数据路径
# 3. 按 F5 运行
# 包含: 训练 + 7项分析 + 综合报告
```

### 5. 命令行运行

```bash
# 切换到项目根目录
cd CT-OTS-U-Clean

# 运行简单训练脚本
python scripts/train_simple.py

# 或运行完整流程
python scripts/run_full_pipeline.py
```

---

## 数据说明

项目包含三个预处理好的数据集：

### 1. GSE160936 - Microglia AD (主训练集)

- **路径**: `data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad`
- **规模**: 29,188细胞, 3,000基因, 24供者
- **标注**:
  - `braak_stage`: 0-6 (AD病理分期)
  - `sex`: male/female
  - `region`: EC (内嗅皮层) / SSC (体感皮层)
- **用途**: CT-OTS-U主训练、AD轨迹推断
- **阶段划分**:
  - Early: 0, 1, 2
  - Mid: 3, 4
  - Late: 5, 6

### 2. GSE213516 - PBMC Aging

- **路径**: `data/raw-use/GSE213516/processed/GSE213516_pbmc_processed.h5ad`
- **规模**: 130,581细胞, 3,000基因, 33样本
- **标注**:
  - `age`: 28-110岁
  - `sex`: male/female
- **用途**: 衰老轨迹推断、算法验证
- **阶段划分**:
  - Early: 40-49, 50-59
  - Mid: 60-69
  - Late: 70-79, 80+

### 3. GSE157783 - Midbrain PD (外部验证)

- **路径**: `data/raw-use/GSE157783/processed/GSE157783_midbrain_processed.h5ad`
- **规模**: 8,611细胞 (3,903微胶质), 11患者
- **标注**:
  - `diagnosis`: Control/PD
  - `cell_type`: Microglia/Astrocytes
- **用途**: 跨疾病迁移验证（AD→PD）

### 数据格式

所有数据均为 **AnnData** 格式 (.h5ad)，包含：

- `X`: 基因表达矩阵（归一化后）
- `obs`: 细胞元数据（阶段、批次、性别等）
- `var`: 基因信息
- `obsm['X_pca']`: PCA降维结果（64维）

---

## 脚本使用说明

### 脚本1: train_simple.py - 简易训练（推荐新手）

**特点**: 所有参数都在文件顶部，最简单易用。

**使用步骤**:

1. 打开 `scripts/train_simple.py`
2. 修改配置区（第15-67行）:

```python
# 数据文件
DATA_FILE = 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad'

# 阶段划分
EARLY_STAGES = '0,1,2'
MID_STAGES = '3,4'
LATE_STAGES = '5,6'

# 优化开关（推荐全部True）
USE_BGMM_GATING = True
USE_UOT_PLATEAU = True
USE_STABILITY = True
USE_TEMPERATURE = True
```

3. 运行: `python scripts/train_simple.py`

**输出**:
- 模型: `results/my_model.pkl`
- 摘要: `results/my_summary.json`

---

### 脚本2: train_quick.py - 快速场景切换

**特点**: 预设4种训练场景，一键切换。

**使用步骤**:

1. 打开 `scripts/train_quick.py`
2. 选择场景（第19行）:

```python
SCENARIO = 'AD_TO_PD'  # 可选: 'AD_FULL', 'AD_TO_PD', 'PBMC', 'BASELINE'
```

3. 运行: `python scripts/train_quick.py`

**场景说明**:

| 场景 | 说明 | 优化项 | 时间 |
|------|------|--------|------|
| `AD_FULL` | AD微胶质完整优化 | 6项 | 20-30min |
| `AD_TO_PD` | AD→PD跨域验证 | 10项全开 | 30-40min |
| `PBMC` | PBMC衰老训练 | 3项 | 15-25min |
| `BASELINE` | 基线对照（无优化） | 0项 | 15-20min |

---

### 脚本3: run_full_pipeline.py - 完整流程（最推荐）

**特点**: 训练 + 7项分析 + 综合报告，一站式完成。

**使用步骤**:

1. 打开 `scripts/run_full_pipeline.py`
2. 修改配置（第20-75行）:

```python
# 数据路径
DATA_MICROGLIA = Path('data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad')

# 优化开关
USE_BGMM_GATING = True
USE_TEMPERATURE = True
USE_CROSS_DOMAIN = True
```

3. 运行: `python scripts/run_full_pipeline.py`

**输出**:
- 训练模型: `results/full_pipeline/train_model.json`
- 训练摘要: `results/full_pipeline/train_summary.json`
- 分析报告: `results/full_pipeline/analysis/summary_report.md`

**包含的7项分析**:

1. **UOT反事实分析** - 干预实验模拟
2. **半群一致性验证** - 时间可组合性检查
3. **Rank稳定性分析** - 秩选择鲁棒性
4. **跨数据集迁移** - AD→PD泛化能力
5. **通路富集分析** - 生物学解释（需GMT文件）
6. **mLOY基因检测** - Y染色体丢失分析
7. **PBMC衰老分析** - 衰老轨迹验证

---

### 脚本4: run_train_config.py - 高级配置

**特点**: 完整的参数控制，适合高级用户和消融实验。

**使用步骤**:

1. 打开 `scripts/run_train_config.py`
2. 修改 `TRAIN_CONFIG` 字典（第19行开始）
3. 运行: `python scripts/run_train_config.py`
4. 按 Enter 确认开始训练

**配置示例**:

```python
TRAIN_CONFIG = {
    'data': {
        'h5ad': 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad',
        'stage_key': 'braak_stage',
        'early_stages': '0,1,2',
        'mid_stages': '3,4',
        'late_stages': '5,6',
        'batch_key': 'region',
    },
    'training': {
        'steps': 100,
        'lr': 0.01,
        'rank_grid': [16, 24, 32, 40, 48],
    },
    'optimizations': {
        'use_bgmm': True,
        'use_plateau_lock': True,
        'use_temperature_scaling': True,
    },
    # ... 更多参数
}
```

---

## 常见问题

### Q1: 如何查看训练结果？

**A**: 查看输出的JSON文件，例如：

```bash
# 使用任何文本编辑器或JSON查看器
code results/my_summary.json  # VS Code
notepad results/my_summary.json  # Windows记事本
```

关键字段：
- `final_rank`: 最终选择的秩
- `final_k`: 最终门控分支数
- `spectral_radius`: 谱半径（应<1）
- `semigroup_consistency`: 半群一致性得分

### Q2: 训练时间太长怎么办？

**A**: 三个方法：

1. **减少训练步数**（train_simple.py中修改）:
```python
# 找到并修改（不影响结果质量）
config.train.steps = 50  # 默认100
```

2. **减少数据采样**:
```python
config.data.sample_size = 1000  # 默认2000
```

3. **安装GPU加速**（见上文"GPU加速"部分）

### Q3: 出现"GeomLoss not available"警告？

**A**: 这是正常的，算法会自动降级到POT后端：
- 算法完全一致
- 仅速度稍慢（约1.5-2倍）
- 如需加速，安装: `pip install geomloss pykeops`

### Q4: 如何使用自己的数据？

**A**: 准备AnnData格式数据：

```python
import scanpy as sc
import anndata as ad

# 读取你的数据
adata = sc.read_csv('your_data.csv')  # 或其他格式

# 添加必要的元数据列
adata.obs['stage'] = ...  # 阶段标签
adata.obs['batch'] = ...  # 批次标签（可选）

# 预处理
sc.pp.filter_cells(adata, min_counts=500)
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
sc.pp.pca(adata, n_comps=64)

# 保存
adata.write('my_data.h5ad')
```

然后在脚本中修改 `DATA_FILE = 'my_data.h5ad'`

### Q5: 结果如何解释？

**A**: 关键指标：

- **Spectral Radius < 1**: 系统稳定，不会爆炸增长
- **Semigroup Consistency**: 时间可组合性，越接近1越好
- **Final K**: 发现的细胞亚群分支数
- **Final Rank**: 生成元矩阵的秩（复杂度）

---

## 项目结构

```
CT-OTS-U-Clean/
├── ct_ots_u/                    # 核心算法包
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── ct_ots_model.py         # 半群模型
│   ├── gating_kselect.py       # 门控选择
│   ├── rank_select.py          # 秩选择
│   ├── stability.py            # 稳定性投影
│   ├── io_prep.py              # 数据预处理
│   ├── ot/                     # 最优传输
│   │   └── uot_losses.py       # UOT损失函数
│   ├── model/                  # 训练模块
│   │   ├── train.py            # 梯度下降训练
│   │   └── semigroup.py        # 半群算子
│   ├── scripts/                # 内部脚本
│   │   ├── run_train.py        # 训练主程序
│   │   └── run_validate.py     # 验证程序
│   └── utils/                  # 工具函数
│       ├── device_detect.py    # GPU检测
│       └── seed.py             # 随机种子
│
├── data/                        # 数据目录
│   └── raw-use/
│       ├── GSE160936/          # AD微胶质数据
│       ├── GSE213516/          # PBMC衰老数据
│       └── GSE157783/          # PD中脑数据
│
├── scripts/                     # 运行脚本（用户使用）
│   ├── train_simple.py         # 简易训练
│   ├── train_quick.py          # 快速场景
│   ├── run_full_pipeline.py    # 完整流程
│   └── run_train_config.py     # 高级配置
│
├── results/                     # 输出目录（运行后生成）
├── resources/                   # 外部资源（如GMT文件）
├── requirements.txt             # Python依赖
└── README.md                    # 本文档
```

---

## 高级用法

### 消融实验

比较不同优化的效果：

```bash
# 1. 基线（无优化）
python scripts/train_quick.py  # SCENARIO='BASELINE'

# 2. 仅BGMM门控
# 修改train_simple.py:
USE_BGMM_GATING = True
USE_UOT_PLATEAU = False
USE_STABILITY = False

# 3. 完整优化
USE_BGMM_GATING = True
USE_UOT_PLATEAU = True
USE_STABILITY = True
USE_TEMPERATURE = True
```

### 超参数网格搜索

修改 `run_train_config.py`:

```python
TRAIN_CONFIG = {
    'hyperparam_search': {
        'tau_grid': [0.5, 1.0, 2.0],
        'reg_grid': [0.05, 0.08, 0.1],
        'repeats': 3,
    }
}
```

### 批处理多个数据集

创建批处理脚本：

```python
datasets = [
    'GSE160936',
    'GSE213516',
    'GSE157783',
]

for dataset in datasets:
    # 修改配置
    # 运行训练
    # 保存结果
```

---

## 引用

如果使用本算法，请引用：

```bibtex
@article{ct-ots-u-2024,
  title={CT-OTS-U: Continuous Time Optimal Transport with Unbalanced Gates for Single-Cell Trajectory Inference},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

---

## 技术支持

- **问题报告**: 在项目issue中提交
- **文档**: 查看 `README.md` 和脚本内注释
- **GPU检测**: `python -m ct_ots_u.utils.device_detect`

---

## 更新日志

### v1.0.0 (2024-10-04)

- ✅ 完整的CT-OTS-U算法实现
- ✅ 10项优化全部集成（BGMM、UOT平台、温度缩放等）
- ✅ GPU自动检测和降级
- ✅ 4个用户友好的运行脚本
- ✅ 3个预处理数据集
- ✅ 完整的文档和使用说明

---

## 许可证

本项目遵循 MIT 许可证。

---

**祝使用愉快！**
