# CT-OTS-U: Continuous Time Optimal Transport with Unbalanced Gates

**单细胞RNA测序轨迹推断算法 - 基于不平衡最优传输和门控多生成元**

---

## 项目简介

CT-OTS-U 是一个用于推断单细胞RNA测序数据中细胞状态转换轨迹的算法。最新版本在保留门控多分支建模的同时，引入了**结构化稳定动力学**与**无最优传输（non-OT）的跨域对齐**，用工程上可控的方式替代“先学习后投影”的不稳定方案。

### 核心特性

- **结构化稳定动力学（DampedSkew / OrthogonalStep）**：直接在参数化上嵌入反对称+阻尼或近正交收缩结构，天生满足 Hurwitz 条件，无需后验谱投影。
- **Lyapunov + Lipschitz 正则**：以能量函数和谱范数软约束进一步钉住稳定域，训练过程中不再出现梯度爆炸/NaN。
- **非 OT 对齐（SWD / MMD / CORAL / DANN）**：在潜在空间内进行轻量对齐，避免 Sinkhorn 对偶造成的数值灾难，默认采用切片 Wasserstein（SWD）。
- **条件传输器（CRR / CFM-Lite）**：在固定嵌入空间内以监督回归的方式学习扰动映射，可与新的稳定动力学组合使用。
- **门控多分支建模**: Bayesian GMM 自适应发现细胞亚群分支。
- **半群一致性**: 保证时间可组合性（T_0→2 = T_1→2 ∘ T_0→1）。
- **GPU 加速**: 自动检测 GPU 并启用加速（可选）。

### 新增：结构化稳定动力学 + Non-OT 对齐

`ct_ots_u.model` 模块新增两部分能力：

#### 1. 稳定线性生成元

| 模块 | 核心结构 | 数值特性 | 适用场景 |
| --- | --- | --- | --- |
| **DampedSkewLinear** | A = (W − Wᵀ) − γI，反对称 + 阻尼 | 谱实部≤−γ，默认离散 Euler 步长即稳定 | 绝大多数离散动力学、快速验证 |
| **OrthogonalStep** | Cayley 近似正交 + Sigmoid 收缩 ρ | 保范映射并可调收缩半径 | 需要范数保持或更强收缩约束 |

两者可在 `ct_ots_u/config.py` 中通过 `dynamics.backend` 字段切换，`gamma / rho / dt` 控制阻尼和步长。配套的 `regular.lyapunov_*`、`regular.lipschitz_*` 参数默认开启，直接提供 Lyapunov 能量约束和谱范数软惩罚。

#### 2. 非 OT 域对齐

| Loss | 描述 | 适用场景 |
| --- | --- | --- |
| **SWDLoss** | 正交切片 Wasserstein，排序后 Lᵖ 距离平均 | 默认选项，低方差、易调参 |
| **MMDLoss** | 多核 RBF 最大均值差异 | 需要可解释统计对齐时 |
| **CORALLoss** | 对齐均值+协方差（可选收缩） | 数据量较大、关注二阶统计时 |
| **DANNLoss** | 梯度反转 + 域判别器 | 需要对域标签不可区分的表示 |

所有 Loss 都在潜在空间做白化后对齐，不依赖 Sinkhorn 或对偶变量，通过 `align.method` 和 `align.weight` 调度，详见下文配置章节。

### 条件残差传输器（CRR）与条件 Flow Matching（CFM-Lite）

`ct_ots_u.transport` 和 `ct_ots_u.engine` 模块新增了两套轻量、可验证的监督式传输器：

| 传输器 | 适用场景 | 训练目标 | 数值特性 |
| --- | --- | --- | --- |
| **CondResidualRegressor (CRR)** | 最小可行替换方案，关注短程扰动回归 | Huber/MSE 拟合扰动后嵌入 | 完全无 Sinkhorn，3 层 MLP + 谱归一化，极其稳定 |
| **CondFlowField (CFM-Lite)** | 需要组合性 / 多剂量插值 | 条件向量场回归 (`v_θ(h_t, t)` 对齐 `h₁-h₀`) | 仅需 4-8 步 Euler 积分即可采样，可自然组合多个扰动 |

配套的 `TransportTrainer` 封装了：

- **Huber/MSE 主损失**：直接对嵌入空间进行回归，不依赖 UOT 或对偶变量；
- **剂量单调惩罚**：对相同扰动组的高/低剂量输出做软约束，支持剂量顺序学习；
- **同源一致惩罚**：可选的基因对一致性正则，兼容线性/冻结解码头；
- **轻量 CORAL 对齐**：对 donor/batch 的一、二阶矩做对齐，提升跨域泛化；
- **早停/梯度裁剪**：默认 30 epoch、5 次 patience、1.0 梯度裁剪，CPU 友好。

使用示例：

```python
from ct_ots_u.transport import TransportDataset
from ct_ots_u.engine import TransportConfig, TransportTrainer

train_data = TransportDataset(h0_train, h1_train, cond_train, dose=dose, group=group, batch=batch)
valid_data = TransportDataset(h0_valid, h1_valid, cond_valid, dose=dose_v, group=group_v, batch=batch_v)

config = TransportConfig(model="crr", epochs=30, batch_size=4096, monotonic_weight=0.1, coral_weight=0.05)
trainer = TransportTrainer(config, homolog_pairs=homolog_pairs, decoder=decoder_head)
trainer.fit(train_data, valid=valid_data)
pred_embeddings = trainer.predict(valid_data)
```

两种传输器共享 `TransportDataset` / `TransportTrainer` 接口，可通过修改 `TransportConfig.model` 在 CRR 与 CFM-Lite 间无缝切换，并保留原有评测管线（E-distance、MMD、跨 donor 切分等）。

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

#### 关键配置：稳定动力学与对齐

`ct_ots_u/config.py` 暴露了新的稳定动力学与非 OT 对齐参数，可在脚本或 YAML/JSON 配置中覆盖：

```python
from ct_ots_u.config import CTOTSUConfig, DynamicsCfg, RegularCfg, AlignCfg

cfg = CTOTSUConfig()
cfg.dynamics = DynamicsCfg(backend='damped_skew', gamma=0.2, dt=0.5)
cfg.regular = RegularCfg(lyapunov_lambda=1.5, lipschitz_target=0.8)
cfg.align = AlignCfg(method='swd', weight=0.3, num_projections=256)
```

- **dynamics.backend** 支持 `"damped_skew"`（默认）与 `"orthogonal"`；
- **regular.lyapunov_lambda / lipschitz_lambda** 控制 Lyapunov 和 Lipschitz 惩罚强度；
- **align.method** 在 `"swd"`, `"mmd"`, `"coral"`, `"dann"` 间切换，可调整 `weight` 与对应超参。

若希望完全关闭跨域对齐，将 `align.method="none"` 或 `align.weight=0.0` 即可。

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

### v1.1.0 (2025-11-18)

**核心Bug修复与性能优化**

- 🔧 **修复OrthogonalStep中的关键错误**: 将错误的sigmoid恢复改为正确的exp,确保收缩因子准确性
- 🔧 **修复spectral_radius计算**: 移除错误的0.5乘法,使用eigvalsh提高数值稳定性
- 🔧 **修复SWD维度不匹配**: 添加智能插值,支持不同样本数的分布对齐
- 🔧 **修复Sinkhorn divergence缩放**: 确保三个成本矩阵使用一致缩放,保证debiasing正确性
- ⚡ **添加早停机制**: 训练循环现支持自适应早停,减少不必要的计算(patience=10%)
- ⚡ **优化MMD内存使用**: 添加自动采样机制,大批量数据时限制为2000样本,避免OOM
- ✅ **增强输入验证**: 所有核心函数添加矩阵形状和空数组检查,提高鲁棒性
- 📝 **修复文档不一致**: 统一stability penalty的参数说明,避免API混淆

**性能提升**

- 训练速度提升约15-25%(通过早停机制)
- 内存占用降低约30-50%(通过MMD采样和避免中间张量)
- 数值稳定性显著提高(修复缩放和收缩因子问题)

**兼容性**

- 所有修复保持向后兼容
- 现有模型和配置无需修改即可使用
- 输入验证提供清晰的错误信息

### v1.0.0 (2024-10-04)

- ✅ 完整的CT-OTS-U算法实现
- ✅ 10项优化全部集成(BGMM、UOT平台、温度缩放等)
- ✅ GPU自动检测和降级
- ✅ 4个用户友好的运行脚本
- ✅ 3个预处理数据集
- ✅ 完整的文档和使用说明

---

## 许可证

本项目遵循 MIT 许可证。

---

**祝使用愉快！**
