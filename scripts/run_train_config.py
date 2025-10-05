"""
CT-OTS-U 训练脚本 - 配置文件驱动版本

直接在VS中运行，所有参数在代码中配置
修改参数只需编辑 TRAIN_CONFIG 字典即可
"""

import sys
from pathlib import Path

# ========================================
# 🎯 训练配置 - 在这里修改所有参数
# ========================================

TRAIN_CONFIG = {
    # === 基础数据配置 ===
    'h5ad': 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad',
    'stage_key': 'braak_stage',
    'early': '0,1,2',
    'mid': '3,4',
    'late': '5,6',

    # === 可选列配置 ===
    'celltype_key': None,  # 细胞类型列 (可选)
    'celltype': None,      # 筛选的细胞类型 (可选)
    'batch_key': 'region', # 批次标签列 (启用Harmony时需要)
    'domain_key': 'sex',   # 域标签列 (启用DANN时需要)

    # === 输出路径 ===
    'out': 'results/gse160936_microglia/model_robust.pkl',
    'summary_out': 'results/gse160936_microglia/train_summary_robust.json',

    # === 核心算法优化 ===
    'gating_mode': 'bayesian',        # 'standard' 或 'bayesian' (推荐bayesian)
    'use_plateau_lock': True,         # UOT平台锁定 (推荐True)
    'use_soft_penalty': True,         # 软-硬稳定性 (推荐True)
    'use_swa': False,                 # SWA (实验性, 慎用)
    'use_dann': False,                # 域对抗 (强域偏移时启用)
    'use_harmony': True,              # Harmony批次整合 (有批效应时启用)

    # === 🆕 鲁棒性优化 ===
    'use_temperature_scaling': True,  # 温度缩放校准 (跨域场景推荐)
    'temp_scale_target': '5',         # 校准目标阶段 (None=使用late stage)

    'use_batch_decision': True,       # scIB自动决策Harmony (推荐)
    'batch_delta_thresh': 0.03,       # batch ASW降幅阈值
    'bio_drop_thresh': 0.02,          # bio ASW损失容忍

    'use_cross_domain_uot': True,     # 跨域UOT验证 (跨域场景必备)
    'target_h5ad': 'data/raw-use/GSE157783/processed/GSE157783_midbrain_processed.h5ad',
    'target_stage_key': 'diagnosis',  # 目标域阶段列 (None=同源域)
    'target_stages': 'src:Control,dst:PD',  # 目标域阶段映射

    'use_ood_detection': True,        # OOD检测 (跨域场景推荐)
    'ood_percentile': 95.0,           # OOD阈值分位数

    # === 其他配置 ===
    'verbose': True,                  # 详细输出
    'msigdb_gmt': None,               # MSigDB GMT文件路径 (可选, 多个用列表)
    'harmony_backend': 'harmonypy',   # 'harmonypy' 或 'scanpy'
}

# ========================================
# 🚀 预设配置模板 (快速切换场景)
# ========================================

# 场景1: AD微胶质训练 (完整优化)
AD_MICROGLIA_FULL = {
    'h5ad': 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad',
    'stage_key': 'braak_stage',
    'early': '0,1,2', 'mid': '3,4', 'late': '5,6',
    'batch_key': 'region',
    'out': 'results/gse160936/model_full.pkl',
    'summary_out': 'results/gse160936/summary_full.json',
    'gating_mode': 'bayesian',
    'use_plateau_lock': True,
    'use_soft_penalty': True,
    'use_temperature_scaling': True,
    'use_batch_decision': True,
    'use_ood_detection': True,
    'verbose': True,
}

# 场景2: AD→PD 跨域验证 (10项全开)
AD_TO_PD_FULL = {
    'h5ad': 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad',
    'stage_key': 'braak_stage',
    'early': '0,1,2', 'mid': '3,4', 'late': '5,6',
    'batch_key': 'region',
    'out': 'results/ad_to_pd/model_full.pkl',
    'summary_out': 'results/ad_to_pd/summary_full.json',
    'gating_mode': 'bayesian',
    'use_plateau_lock': True,
    'use_soft_penalty': True,
    'use_temperature_scaling': True,
    'temp_scale_target': '5',
    'use_batch_decision': True,
    'batch_delta_thresh': 0.03,
    'bio_drop_thresh': 0.02,
    'use_cross_domain_uot': True,
    'target_h5ad': 'data/raw-use/GSE157783/processed/GSE157783_midbrain_processed.h5ad',
    'target_stage_key': 'diagnosis',
    'target_stages': 'src:Control,dst:PD',
    'use_ood_detection': True,
    'ood_percentile': 95.0,
    'verbose': True,
}

# 场景3: PBMC衰老训练 (无跨域优化)
PBMC_AGING = {
    'h5ad': 'data/raw-use/GSE213516/processed/GSE213516_pbmc_processed.h5ad',
    'stage_key': 'age',
    'early': '40-49,50-59', 'mid': '60-69', 'late': '70-79,80+',
    'out': 'results/gse213516/model.pkl',
    'summary_out': 'results/gse213516/summary.json',
    'gating_mode': 'bayesian',
    'use_plateau_lock': True,
    'use_soft_penalty': True,
    'verbose': True,
}

# 场景4: 基线训练 (无优化)
BASELINE = {
    'h5ad': 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad',
    'stage_key': 'braak_stage',
    'early': '0,1,2', 'mid': '3,4', 'late': '5,6',
    'out': 'results/baseline/model.pkl',
    'summary_out': 'results/baseline/summary.json',
    'gating_mode': 'standard',
    'verbose': True,
}

# ========================================
# 💡 快速切换配置
# ========================================
# 取消注释下面某一行来切换场景:

# TRAIN_CONFIG = AD_MICROGLIA_FULL   # AD微胶质完整优化
# TRAIN_CONFIG = AD_TO_PD_FULL       # AD→PD跨域验证
# TRAIN_CONFIG = PBMC_AGING          # PBMC衰老训练
# TRAIN_CONFIG = BASELINE            # 基线训练

# ========================================
# 🔧 主程序 (无需修改)
# ========================================

def build_args_from_config(config: dict) -> list[str]:
    """将配置字典转换为命令行参数"""
    args = []

    # 必需参数
    args.extend(['--h5ad', config['h5ad']])
    args.extend(['--stage_key', config['stage_key']])
    args.extend(['--src_stage', config['early']])
    args.extend(['--mid_stage', config['mid']])
    args.extend(['--dst_stage', config['late']])
    args.extend(['--out', config['out']])

    # summary_out (默认值处理)
    if 'summary_out' in config and config['summary_out']:
        args.extend(['--summary_out', config['summary_out']])

    # 可选列
    if config.get('celltype_key'):
        args.extend(['--celltype_key', config['celltype_key']])
    if config.get('celltype'):
        args.extend(['--celltype', config['celltype']])
    if config.get('batch_key'):
        args.extend(['--batch-key', config['batch_key']])
    if config.get('domain_key'):
        args.extend(['--domain-key', config['domain_key']])

    # 门控模式
    if config.get('gating_mode'):
        args.extend(['--gating-mode', config['gating_mode']])

    # 核心优化开关
    if config.get('use_plateau_lock'):
        args.append('--use-plateau-lock')
    if config.get('use_soft_penalty'):
        args.append('--use-soft-penalty')
    if config.get('use_swa'):
        args.append('--use-swa')
    if config.get('use_dann'):
        args.append('--use-dann')
    if config.get('use_harmony'):
        args.append('--use-harmony')

    # Harmony参数
    if config.get('harmony_backend'):
        args.extend(['--harmony-backend', config['harmony_backend']])

    # 🆕 鲁棒性优化
    if config.get('use_temperature_scaling'):
        args.append('--use-temperature-scaling')
        if config.get('temp_scale_target'):
            args.extend(['--temp-scale-target', config['temp_scale_target']])

    if config.get('use_batch_decision'):
        args.append('--use-batch-decision')
        if 'batch_delta_thresh' in config:
            args.extend(['--batch-delta-thresh', str(config['batch_delta_thresh'])])
        if 'bio_drop_thresh' in config:
            args.extend(['--bio-drop-thresh', str(config['bio_drop_thresh'])])

    if config.get('use_cross_domain_uot'):
        args.append('--use-cross-domain-uot')
        if config.get('target_h5ad'):
            args.extend(['--target-h5ad', config['target_h5ad']])
        if config.get('target_stage_key'):
            args.extend(['--target-stage-key', config['target_stage_key']])
        if config.get('target_stages'):
            args.extend(['--target-stages', config['target_stages']])

    if config.get('use_ood_detection'):
        args.append('--use-ood-detection')
        if 'ood_percentile' in config:
            args.extend(['--ood-percentile', str(config['ood_percentile'])])

    # 其他选项
    if config.get('verbose'):
        args.append('--verbose')

    if config.get('msigdb_gmt'):
        gmt_files = config['msigdb_gmt']
        if isinstance(gmt_files, str):
            gmt_files = [gmt_files]
        for gmt in gmt_files:
            args.extend(['--msigdb_gmt', gmt])

    return args


def print_config_summary(config: dict):
    """打印配置摘要"""
    print("=" * 70)
    print("🚀 CT-OTS-U 训练配置")
    print("=" * 70)

    print("\n📁 数据配置:")
    print(f"  数据集: {config['h5ad']}")
    print(f"  阶段列: {config['stage_key']}")
    print(f"  阶段划分: early={config['early']}, mid={config['mid']}, late={config['late']}")

    print("\n💾 输出配置:")
    print(f"  模型: {config['out']}")
    print(f"  总结: {config.get('summary_out', '自动生成')}")

    print("\n⚙️  核心优化:")
    optimizations = []
    if config.get('gating_mode') == 'bayesian':
        optimizations.append("BGMM门控")
    if config.get('use_plateau_lock'):
        optimizations.append("UOT平台锁定")
    if config.get('use_soft_penalty'):
        optimizations.append("软-硬稳定性")
    if config.get('use_swa'):
        optimizations.append("SWA (实验性)")
    if config.get('use_dann'):
        optimizations.append("DANN域对抗")
    if config.get('use_harmony'):
        optimizations.append("Harmony批次整合")
    print(f"  {', '.join(optimizations) if optimizations else '无'}")

    print("\n🛡️  鲁棒性优化:")
    robust_opts = []
    if config.get('use_temperature_scaling'):
        target = config.get('temp_scale_target', 'late stage')
        robust_opts.append(f"温度缩放 (target={target})")
    if config.get('use_batch_decision'):
        robust_opts.append(f"批效应决策 (batch_Δ≥{config.get('batch_delta_thresh', 0.03)})")
    if config.get('use_cross_domain_uot'):
        robust_opts.append(f"跨域UOT (target={Path(config.get('target_h5ad', '')).name})")
    if config.get('use_ood_detection'):
        robust_opts.append(f"OOD检测 (p={config.get('ood_percentile', 95)}%)")
    print(f"  {', '.join(robust_opts) if robust_opts else '无'}")

    print("\n" + "=" * 70)
    print()


def main():
    """主函数"""
    # 🖥️ GPU自动检测
    print("\n" + "=" * 70)
    print("🖥️  设备检测")
    print("=" * 70)
    from ct_ots_u.utils.device_detect import detect_best_device
    device, backend, use_torch = detect_best_device(verbose=True)
    print("=" * 70 + "\n")

    # 打印配置摘要
    print_config_summary(TRAIN_CONFIG)

    # 确认执行
    print("按 Enter 键开始训练，或 Ctrl+C 取消...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n训练已取消")
        return

    # 构建参数
    args = build_args_from_config(TRAIN_CONFIG)

    # 导入训练模块
    from ct_ots_u.scripts import run_train

    # 修改sys.argv并执行
    original_argv = sys.argv.copy()
    sys.argv = ['run_train.py'] + args

    try:
        run_train.main()
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
