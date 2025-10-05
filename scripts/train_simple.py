"""
CT-OTS-U 简易训练脚本 - VS一键运行版本

🎯 使用说明:
1. 修改下方 "配置区" 的参数
2. 在VS中直接 F5 运行
3. 无需任何命令行操作

💡 提示:
- True/False 控制开关
- 字符串用引号 'xxx'
- 不需要的参数设为 None
"""

# ========================================
# 📋 配置区 - 在这里修改所有参数
# ========================================

# --- 数据文件路径 ---
DATA_FILE = 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad'
STAGE_COLUMN = 'braak_stage'  # 阶段列名

# --- 阶段划分 ---
EARLY_STAGES = '0,1,2'  # 早期阶段 (逗号分隔)
MID_STAGES = '3,4'      # 中期阶段
LATE_STAGES = '5,6'     # 晚期阶段

# --- 可选列 ---
BATCH_COLUMN = 'region'  # 批次列 (None=不使用)
DOMAIN_COLUMN = None     # 域列 (None=不使用, 例如: 'sex')

# --- 输出路径 ---
OUTPUT_MODEL = 'results/my_model.pkl'
OUTPUT_SUMMARY = 'results/my_summary.json'

# ========================================
# ⚙️ 优化开关 - True=启用, False=关闭
# ========================================

# --- 核心优化 (推荐全部True) ---
USE_BGMM_GATING = True        # BGMM自适应门控
USE_UOT_PLATEAU = True        # UOT平台锁定
USE_STABILITY = True          # 软-硬稳定性

# --- 批次和域优化 ---
USE_HARMONY = True            # Harmony批次整合 (需要BATCH_COLUMN)
USE_DANN = False              # 域对抗 (需要DOMAIN_COLUMN)

# --- 🆕 鲁棒性优化 (跨域场景推荐) ---
USE_TEMPERATURE = True        # 温度缩放校准
TEMP_TARGET_STAGE = '5'       # 温度校准目标阶段 (None=自动)

USE_BATCH_DECISION = True     # scIB自动决策Harmony
BATCH_THRESHOLD = 0.03        # batch ASW降幅阈值
BIO_THRESHOLD = 0.02          # bio ASW损失容忍

USE_CROSS_DOMAIN = True       # 跨域UOT验证
TARGET_DATA = 'data/raw-use/GSE157783/processed/GSE157783_midbrain_processed.h5ad'
TARGET_STAGE_COL = 'diagnosis'
TARGET_STAGE_MAP = 'src:Control,dst:PD'  # 源:目标映射

USE_OOD_DETECTION = True      # OOD检测
OOD_THRESHOLD = 95.0          # OOD阈值百分位数

# --- 实验性优化 (慎用!) ---
USE_SWA = False               # 随机权重平均 (实验性)

# ========================================
# 🚀 自动执行 - 无需修改下方代码
# ========================================

def main():
    import sys
    from pathlib import Path

    print("=" * 70)
    print("🚀 CT-OTS-U 训练")
    print("=" * 70)

    # 🖥️ GPU自动检测
    print("\n" + "=" * 70)
    print("🖥️  设备检测")
    print("=" * 70)
    from ct_ots_u.utils.device_detect import detect_best_device
    device, backend, use_torch = detect_best_device(verbose=True)
    print("=" * 70)

    # 显示配置
    print(f"\n📁 数据: {DATA_FILE}")
    print(f"📊 阶段: early={EARLY_STAGES}, mid={MID_STAGES}, late={LATE_STAGES}")
    print(f"💾 输出: {OUTPUT_MODEL}")

    # 显示启用的优化
    opts = []
    if USE_BGMM_GATING: opts.append("BGMM门控")
    if USE_UOT_PLATEAU: opts.append("UOT平台")
    if USE_STABILITY: opts.append("稳定性")
    if USE_HARMONY: opts.append("Harmony")
    if USE_DANN: opts.append("DANN")
    if USE_TEMPERATURE: opts.append("温度缩放")
    if USE_BATCH_DECISION: opts.append("批效应决策")
    if USE_CROSS_DOMAIN: opts.append("跨域UOT")
    if USE_OOD_DETECTION: opts.append("OOD检测")
    if USE_SWA: opts.append("SWA")

    print(f"⚙️  优化: {', '.join(opts) if opts else '无'}")
    print("=" * 70)
    print("\n🔄 开始训练...\n")

    # 构建参数
    args = [
        '--h5ad', DATA_FILE,
        '--stage_key', STAGE_COLUMN,
        '--src_stage', EARLY_STAGES,
        '--mid_stage', MID_STAGES,
        '--dst_stage', LATE_STAGES,
        '--out', OUTPUT_MODEL,
        '--summary_out', OUTPUT_SUMMARY,
        '--verbose',
    ]

    # 添加可选列
    if BATCH_COLUMN:
        args.extend(['--batch-key', BATCH_COLUMN])
    if DOMAIN_COLUMN:
        args.extend(['--domain-key', DOMAIN_COLUMN])

    # 添加优化参数
    if USE_BGMM_GATING:
        args.extend(['--gating-mode', 'bayesian'])
    else:
        args.extend(['--gating-mode', 'standard'])

    if USE_UOT_PLATEAU:
        args.append('--use-plateau-lock')
    if USE_STABILITY:
        args.append('--use-soft-penalty')
    if USE_SWA:
        args.append('--use-swa')
    if USE_DANN:
        args.append('--use-dann')
    if USE_HARMONY:
        args.append('--use-harmony')

    # 鲁棒性优化
    if USE_TEMPERATURE:
        args.append('--use-temperature-scaling')
        if TEMP_TARGET_STAGE:
            args.extend(['--temp-scale-target', TEMP_TARGET_STAGE])

    if USE_BATCH_DECISION:
        args.append('--use-batch-decision')
        args.extend(['--batch-delta-thresh', str(BATCH_THRESHOLD)])
        args.extend(['--bio-drop-thresh', str(BIO_THRESHOLD)])

    if USE_CROSS_DOMAIN:
        args.append('--use-cross-domain-uot')
        if TARGET_DATA:
            args.extend(['--target-h5ad', TARGET_DATA])
        if TARGET_STAGE_COL:
            args.extend(['--target-stage-key', TARGET_STAGE_COL])
        if TARGET_STAGE_MAP:
            args.extend(['--target-stages', TARGET_STAGE_MAP])

    if USE_OOD_DETECTION:
        args.append('--use-ood-detection')
        args.extend(['--ood-percentile', str(OOD_THRESHOLD)])

    # 执行训练
    from ct_ots_u.scripts import run_train

    original_argv = sys.argv.copy()
    sys.argv = ['run_train.py'] + args

    try:
        run_train.main()
        print(f"\n✅ 训练完成!")
        print(f"📦 模型: {OUTPUT_MODEL}")
        print(f"📄 总结: {OUTPUT_SUMMARY}")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    main()
