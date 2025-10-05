"""
CT-OTS-U 快速训练脚本

🎯 使用方法:
1. 在下方 CONFIG 区域修改参数
2. 在VS中直接按 F5 或点击运行
3. 无需命令行，所有配置都在代码中

📝 修改训练场景只需改变 SCENARIO 变量
"""

import sys
from pathlib import Path

# ========================================
# 🎯 选择训练场景 (修改这里!)
# ========================================

SCENARIO = 'AD_TO_PD'  # 可选: 'AD_FULL', 'AD_TO_PD', 'PBMC', 'BASELINE'

# ========================================
# ⚙️ 场景配置
# ========================================

DEFAULT_TORCH_COMPILE = True
DEFAULT_SINKHORN_MINIBATCH = True
DEFAULT_SINKHORN_BATCH_SIZE = 512
DEFAULT_SINKHORN_BACKEND = 'auto'

SCENARIOS = {
    # AD微胶质完整优化训练
    'AD_FULL': {
        'name': 'AD微胶质完整优化',
        'data': {
            'h5ad': 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad',
            'stage_key': 'braak_stage',
            'stages': {'early': '0,1,2', 'mid': '3,4', 'late': '5,6'},
            'batch_key': 'region',
        },
        'output': {
            'model': 'results/gse160936/model_full.pkl',
            'summary': 'results/gse160936/summary_full.json',
        },
        'optimizations': {
            'gating_mode': 'bayesian',
            'use_plateau_lock': True,
            'use_soft_penalty': True,
            'use_temperature_scaling': True,
            'use_batch_decision': True,
            'use_ood_detection': True,
        },
    },

    # AD→PD 跨域验证 (10项全开)
    'AD_TO_PD': {
        'name': 'AD→PD跨域验证 (10项优化)',
        'data': {
            'h5ad': 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad',
            'stage_key': 'braak_stage',
            'stages': {'early': '0,1,2', 'mid': '3,4', 'late': '5,6'},
            'batch_key': 'region',
        },
        'output': {
            'model': 'results/ad_to_pd/model_full.pkl',
            'summary': 'results/ad_to_pd/summary_full.json',
        },
        'optimizations': {
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
        },
    },

    # PBMC衰老训练
    'PBMC': {
        'name': 'PBMC衰老训练',
        'data': {
            'h5ad': 'data/raw-use/GSE213516/processed/GSE213516_pbmc_processed.h5ad',
            'stage_key': 'age',
            'stages': {'early': '40-49,50-59', 'mid': '60-69', 'late': '70-79,80+'},
        },
        'output': {
            'model': 'results/gse213516/model.pkl',
            'summary': 'results/gse213516/summary.json',
        },
        'optimizations': {
            'gating_mode': 'bayesian',
            'use_plateau_lock': True,
            'use_soft_penalty': True,
        },
    },

    # 基线训练 (无优化)
    'BASELINE': {
        'name': '基线训练 (无优化)',
        'data': {
            'h5ad': 'data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad',
            'stage_key': 'braak_stage',
            'stages': {'early': '0,1,2', 'mid': '3,4', 'late': '5,6'},
        },
        'output': {
            'model': 'results/baseline/model.pkl',
            'summary': 'results/baseline/summary.json',
        },
        'optimizations': {
            'gating_mode': 'standard',
        },
    },
}

# ========================================
# 🔧 主程序
# ========================================

def run_training():
    """执行训练"""
    if SCENARIO not in SCENARIOS:
        print(f"❌ 错误: 未知场景 '{SCENARIO}'")
        print(f"可用场景: {', '.join(SCENARIOS.keys())}")
        return

    config = SCENARIOS[SCENARIO]

    # 🖥️ GPU自动检测
    from ct_ots_u.utils.device_detect import detect_best_device
    print("\n" + "=" * 70)
    print("🖥️  设备检测")
    print("=" * 70)
    device, backend, use_torch = detect_best_device(verbose=True)
    print("=" * 70)

    # 打印配置
    print("\n" + "=" * 70)
    print(f"🚀 {config['name']}")
    print("=" * 70)
    print(f"\n📁 数据: {config['data']['h5ad']}")
    print(f"📊 阶段: {config['data']['stages']}")
    print(f"💾 输出: {config['output']['model']}")

    opt_names = []
    opts = config['optimizations']
    if opts.get('gating_mode') == 'bayesian':
        opt_names.append("BGMM门控")
    if opts.get('use_plateau_lock'):
        opt_names.append("UOT平台")
    if opts.get('use_soft_penalty'):
        opt_names.append("稳定性")
    if opts.get('use_temperature_scaling'):
        opt_names.append("温度缩放")
    if opts.get('use_batch_decision'):
        opt_names.append("批效应决策")
    if opts.get('use_cross_domain_uot'):
        opt_names.append("跨域UOT")
    if opts.get('use_ood_detection'):
        opt_names.append("OOD检测")

    print(f"⚙️  优化: {', '.join(opt_names) if opt_names else '无'}")
    print("=" * 70)
    print()

    # 构建参数
    torch_compile = bool(config['optimizations'].get('torch_compile', DEFAULT_TORCH_COMPILE))
    sinkhorn_minibatch = bool(config['optimizations'].get('sinkhorn_minibatch', DEFAULT_SINKHORN_MINIBATCH))
    sinkhorn_batch_size = config['optimizations'].get('sinkhorn_batch_size', DEFAULT_SINKHORN_BATCH_SIZE)
    sinkhorn_backend = config['optimizations'].get('sinkhorn_backend', DEFAULT_SINKHORN_BACKEND)
    if sinkhorn_batch_size is not None:
        sinkhorn_batch_size = int(sinkhorn_batch_size)
    args = []

    # 数据参数
    data = config['data']
    args.extend(['--h5ad', data['h5ad']])
    args.extend(['--stage_key', data['stage_key']])
    args.extend(['--src_stage', data['stages']['early']])
    args.extend(['--mid_stage', data['stages']['mid']])
    args.extend(['--dst_stage', data['stages']['late']])

    if 'batch_key' in data:
        args.extend(['--batch-key', data['batch_key']])
    if 'domain_key' in data:
        args.extend(['--domain-key', data['domain_key']])

    # 输出参数
    output = config['output']
    args.extend(['--out', output['model']])
    if 'summary' in output:
        args.extend(['--summary_out', output['summary']])

    # 优化参数
    opts = config['optimizations']

    if 'gating_mode' in opts:
        args.extend(['--gating-mode', opts['gating_mode']])

    if opts.get('use_plateau_lock'):
        args.append('--use-plateau-lock')
    if opts.get('use_soft_penalty'):
        args.append('--use-soft-penalty')
    if opts.get('use_swa'):
        args.append('--use-swa')
    if opts.get('use_dann'):
        args.append('--use-dann')
    if opts.get('use_harmony'):
        args.append('--use-harmony')

    # 鲁棒性优化
    if opts.get('use_temperature_scaling'):
        args.append('--use-temperature-scaling')
        if 'temp_scale_target' in opts:
            args.extend(['--temp-scale-target', opts['temp_scale_target']])

    if opts.get('use_batch_decision'):
        args.append('--use-batch-decision')
        if 'batch_delta_thresh' in opts:
            args.extend(['--batch-delta-thresh', str(opts['batch_delta_thresh'])])
        if 'bio_drop_thresh' in opts:
            args.extend(['--bio-drop-thresh', str(opts['bio_drop_thresh'])])

    if opts.get('use_cross_domain_uot'):
        args.append('--use-cross-domain-uot')
        if 'target_h5ad' in opts:
            args.extend(['--target-h5ad', opts['target_h5ad']])
        if 'target_stage_key' in opts:
            args.extend(['--target-stage-key', opts['target_stage_key']])
        if 'target_stages' in opts:
            args.extend(['--target-stages', opts['target_stages']])

    if opts.get('use_ood_detection'):
        args.append('--use-ood-detection')
        if 'ood_percentile' in opts:
            args.extend(['--ood-percentile', str(opts['ood_percentile'])])

    if use_torch:
        args.append('--use-torch')
        if device:
            args.extend(['--torch-device', device])
        if torch_compile:
            args.append('--torch-compile')
    else:
        if torch_compile:
            args.append('--torch-compile')
    if sinkhorn_minibatch:
        args.append('--sinkhorn-minibatch')
        if sinkhorn_batch_size:
            args.extend(['--sinkhorn-batch-size', str(sinkhorn_batch_size)])
    if sinkhorn_backend:
        args.extend(['--sinkhorn-backend', sinkhorn_backend])

    args.append('--verbose')

    # 执行训练
    from ct_ots_u.scripts import run_train

    original_argv = sys.argv.copy()
    sys.argv = ['run_train.py'] + args

    try:
        run_train.main()
        print(f"\n✅ 训练完成! 结果已保存至: {output['model']}")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.argv = original_argv


if __name__ == '__main__':
    run_training()
