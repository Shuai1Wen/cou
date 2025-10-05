#!/usr/bin/env python3
"""CT-OTS-U full pipeline orchestrator.

This utility mirrors the VS Code one-click workflow while staying aligned
with the latest training CLI. It runs model training followed by the full
analysis suite (UOT counterfactuals, semigroup checks, GSEA, etc.).

Usage: review the configuration block below, then run `python run_full_pipeline.py`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# =============================================================================
# Configuration (edit to suit your environment)
# =============================================================================

# --- Data paths -----------------------------------------------------------------
DATA_MICROGLIA = Path('data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad')
DATA_PD = Path('data/raw-use/GSE157783/processed/GSE157783_midbrain_processed.h5ad')
DATA_PBMC = Path('data/raw-use/GSE213516/processed/GSE213516_pbmc_processed.h5ad')

# --- Stage configuration --------------------------------------------------------
MICROGLIA_STAGE_KEY = 'braak_stage'
MICROGLIA_EARLY = '0,1,2'
MICROGLIA_MID = '3,4'
MICROGLIA_LATE = '5,6'
MICROGLIA_BATCH_KEY = 'region'
MICROGLIA_SEX_KEY = 'sex'

PD_STAGE_KEY = 'diagnosis'
PD_CONTROL = 'Control'
PD_CASE = 'PD'

PBMC_AGE_KEY = 'age'
PBMC_YOUNG_MAX = 40
PBMC_OLD_MIN = 70

# --- Output locations -----------------------------------------------------------
OUTPUT_DIR = Path('results/full_pipeline')
MODEL_OUTPUT = OUTPUT_DIR / 'train_model.json'
SUMMARY_OUTPUT = OUTPUT_DIR / 'train_summary.json'
ANALYSIS_DIR = OUTPUT_DIR / 'analysis'

# --- External resources ---------------------------------------------------------
MSIGDB_GMT = Path('resources/msigdb/h.all.v2023.1.Hs.symbols.gmt')  # set to None to skip

# --- Training knobs -------------------------------------------------------------
USE_BGMM_GATING = True
USE_PLATEAU_LOCK = False
USE_SOFT_PENALTY = True
USE_HARMONY = True
USE_BATCH_DECISION = True
USE_TEMPERATURE = True
USE_CROSS_DOMAIN = True
USE_OOD_DETECTION = True
USE_TORCH = True
TORCH_DEVICE = 'cpu'
USE_TORCH_COMPILE = True
USE_SINKHORN_MINIBATCH = True
SINKHORN_BATCH_SIZE: int | None = 512
SINKHORN_BACKEND: str | None = 'auto'
PCT_MT_LIMIT: float | None = None
USE_SWA = True

# --- Analysis tasks -------------------------------------------------------------
RUN_UOT_COUNTERFACTUAL = True
RUN_SEMIGROUP_CONSISTENCY = True
RUN_RANK_STABILITY = True
RUN_CROSS_DATASET = True
RUN_GSEA = True
RUN_MLOY = True
RUN_PBMC_AGING = True

# =============================================================================
# Helpers
# =============================================================================

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / 'scripts' / 'analysis'


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def print_section(title: str) -> None:
    print('\n' + '=' * 70)
    print(f"[*] {title}")
    print('=' * 70)


def run_command(cmd: List[str], description: str) -> bool:
    print(f"\n[>] {description}")
    print('    ' + ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print('[OK] 完成')
        return True
    except subprocess.CalledProcessError as exc:  # noqa: PERF203 - explicit logging
        print(f"[ERROR] 失败: {exc}")
        return False


def check_file_exists(path: Path) -> bool:
    if path.exists():
        return True
    print(f"[WARN] 文件不存在: {path}")
    return False


def find_gmt_file() -> Path | None:
    gmt_dir = ROOT / 'resources' / 'msigdb'
    if not gmt_dir.exists():
        return None
    candidates = sorted(gmt_dir.glob('*.gmt'))
    return candidates[0] if candidates else None


def build_training_command() -> List[str]:
    cmd = [
        sys.executable,
        '-m',
        'ct_ots_u.scripts.run_train',
        '--h5ad', str(DATA_MICROGLIA),
        '--stage_key', MICROGLIA_STAGE_KEY,
        '--src_stage', MICROGLIA_EARLY,
        '--mid_stage', MICROGLIA_MID,
        '--dst_stage', MICROGLIA_LATE,
        '--out', str(MODEL_OUTPUT),
        '--summary_out', str(SUMMARY_OUTPUT),
        '--verbose',
    ]

    if PCT_MT_LIMIT is not None:
        cmd.extend(['--pct-mt-limit', str(PCT_MT_LIMIT)])
    if MICROGLIA_BATCH_KEY:
        cmd.extend(['--batch-key', MICROGLIA_BATCH_KEY])

    if USE_BGMM_GATING:
        cmd.extend(['--gating-mode', 'bayesian'])
    if USE_PLATEAU_LOCK:
        cmd.append('--use-plateau-lock')
    if USE_SOFT_PENALTY:
        cmd.append('--use-soft-penalty')
    if USE_HARMONY:
        cmd.append('--use-harmony')
    if USE_BATCH_DECISION:
        cmd.append('--use-batch-decision')
    if USE_TEMPERATURE:
        cmd.append('--use-temperature-scaling')
    if USE_SWA:
        cmd.append('--use-swa')
    if USE_TORCH:
        cmd.append('--use-torch')
        if TORCH_DEVICE:
            cmd.extend(['--torch-device', TORCH_DEVICE])
        if USE_TORCH_COMPILE:
            cmd.append('--torch-compile')
    if USE_SINKHORN_MINIBATCH:
        cmd.append('--sinkhorn-minibatch')
        if SINKHORN_BATCH_SIZE:
            cmd.extend(['--sinkhorn-batch-size', str(SINKHORN_BATCH_SIZE)])
    if SINKHORN_BACKEND:
        cmd.extend(['--sinkhorn-backend', SINKHORN_BACKEND])

    gmt_candidates = []
    if MSIGDB_GMT and Path(MSIGDB_GMT).exists():
        gmt_candidates.append(Path(MSIGDB_GMT))
    elif MSIGDB_GMT is None:
        auto = find_gmt_file()
        if auto:
            gmt_candidates.append(auto)
    if gmt_candidates:
        for gmt in gmt_candidates:
            cmd.extend(['--msigdb_gmt', str(gmt)])

    if USE_CROSS_DOMAIN and DATA_PD.exists():
        target_stage_map = json.dumps({
            'early': [PD_CONTROL],
            'mid': [PD_CONTROL],
            'late': [PD_CASE],
        }, ensure_ascii=False)
        cmd.append('--use-cross-domain-uot')
        cmd.extend(['--target-h5ad', str(DATA_PD)])
        cmd.extend(['--target-stage-key', PD_STAGE_KEY])
        cmd.extend(['--target-stage-map', target_stage_map])
    if USE_OOD_DETECTION:
        cmd.append('--use-ood-detection')

    return cmd


# =============================================================================
# Step 1: Training
# =============================================================================

def step1_training() -> bool:
    print_section('步骤 1/2: 训练 CT-OTS-U 模型')

    if not check_file_exists(DATA_MICROGLIA):
        print('[ERROR] 微胶质数据不存在，终止流程')
        return False

    ensure_dir(OUTPUT_DIR)

    cmd = build_training_command()

    print(f"\n[DATA] 输入数据: {DATA_MICROGLIA}")
    print(f"[DATA] 阶段: early={MICROGLIA_EARLY} | mid={MICROGLIA_MID} | late={MICROGLIA_LATE}")

    enabled_opts = []
    if USE_BGMM_GATING:
        enabled_opts.append('BGMM')
    if USE_PLATEAU_LOCK:
        enabled_opts.append('UOT plateau')
    if USE_SOFT_PENALTY:
        enabled_opts.append('soft stability')
    if USE_HARMONY:
        enabled_opts.append('Harmony')
    if USE_BATCH_DECISION:
        enabled_opts.append('auto-batch')
    if USE_TEMPERATURE:
        enabled_opts.append('temperature')
    if USE_SWA:
        enabled_opts.append('SWA')
    if USE_TORCH:
        enabled_opts.append('PyTorch')
    if USE_CROSS_DOMAIN:
        enabled_opts.append('cross-domain UOT')
    if USE_OOD_DETECTION:
        enabled_opts.append('OOD guard')

    print(f"[OPT] 启用: {', '.join(enabled_opts) if enabled_opts else '无'}")

    success = run_command(cmd, '训练模型')
    if success:
        print(f"\n[OK] 训练完成 -> 模型: {MODEL_OUTPUT}")
    return success


# =============================================================================
# Step 2: Analysis
# =============================================================================

def step2_analysis() -> bool:
    print_section('步骤 2/2: 分析与报告')

    ensure_dir(ANALYSIS_DIR)

    results: Dict[str, Path] = {}
    success_count = 0
    total_count = 0

    summary_exists = check_file_exists(SUMMARY_OUTPUT)

    # 1. UOT counterfactuals
    if RUN_UOT_COUNTERFACTUAL and summary_exists:
        total_count += 1
        out_json = ANALYSIS_DIR / 'uot_counterfactual.json'
        cmd = [
            sys.executable, str(SCRIPTS_DIR / 'eval_uot_counterfactual.py'),
            '--train-summary', str(SUMMARY_OUTPUT),
            '--out', str(out_json),
        ]
        if run_command(cmd, '1/7 UOT 反事实分析'):
            results['uot_counterfactual'] = out_json
            success_count += 1

    # 2. Semigroup consistency
    if RUN_SEMIGROUP_CONSISTENCY and summary_exists:
        total_count += 1
        out_json = ANALYSIS_DIR / 'semigroup_consistency.json'
        cmd = [
            sys.executable, str(SCRIPTS_DIR / 'eval_semigroup_consistency.py'),
            '--train-summary', str(SUMMARY_OUTPUT),
            '--out', str(out_json),
        ]
        if run_command(cmd, '2/7 半群一致性分析'):
            results['semigroup_consistency'] = out_json
            success_count += 1

    # 3. Rank stability
    if RUN_RANK_STABILITY and summary_exists:
        total_count += 1
        out_json = ANALYSIS_DIR / 'rank_stability.json'
        cmd = [
            sys.executable, str(SCRIPTS_DIR / 'eval_rank_stability.py'),
            '--train-summary', str(SUMMARY_OUTPUT),
            '--out', str(out_json),
        ]
        if run_command(cmd, '3/7 Rank 稳定性分析'):
            results['rank_stability'] = out_json
            success_count += 1

    # 4. Cross dataset transfer
    if RUN_CROSS_DATASET and check_file_exists(DATA_MICROGLIA) and check_file_exists(DATA_PD):
        total_count += 1
        out_json = ANALYSIS_DIR / 'cross_dataset_ad_pd.json'
        cmd = [
            sys.executable, str(SCRIPTS_DIR / 'eval_cross_dataset.py'),
            '--src', str(DATA_MICROGLIA),
            '--src-stage-key', MICROGLIA_STAGE_KEY,
            '--src-early', MICROGLIA_EARLY,
            '--src-late', MICROGLIA_LATE,
            '--tgt', str(DATA_PD),
            '--tgt-stage-key', PD_STAGE_KEY,
            '--tgt-control', PD_CONTROL,
            '--tgt-case', PD_CASE,
            '--n', '2000',
            '--out', str(out_json),
        ]
        if run_command(cmd, '4/7 AD→PD 跨数据集验证'):
            results['cross_dataset'] = out_json
            success_count += 1

    # 5. GSEA enrichment
    if RUN_GSEA and check_file_exists(DATA_MICROGLIA):
        total_count += 1
        gmt_path = None
        if MSIGDB_GMT and Path(MSIGDB_GMT).exists():
            gmt_path = Path(MSIGDB_GMT)
        elif not MSIGDB_GMT:
            gmt_path = find_gmt_file()
        if gmt_path:
            out_json = ANALYSIS_DIR / 'gsea_traj_microglia.json'
            cmd = [
                sys.executable, str(SCRIPTS_DIR / 'biol_gsea_traj.py'),
                '--h5ad', str(DATA_MICROGLIA),
                '--stage-key', MICROGLIA_STAGE_KEY,
                '--early', MICROGLIA_EARLY,
                '--late', MICROGLIA_LATE,
                '--gene-col', 'gene_symbols',
                '--gmt', str(gmt_path),
                '--out-json', str(out_json),
            ]
            if run_command(cmd, '5/7 GSEA 轨迹富集分析'):
                results['gsea'] = out_json
                success_count += 1
        else:
            print(f"[WARN] 未找到 GMT 文件 ({MSIGDB_GMT or 'resources/msigdb/*.gmt'})，跳过 GSEA")

    # 6. mLOY external validation
    if RUN_MLOY and check_file_exists(DATA_MICROGLIA):
        total_count += 1
        out_json = ANALYSIS_DIR / 'mloy_external_microglia.json'
        cmd = [
            sys.executable, str(SCRIPTS_DIR / 'biol_mloy_external.py'),
            '--h5ad', str(DATA_MICROGLIA),
            '--sex-key', MICROGLIA_SEX_KEY,
            '--out', str(out_json),
        ]
        if run_command(cmd, '6/7 mLOY 外部验证'):
            results['mloy'] = out_json
            success_count += 1

    # 7. PBMC virtual ageing
    if RUN_PBMC_AGING and check_file_exists(DATA_PBMC):
        total_count += 1
        out_json = ANALYSIS_DIR / 'pbmc_virtual_aging.json'
        cmd = [
            sys.executable, str(SCRIPTS_DIR / 'pbmc_virtual_aging.py'),
            '--h5ad', str(DATA_PBMC),
            '--age-key', PBMC_AGE_KEY,
            '--young-max', str(PBMC_YOUNG_MAX),
            '--old-min', str(PBMC_OLD_MIN),
            '--n', '2000',
            '--out', str(out_json),
        ]
        if run_command(cmd, '7/7 PBMC 虚拟衰老分析'):
            results['pbmc_aging'] = out_json
            success_count += 1

    if results:
        report_path = ANALYSIS_DIR / 'summary_report.md'
        lines: List[str] = ['# CT-OTS-U 完整流程分析报告', '']
        for key, output in results.items():
            if output.exists():
                data = json.loads(output.read_text(encoding='utf-8'))
                lines.append(f"## {key.replace('_', ' ').title()}")
                lines.append('')
                for k, v in data.items():
                    if isinstance(v, (int, float, str)) or v is None:
                        lines.append(f"- **{k}**: {v}")
                    elif isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, (int, float, str)) or sub_v is None:
                                lines.append(f"  - **{k}.{sub_k}**: {sub_v}")
                lines.append('')
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        print(f"\n[REPORT] 综合报告生成: {report_path}")

    print(f"\n[OK] 分析完成: {success_count}/{total_count} 任务成功")
    return success_count == total_count if total_count else True


# =============================================================================
# Main entry
# =============================================================================


def main() -> None:
    print('=' * 70)
    print('CT-OTS-U 完整流程')
    print('=' * 70)

    # 🖥️ GPU自动检测
    print('\n' + '=' * 70)
    print('🖥️  设备检测')
    print('=' * 70)
    from ct_ots_u.utils.device_detect import detect_best_device
    device, backend, use_torch = detect_best_device(verbose=True)
    print('=' * 70)

    # 如果检测到更好的设备，覆盖默认配置
    global USE_TORCH, TORCH_DEVICE
    if use_torch and not USE_TORCH:
        print(f"\n💡 检测到可用加速，自动启用 use_torch={use_torch}, device={device}")
        USE_TORCH = use_torch
        TORCH_DEVICE = device

    print(f"\n[OUTPUT] 输出目录: {OUTPUT_DIR}")
    print('[DATA] 数据列表:')
    print(f"   - Microglia: {DATA_MICROGLIA}")
    print(f"   - PD: {DATA_PD}")
    print(f"   - PBMC: {DATA_PBMC}")

    trained = step1_training()
    analysed = False
    if trained:
        analysed = step2_analysis()
    else:
        print('\n[WARN] 训练失败，跳过后续分析')

    print('\n' + '=' * 70)
    print('完整流程结束')
    print('=' * 70)

    if trained and analysed:
        print('\n[OK] 所有步骤成功完成')
        print(f"   模型: {MODEL_OUTPUT}")
        print(f"   训练总结: {SUMMARY_OUTPUT}")
        print(f"   分析报告: {ANALYSIS_DIR / 'summary_report.md'}")
    elif trained:
        print('\n[WARN] 训练成功但部分分析失败，请检查日志')
    else:
        print('\n[ERROR] 训练步骤失败，未能产生结果')


if __name__ == '__main__':
    main()

