#!/usr/bin/env python3
"""
CT-OTS-U Project Integrity Checker

Run this script to verify that all necessary files and dependencies are present.

Usage:
    python check_project.py
"""

import sys
from pathlib import Path


def check_directories():
    """Check that required directories exist."""
    print("\n" + "="*60)
    print("Checking Directory Structure...")
    print("="*60)

    required_dirs = [
        "ct_ots_u",
        "data/raw-use",
        "scripts",
        "results",
        "resources",
    ]

    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        status = "OK" if path.exists() and path.is_dir() else "MISSING"
        symbol = "[+]" if status == "OK" else "[X]"
        print(f"{symbol} {dir_path:30s} ... {status}")
        if status == "MISSING":
            all_ok = False

    return all_ok


def check_scripts():
    """Check that user scripts exist."""
    print("\n" + "="*60)
    print("Checking User Scripts...")
    print("="*60)

    required_scripts = [
        "scripts/train_simple.py",
        "scripts/train_quick.py",
        "scripts/run_full_pipeline.py",
        "scripts/run_train_config.py",
    ]

    all_ok = True
    for script in required_scripts:
        path = Path(script)
        status = "OK" if path.exists() and path.is_file() else "MISSING"
        symbol = "[+]" if status == "OK" else "[X]"
        size = f"({path.stat().st_size / 1024:.1f} KB)" if status == "OK" else ""
        print(f"{symbol} {script:35s} ... {status} {size}")
        if status == "MISSING":
            all_ok = False

    return all_ok


def check_datasets():
    """Check that datasets exist."""
    print("\n" + "="*60)
    print("Checking Datasets...")
    print("="*60)

    datasets = [
        "data/raw-use/GSE160936/processed/GSE160936_microglia_processed.h5ad",
        "data/raw-use/GSE213516/processed/GSE213516_pbmc_processed.h5ad",
        "data/raw-use/GSE157783/processed/GSE157783_midbrain_processed.h5ad",
    ]

    all_ok = True
    total_size = 0
    for dataset in datasets:
        path = Path(dataset)
        if path.exists() and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"[+] {path.name:45s} ... OK ({size_mb:.1f} MB)")
        else:
            print(f"[X] {path.name:45s} ... MISSING")
            all_ok = False

    if all_ok:
        print(f"\nTotal data size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")

    return all_ok


def check_python_package():
    """Check that ct_ots_u package can be imported."""
    print("\n" + "="*60)
    print("Checking Python Package...")
    print("="*60)

    all_ok = True

    # Add current directory to path
    sys.path.insert(0, str(Path.cwd()))

    tests = [
        ("ct_ots_u", "Core package"),
        ("ct_ots_u.config", "Configuration module"),
        ("ct_ots_u.ct_ots_model", "Model module"),
        ("ct_ots_u.utils.device_detect", "Device detection"),
    ]

    for module_name, description in tests:
        try:
            __import__(module_name)
            print(f"[+] {description:30s} ... OK")
        except ImportError as e:
            print(f"[X] {description:30s} ... FAILED ({e})")
            all_ok = False

    return all_ok


def check_dependencies():
    """Check that required Python packages are installed."""
    print("\n" + "="*60)
    print("Checking Dependencies...")
    print("="*60)

    required_packages = [
        ("numpy", "Scientific computing"),
        ("scipy", "Scientific computing"),
        ("pandas", "Data manipulation"),
        ("scanpy", "Single-cell analysis"),
        ("anndata", "Annotated data"),
        ("sklearn", "Machine learning"),
        ("ot", "Optimal transport (POT)"),
    ]

    optional_packages = [
        ("torch", "PyTorch (GPU acceleration)"),
        ("geomloss", "GeomLoss (GPU acceleration)"),
        ("matplotlib", "Visualization"),
        ("seaborn", "Visualization"),
    ]

    print("\nRequired packages:")
    all_ok = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"[+] {package:15s} ... OK ({description})")
        except ImportError:
            print(f"[X] {package:15s} ... MISSING ({description})")
            all_ok = False

    print("\nOptional packages:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"[+] {package:15s} ... OK ({description})")
        except ImportError:
            print(f"[-] {package:15s} ... Not installed ({description})")

    return all_ok


def check_documentation():
    """Check that documentation files exist."""
    print("\n" + "="*60)
    print("Checking Documentation...")
    print("="*60)

    docs = [
        ("README.md", "Main documentation"),
        ("QUICK_START.md", "Quick start guide"),
        ("requirements.txt", "Python dependencies"),
        ("PROJECT_MANIFEST.txt", "Project manifest"),
    ]

    all_ok = True
    for filename, description in docs:
        path = Path(filename)
        if path.exists() and path.is_file():
            lines = len(path.read_text(encoding='utf-8').splitlines())
            print(f"[+] {filename:25s} ... OK ({lines} lines)")
        else:
            print(f"[X] {filename:25s} ... MISSING")
            all_ok = False

    return all_ok


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("CT-OTS-U Project Integrity Check")
    print("="*60)

    checks = [
        ("Directories", check_directories),
        ("User Scripts", check_scripts),
        ("Datasets", check_datasets),
        ("Python Package", check_python_package),
        ("Dependencies", check_dependencies),
        ("Documentation", check_documentation),
    ]

    results = {}
    for name, check_func in checks:
        results[name] = check_func()

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "[+]" if passed else "[X]"
        print(f"{symbol} {name:20s} ... {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("Project integrity check: PASSED")
        print("\nYou are ready to run CT-OTS-U!")
        print("Try: python scripts/train_simple.py")
    else:
        print("Project integrity check: FAILED")
        print("\nSome components are missing. Please check the errors above.")
        print("You may need to:")
        print("  1. Install missing dependencies: pip install -r requirements.txt")
        print("  2. Check that data files are present")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
