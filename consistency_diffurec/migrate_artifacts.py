"""
Миграция старых артефактов (hp_selection0) к новому формату.

Что делает:
  1. Проходит по всем подпапкам в {ARTIFACTS_ROOT}/<dataset>/
  2. В каждой обновляет run_name в summary.json и config.json так, чтобы
     он совпадал с именем папки (новый формат: hpsel_seed{N}_beta{B}_tau{T}_lr{L}).
  3. Переписывает artifact_paths в summary.json на актуальные Drive-пути.
  4. Опционально переименовывает старые val-CSV файлы вида
     "{stem}.csv.val_lr{L}.csv" в новый формат "{stem}_lr{L}.csv.val.csv".

Идемпотентен — безопасно запускать несколько раз. Файлы где run_name уже
совпадает с именем папки просто пропускаются.

Использование (в Colab-ячейке):
    !python migrate_artifacts.py --dataset toys

Или сухой прогон без записи:
    !python migrate_artifacts.py --dataset toys --dry_run
"""
import argparse
import json
import os
import re
from pathlib import Path

DRIVE_BASE = '/content/drive/MyDrive/consistency_diffurec'
ARTIFACTS_ROOT = f'{DRIVE_BASE}/hp_selection/artifacts'
LOGS_ROOT = f'{DRIVE_BASE}/hp_selection/logs'

# Старый формат val-CSV: hpsel_seed{N}_beta{B}_tau{T}.csv.val_lr{LR}.csv
OLD_VAL_CSV_RE = re.compile(
    r'^(hpsel_seed\d+_beta[\d.]+_tau[\d.]+)\.csv\.val_lr([\d.]+)\.csv$'
)


def migrate_run_dir(run_dir: Path, dataset: str, dry_run: bool):
    """Обновляет JSON-файлы внутри одной run-директории."""
    new_run_name = run_dir.name
    summary_path = run_dir / 'summary.json'
    config_path = run_dir / 'config.json'

    changes = []

    # ----- summary.json -----
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        old_run_name = summary.get('run_name', '')
        if old_run_name != new_run_name:
            summary['run_name'] = new_run_name
            summary['artifact_paths'] = {
                'student_checkpoint': str(run_dir / 'student_final.pt'),
                'test_predictions':   str(run_dir / 'test_predictions_nfe1.npz'),
                'config':             str(config_path),
                'loss_csv':           f'{LOGS_ROOT}/{dataset}/{new_run_name}.csv',
                'val_csv':            f'{LOGS_ROOT}/{dataset}/{new_run_name}.csv.val.csv',
            }
            changes.append(f'summary.json: run_name {old_run_name!r} -> {new_run_name!r}')
            if not dry_run:
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)

    # ----- config.json -----
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)

        old_run_name = cfg.get('run_name', '')
        if old_run_name != new_run_name:
            cfg['run_name'] = new_run_name
            changes.append(f'config.json:  run_name {old_run_name!r} -> {new_run_name!r}')
            if not dry_run:
                with open(config_path, 'w') as f:
                    json.dump(cfg, f, indent=2)

    return changes


def migrate_logs_dir(logs_dataset_dir: Path, dry_run: bool):
    """Переименовывает старые val-CSV файлы к новому формату."""
    if not logs_dataset_dir.exists():
        return []
    renames = []
    for f in logs_dataset_dir.iterdir():
        m = OLD_VAL_CSV_RE.match(f.name)
        if not m:
            continue
        stem, lr = m.group(1), m.group(2)
        new_name = f'{stem}_lr{lr}.csv.val.csv'
        new_path = logs_dataset_dir / new_name
        if new_path.exists():
            renames.append(f'  [skip] target already exists: {new_name}')
            continue
        renames.append(f'  rename {f.name} -> {new_name}')
        if not dry_run:
            f.rename(new_path)
    return renames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='toys',
                   help='Имя датасета (подпапка под artifacts/).')
    p.add_argument('--dry_run', action='store_true',
                   help='Только показать, что будет сделано, без записи.')
    p.add_argument('--skip_csv_rename', action='store_true',
                   help='Не переименовывать val-CSV файлы.')
    args = p.parse_args()

    dataset_dir = Path(ARTIFACTS_ROOT) / args.dataset
    if not dataset_dir.exists():
        raise SystemExit(f'Нет такой папки: {dataset_dir}')

    mode = '[DRY RUN] ' if args.dry_run else ''
    print(f'{mode}Миграция артефактов в {dataset_dir}')
    print('=' * 70)

    total_dirs = 0
    total_changed = 0
    for run_dir in sorted(dataset_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == 'teacher':
            continue
        total_dirs += 1
        changes = migrate_run_dir(run_dir, args.dataset, args.dry_run)
        if changes:
            total_changed += 1
            print(f'\n[{run_dir.name}]')
            for c in changes:
                print(f'  {c}')

    print(f'\n{mode}Обновлено директорий: {total_changed}/{total_dirs}')

    # --- Переименование старых val-CSV ---
    if not args.skip_csv_rename:
        logs_dataset = Path(LOGS_ROOT) / args.dataset
        print(f'\n{mode}Старые val-CSV в {logs_dataset}')
        print('=' * 70)
        renames = migrate_logs_dir(logs_dataset, args.dry_run)
        if renames:
            for r in renames:
                print(r)
        else:
            print('  Нечего переименовывать.')


if __name__ == '__main__':
    main()