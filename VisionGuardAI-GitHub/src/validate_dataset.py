"""Validate flat YOLO detection datasets without loading a model."""
import argparse
from collections import Counter
import math
import re
from pathlib import Path
from common import DATASET, SPLITS, CLASSES, IMAGE_EXTENSIONS, files_in

def check_label(path):
    counts, errors, warnings = Counter(), [], []
    try:
        rows = path.read_text(encoding='utf-8-sig').splitlines()
    except (OSError, UnicodeError) as exc:
        return counts, [f'{path}: cannot read label: {exc}'], warnings
    if not rows or not any(row.strip() for row in rows):
        return counts, errors, [f'{path}: empty label; valid only for an intentional background image.']
    for number, row in enumerate(rows, 1):
        parts = row.split()
        prefix = f'{path}:{number}'
        if len(parts) != 5 or not re.fullmatch(r'[012]', parts[0]):
            errors.append(f'{prefix}: expected class ID 0, 1, or 2 and four coordinates.')
            continue
        try:
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f'{prefix}: nonnumeric coordinates.')
            continue
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, w, h)) or w == 0 or h == 0:
            errors.append(f'{prefix}: coordinates must be finite and in [0, 1]; width/height must be positive.')
            continue
        if x-w/2 < -1e-6 or x+w/2 > 1+1e-6 or y-h/2 < -1e-6 or y+h/2 > 1+1e-6:
            warnings.append(f'{prefix}: box extends beyond image boundaries.')
        counts[int(parts[0])] += 1
    return counts, errors, warnings

def validate_dataset(root=DATASET):
    report = {'errors': [], 'warnings': [], 'splits': {}}
    for split in SPLITS:
        images = files_in(root / 'images' / split, IMAGE_EXTENSIONS)
        labels = files_in(root / 'labels' / split, {'.txt'})
        for folder in (root / 'images' / split, root / 'labels' / split):
            if not folder.is_dir():
                report['warnings'].append(f'Missing folder: {folder}')
            elif any(p.is_dir() for p in folder.iterdir()):
                report['errors'].append(f'{folder}: use flat folders; nested folders are unsupported.')
        stems = Counter(p.stem for p in images)
        label_stems = {p.stem for p in labels}
        for stem, count in stems.items():
            if count > 1:
                report['errors'].append(f'{split}: ambiguous image basename {stem}.')
            if stem not in label_stems:
                report['errors'].append(f'{split}: missing label for {stem}.')
        for stem in label_stems - stems.keys():
            report['errors'].append(f'{split}: orphan label {stem}.txt.')
        distribution = Counter()
        for label in labels:
            counts, errors, warnings = check_label(label)
            distribution.update(counts)
            report['errors'].extend(errors)
            report['warnings'].extend(warnings)
        if not images:
            report['warnings'].append(f'{split}: no images yet.')
        report['splits'][split] = {'images': len(images), 'labels': len(labels), 'classes': distribution}
    return report

def print_report(report):
    for split, stats in report['splits'].items():
        print(f"{split}: {stats['images']} images, {stats['labels']} labels")
        print('  Objects: ' + ', '.join(f"{name}={stats['classes'][cid]}" for cid, name in CLASSES.items()))
    for severity in ('warnings', 'errors'):
        for message in report[severity]:
            print(f'{severity.upper()}: {message}')
    print(f"Summary: {len(report['errors'])} errors, {len(report['warnings'])} warnings.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    args = parser.parse_args()
    try:
        report = validate_dataset(args.dataset.resolve())
        print_report(report)
        return 1 if report['errors'] else 0
    except OSError as exc:
        print(f'Cannot inspect dataset: {exc}')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
