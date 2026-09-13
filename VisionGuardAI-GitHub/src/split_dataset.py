
import argparse
import math
import random
import shutil
from common import DATASET, SPLITS, IMAGE_EXTENSIONS, files_in, project_path
from validate_dataset import check_label

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratios', nargs=3, type=float, default=(.80, .15, .05), metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--source', help='Folder containing unsplit images/ and labels/; relative to project root.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if any(not math.isfinite(r) or r < 0 for r in args.ratios) or not math.isclose(sum(args.ratios), 1, abs_tol=1e-9):
        parser.error('Three nonnegative finite ratios must sum to 1.')
    try:
        if not args.source:
            print('For an already split dataset, run src/validate_dataset.py. To split unsplit pairs, use --source PATH containing images/ and labels/.')
            return 0
        source_root = project_path(args.source)
        for kind in ('images', 'labels'):
            if not (source_root / kind).is_dir():
                raise ValueError(f'Missing source folder: {source_root / kind}')
            if any(p.is_dir() for p in (source_root / kind).iterdir()):
                raise ValueError('Source folders must be flat; nested folders are unsupported.')
        images = files_in(source_root / 'images', IMAGE_EXTENSIONS)
        labels = files_in(source_root / 'labels', {'.txt'})
        if not images and not labels:
            print(f'Source dataset is empty. Add images to {source_root / "images"} and matching .txt files to {source_root / "labels"}.')
            return 0
        
        for kind in ('images', 'labels'):
            for split in SPLITS:
                folder = DATASET / kind / split
                if folder.exists() and any(folder.iterdir()):
                    raise ValueError(f'Destination is not empty: {folder}. Manually back up and clear all splits before resplitting.')
        if len({p.stem.casefold() for p in images}) != len(images) or len({p.stem.casefold() for p in labels}) != len(labels):
            raise ValueError('Duplicate image or label basenames; every pair needs a unique basename.')
        label_map = {p.stem: p for p in labels}
        if {p.stem for p in images} != set(label_map):
            raise ValueError('Unpaired source images or labels. Each image needs a same-name .txt label, including matching case.')
        for label in labels:
            _, errors, warnings = check_label(label)
            for warning in warnings:
                print(f'WARNING: {warning}')
            if errors:
                raise ValueError('\n'.join(errors))
        random.Random(args.seed).shuffle(images)
        
        exact = [len(images) * r for r in args.ratios]
        sizes = [math.floor(n) for n in exact]
        order = sorted(range(3), key=lambda i: exact[i] - sizes[i], reverse=True)
        for i in order[:len(images) - sum(sizes)]:
            sizes[i] += 1
        created, offset = [], 0
        try:
            for split, size in zip(SPLITS, sizes):
                for kind in ('images', 'labels'):
                    (DATASET / kind / split).mkdir(parents=True, exist_ok=True)
                for image in images[offset:offset + size]:
                    for kind, source in (('images', image), ('labels', label_map[image.stem])):
                        target = DATASET / kind / split / source.name
                        with target.open('xb') as output:  
                            created.append(target)
                            with source.open('rb') as input_file:
                                shutil.copyfileobj(input_file, output)
                offset += size
        except Exception:
            for path in reversed(created):
                path.unlink(missing_ok=True)
            raise
        for split, size in zip(SPLITS, sizes):
            print(f'{split}: copied {size} images and {size} labels ({2 * size} files).')
            if not size:
                print(f'WARNING: {split} is empty; collect more pairs or adjust ratios.')
        print('Source files retained. Group related video frames manually to prevent split leakage.')
        return 0
    except (OSError, ValueError) as exc:
        print(f'Split stopped: {exc}')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
