"""Train a three-class YOLO11 detector after annotated data is added."""
import argparse
from common import ROOT, DATASET, CLASSES, experiment_name, select_device
from validate_dataset import validate_dataset, print_report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='auto', help='auto, cpu, or CUDA index such as 0')
    parser.add_argument('--workers', type=int, default=0, help='0 is a safe Windows default.')
    parser.add_argument('--name', type=experiment_name, default='train')
    args = parser.parse_args()
    if min(args.epochs, args.imgsz, args.batch) < 1 or args.workers < 0:
        parser.error('epochs, imgsz, batch must be positive; workers must be nonnegative.')
    try:
        report = validate_dataset(DATASET)
        print_report(report)
        if not any(s['images'] for s in report['splits'].values()):
            print(f'Dataset is empty. Add images to {DATASET / "images"} / train and val,')
            print(f'and matching .txt labels to {DATASET / "labels"} / train and val.')
            print('For unsplit pairs, run src/split_dataset.py --source PATH containing images/ and labels/. No training started.')
            return 0
        if report['errors']:
            print('Fix dataset errors before training.')
            return 1
        if any(not report['splits'][s]['images'] or not sum(report['splits'][s]['classes'].values()) for s in ('train', 'val')):
            print('Training requires images and annotated objects in BOTH train and val.')
            return 1
        # Import heavy packages only after checks; no download on empty data.
        import yaml
        import torch
        from ultralytics import YOLO
        config = yaml.safe_load((DATASET / 'data.yaml').read_text(encoding='utf-8'))
        if config.get('names') != CLASSES or config.get('nc') != 3:
            raise ValueError('data.yaml must define exactly 0: child, 1: window, 2: windowsill; nc: 3.')
        # Use an absolute runtime YAML so Ultralytics global path settings and
        # PyCharm working-directory choices cannot redirect this dataset.
        config['path'] = DATASET.resolve().as_posix()
        for split in ('train', 'val', 'test'):
            if config.get(split) != f'images/{split}':
                raise ValueError(f'data.yaml {split} must be images/{split}.')
        results = ROOT / 'results'
        results.mkdir(exist_ok=True)
        runtime_yaml = results / 'data.resolved.yaml'
        runtime_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        device = select_device(args.device, torch)
        print(f'Using device: {device}')
        (ROOT / 'models').mkdir(exist_ok=True)
        model = YOLO(str(ROOT / 'models' / 'yolo11n.pt'), task='detect')
        model.train(data=str(runtime_yaml), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=device, workers=args.workers,
                    project=str(results), name=args.name, exist_ok=False, seed=42)
        return 0
    except ImportError as exc:
        print(f'Missing dependency: {exc}. Install requirements.txt.')
        return 1
    except Exception as exc:
        print(f'Training could not start or complete: {exc}')
        return 1

if __name__ == '__main__':
    # Windows multiprocessing guard for workers > 0.
    from multiprocessing import freeze_support
    freeze_support()
    raise SystemExit(main())
