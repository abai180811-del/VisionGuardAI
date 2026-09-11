"""Shared project paths and dataset utilities."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'dataset'
SPLITS = ('train', 'val', 'test')
CLASSES = {0: 'child', 1: 'window', 2: 'windowsill'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def files_in(folder, extensions):
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions), key=lambda p: p.name) if folder.is_dir() else []

def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def experiment_name(value):
    import argparse
    if not value or value in {'.', '..'} or any(c in value for c in '<>:"/\\|?*') or value.endswith((' ', '.')):
        raise argparse.ArgumentTypeError('Use a simple name without path separators.')
    return value

def select_device(value, torch):
    if value == 'auto':
        return 0 if torch.cuda.is_available() else 'cpu'
    if value == 'cpu':
        return value
    if value.isdigit() and torch.cuda.is_available() and int(value) < torch.cuda.device_count():
        return int(value)
    raise ValueError('CUDA device unavailable. Use --device cpu or install CUDA-enabled PyTorch.')
