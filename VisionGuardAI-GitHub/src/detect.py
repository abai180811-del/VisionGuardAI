
import argparse
import os
import math
import time
from datetime import datetime
from proximity import ProximityMonitor
from telegram_alerts import TelegramAlerts
from sound_alerts import SoundAlerts
from urllib.parse import urlsplit
from common import ROOT, CLASSES, project_path, experiment_name, select_device

def resolve_source(value):
    
    value = value.strip()
    if not value:
        raise ValueError('Source cannot be empty.')
    if '://' in value:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {'rtsp', 'rtmp', 'http', 'https', 'tcp'}:
            raise ValueError('Use an RTSP, RTMP, HTTP, HTTPS, or TCP camera stream URL.')
        if not parsed.hostname:
            raise ValueError('Camera URL must include a hostname or IP address.')
        
        _ = parsed.port
        return value
    if value.isdigit():
        return int(value)
    path = project_path(value)
    if not path.is_file():
        raise ValueError(f'Image or video does not exist: {path}')
    return str(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='results/train/weights/best.pt')
    parser.add_argument('--source', default=os.environ.get('VISIONGUARD_CAMERA_URL', '0'),
                        help='Wi-Fi camera stream URL, webcam index, or image/video path. Defaults to VISIONGUARD_CAMERA_URL, otherwise webcam 0.')
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--device', default='auto', help='auto, cpu, or CUDA index')
    parser.add_argument('--name', type=experiment_name, default='detect')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--telegram', action='store_true', help='Send proximity alerts to configured Telegram chat.')
    parser.add_argument('--overlap', type=float, default=.15, help='Required intersection area divided by child box area.')
    parser.add_argument('--confirmations', type=int, choices=(2, 3), default=3)
    parser.add_argument('--volume', type=float, default=.28, help='Sound amplitude 0 to 0.5; speaker volume also applies.')
    parser.add_argument('--mute', action='store_true', help='Disable local alert sound.')
    parser.add_argument('--hold-seconds', type=float, default=1., help='Seconds per confirmation period (1 to 2).')
    parser.add_argument('--cooldown', type=float, default=30., help='Minimum seconds between alert attempts.')
    parser.add_argument('--alert-conf', type=float, default=.5, help='Minimum child AND window confidence for alerts.')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('conf must be between 0 and 1.')
    if not all(math.isfinite(v) for v in (args.overlap, args.hold_seconds, args.cooldown, args.alert_conf, args.volume)) or not 0 < args.overlap <= 1 or not 1 <= args.hold_seconds <= 2 or not 0 <= args.volume <= .5 or args.cooldown < 1 or not 0 <= args.alert_conf <= 1:
        parser.error('overlap: (0,1]; alert-conf: [0,1]; hold-seconds: [1,2]; volume: [0,0.5]; cooldown >= 1. Values must be finite.')
    model_path = project_path(args.model)
    if not model_path.is_file():
        print(f'No trained model found at {model_path}.')
        print('Add and validate annotated data, then run src/train.py. Afterwards use --model results/<experiment>/weights/best.pt.')
        return 0
    try:
        source = resolve_source(args.source)
        monitor = ProximityMonitor(args.overlap, args.hold_seconds, args.cooldown, args.alert_conf, args.confirmations)
        sound = None if args.mute else SoundAlerts(args.volume)
        sender = TelegramAlerts() if args.telegram else None
        print('Proximity monitoring enabled (image-space estimate). Telegram: ' + ('enabled' if sender else 'OFF'))
        print('Local sound: ' + ('enabled' if sound else 'muted'))
        import torch
        from ultralytics import YOLO
        model = YOLO(str(model_path), task='detect')
        if model.task != 'detect' or model.names != CLASSES:
            raise ValueError('Use a detection model trained with 0: child, 1: window, 2: windowsill.')
        
        status_time = -math.inf
        diagnostics = ROOT / 'results' / 'proximity_status.log'
        diagnostics.parent.mkdir(exist_ok=True)
        diagnostics.write_text('New detection session. Telegram: ' + ('enabled' if sender else 'OFF') + '\n', encoding='utf-8')
        for result in model.predict(source=source, conf=args.conf, device=select_device(args.device, torch),
                               show=not args.no_show, save=True, stream=True, stream_buffer=False,
                               project=str(ROOT / 'results'), name=args.name, exist_ok=False,
                               show_labels=True, show_boxes=True, verbose=False):
            rows = result.boxes.data.cpu().tolist() if result.boxes is not None else []
            now = time.monotonic()
            triggered = monitor.update(rows, now)
            if triggered or now-status_time >= 2:
                confidence = ', '.join(f'{name}={max((r[4] for r in rows if r[5] == cid), default=0):.2f}'
                                       for cid, name in ((0, 'child'), (1, 'window')))
                status = f'{datetime.now().isoformat(timespec="seconds")} | {monitor.status} | {confidence}'
                print(status, flush=True)
                with diagnostics.open('a', encoding='utf-8') as log:
                    log.write(status + '\n')
                status_time = now
            if triggered:
                message = ('VisionGuardAI: possible child near window. Please check the camera now. '
                           'Image-space proximity detected; physical distance is not confirmed. '
                           + datetime.now().astimezone().isoformat(timespec='seconds'))
                print(message)
                if sound:
                    sound.play()
                if sender:
                    sender.submit(message)
        return 0
    except KeyboardInterrupt:
        print('Detection stopped.')
        return 0
    except ImportError as exc:
        print(f'Missing dependency: {exc}. Install requirements.txt.')
        return 1
    except Exception as exc:
        print(f'Detection failed: {exc}')
        if '://' in args.source:
            print('Check camera power, network access, stream URL, login details, and that RTSP/HTTP streaming is enabled. Use the video stream address, not the camera settings webpage.')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
