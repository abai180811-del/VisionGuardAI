# VisionGuardAI

A school computer-vision project using **Ultralytics YOLO11n** to detect children, windows, and windowsills. Sustained overlap between child and window boxes triggers two short local alert tones. Telegram notifications are optional.

GitHub hosts the source and trained model. The camera app runs locally on Windows, not on the GitHub webpage.

## Run the project

Install 64-bit Python 3.13. Select **Code > Download ZIP**, extract it, and open the extracted folder in PyCharm. In that folder's PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\detect.py --source 0
```

Select `.venv/Scripts/python.exe` as the PyCharm interpreter. Source 0 uses the laptop webcam. Ctrl+C stops detection. The included model is `results/train/weights/best.pt`. Internet is needed to install dependencies; local webcam detection and sound do not require Telegram.

A camera exposing RTSP can be used instead:

```powershell
.\.venv\Scripts\python.exe src\detect.py --source "rtsp://CAMERA_IP:554/STREAM_PATH"
```

Replace the placeholder address with the camera's actual local stream URL. Availability depends on firmware. Keep credentials private.

## Alert behavior

- Classes: **0 child**, **1 window**, **2 windowsill**.
- Display confidence: 25%; alert confidence for both child and window: 50%.
- Intersection must cover at least 15% of the child box.
- Three consecutive one-second periods of overlap trigger the sound. Missing detections, a changed pair, or a gap longer than one second reset confirmation.
- Two short faded tones play at 15% waveform amplitude, with a 30-second cooldown. Windows speaker volume also affects loudness.

```powershell
.\.venv\Scripts\python.exe src\detect.py --source 0 --overlap 0.15 --confirmations 3 --hold-seconds 1 --cooldown 30 --volume 0.15
```

Use `--mute` to disable sound. Status reasons appear in the terminal and `results/proximity_status.log`. Detection output is saved locally under `results/`. Timing uses processing time, not the original timeline of recorded videos.

**Limitation:** box overlap is not a physical-distance measurement. False alerts and missed detections are possible. This school prototype cannot replace child supervision or establish that a scene is safe.

## Training and evaluation

YOLO11n was fine-tuned for **50 epochs**, image size 640, batch 4, on CPU. The dataset had **55 real images: 39 train, 11 validation, 5 test**. The best model achieved validation mAP50 **0.556**, mAP50-95 **0.370**, precision **0.500**, and recall **0.593**.

| Class | Validation mAP50 |
| --- | ---: |
| child | 0.763 |
| window | 0.549 |
| windowsill | 0.355 |

These results are from only 11 validation images, not a held-out test evaluation. More varied data and camera-specific evaluation are needed. A supplemental stock image collected after training was not used in this checkpoint. Training images and camera recordings are not included in this repository.

## Train with your own dataset

Put images in `dataset/images/train`, `val`, and `test`, with matching same-basename `.txt` files in `dataset/labels/train`, `val`, and `test`. Use flat folders. Class order must match the mapping above.

YOLO detection labels have one object per row: `class_id x_center y_center width height`. Coordinates are normalized to image dimensions. For example:

```text
0 0.50 0.60 0.20 0.40
```

```powershell
.\.venv\Scripts\python.exe src\validate_dataset.py
.\.venv\Scripts\python.exe src\train.py --epochs 50 --name school_run --device auto
```

Training downloads pretrained YOLO11n weights on first use. `--device cpu` forces CPU; `--device 0` requires CUDA-enabled PyTorch. Unsplit pairs can be copied with `src/split_dataset.py --source "C:\path\to\source"`, where the source has images/ and labels/ folders. Populated destination splits are refused.

## Optional Telegram

Start a chat with your bot first. Set credentials locally, never in source control:

```powershell
$env:TELEGRAM_BOT_TOKEN = 'YOUR_PRIVATE_TOKEN'
$env:TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID'
.\.venv\Scripts\python.exe src\telegram_alerts.py
.\.venv\Scripts\python.exe src\detect.py --source 0 --telegram
```

The first script sends a connection-test message; the second enables actual overlap alerts. Telegram requires internet access. Sound remains enabled unless `--mute` is passed.

## Tests and files

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Eight offline tests cover overlap, confirmation, resets, cooldown, and mocked Telegram requests. They do not measure live detection quality or actual delivery.

- `src/detect.py`: inference and alert routing
- `src/proximity.py`: box-overlap timing logic
- `src/sound_alerts.py`: Windows sound generation/playback
- `src/telegram_alerts.py`: optional Telegram integration
- `src/train.py`, `validate_dataset.py`, `split_dataset.py`: training and dataset tools
- `results/train/weights/best.pt`: trained checkpoint
- `tests/`: offline tests

## Credits

Built with [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), PyTorch, OpenCV, NumPy and PyYAML. Consult [Ultralytics licensing](https://www.ultralytics.com/license) for applicable third-party software and model terms; this repository does not relicense them.

Training data came from the user's [Roboflow export](https://universe.roboflow.com/cvbnm1/visionguard-ai-whqkm/dataset/dataset), whose metadata declared CC BY 4.0. Original training images are not redistributed here.
