# VisionGuardAI

A fresh Python 3.13 project for collecting bounding-box annotations and later training an Ultralytics YOLO11 object detector. Exactly three classes: **0 child**, **1 window**, **2 windowsill**. A configurable image-space proximity heuristic can issue console and optional Telegram alerts. All image and label folders are intentionally empty; no model or dataset is included.

## Windows and PyCharm setup

Use standard **64-bit CPython 3.13**. Open this folder as a project in PyCharm. In its PowerShell terminal, run:

```powershell
cd C:\Users\abay.ippo\VisionGuardAI
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The first command must report Python 3.13.x before creating the environment. If needed, install Python 3.13 and use its full python.exe path. If the Windows Python launcher recognizes your installation, `py -3.13 -m venv .venv` is another option.

In PyCharm Settings, find **Project > Python Interpreter**, choose **Add Interpreter > Add Local Interpreter > Existing**, and select `VisionGuardAI\.venv\Scripts\python.exe` (menu wording can vary). You can also create a new Virtualenv there with Python 3.13 as its base interpreter. Right-click any script in `src` and choose Run. Add options in Run > Edit Configurations > Parameters. Default project paths derive from `__file__`, so the working directory does not affect them. The optional validator `--dataset` path follows the terminal working directory.

Optional terminal activation: `.\.venv\Scripts\Activate.ps1` in PowerShell, or `.venv\Scripts\activate.bat` in cmd. If activation is blocked, keep using the explicit interpreter commands above; no execution-policy change is needed.

Dependencies are intentionally unpinned. Current [PyTorch](https://pypi.org/project/torch/), [NumPy](https://pypi.org/project/numpy/), and [PyYAML](https://pypi.org/project/PyYAML/) publish CPython 3.13 Windows x64 wheels; [OpenCV](https://pypi.org/project/opencv-python/) publishes a compatible stable-ABI Windows wheel. Ultralytics installs PyTorch and its other dependencies. Availability of wheels does not guarantee every future combination or CUDA driver works. If your package index or chosen CUDA build has no Python 3.13 distribution, use **Python 3.12** in a new environment as the closest fallback; do not attempt an improvised source-build workaround. Do not install headless OpenCV alongside opencv-python if you want display windows.

## Dataset and annotations

```text
VisionGuardAI/
|-- dataset/
|   |-- images/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   |-- labels/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   `-- data.yaml
|-- models/
|-- results/
|-- src/
|   |-- common.py
|   |-- train.py
|   |-- validate_dataset.py
|   |-- split_dataset.py
|   `-- detect.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

Use flat folders and JPG, JPEG, PNG, BMP, or WEBP images. For example, a real `dataset/images/train/room001.jpg` requires `dataset/labels/train/room001.txt`. The identical basename, including case, links the image to its boxes. Do not use both room001.jpg and room001.png. Export labels as YOLO **object detection**, not classification, polygons, or segmentation. Annotate all relevant objects consistently.

Each UTF-8 label file has one row per object, with five space-separated fields:

```text
class_id x_center y_center width height
```

Coordinates are normalized: divide horizontal values by image width and vertical values by image height. Centers are measured from the top-left origin. All four values must be between 0 and 1; box width and height must be positive. IDs must be integer 0 (child), 1 (window), or 2 (windowsill). Example syntax only, not a supplied dataset file:

```text
0 0.50 0.60 0.20 0.40
1 0.60 0.35 0.50 0.50
2 0.60 0.63 0.55 0.06
```

An empty label is appropriate only for an intentionally verified background image with none of these objects. The validator warns on empty labels so accidental omissions can be reviewed. Missing labels are errors in this project. It checks annotation syntax and pairing, not whether image pixels are decodable or annotations are visually accurate.

Place an already split dataset directly in the matching `dataset/images/train`, `val`, `test` and `dataset/labels/train`, `val`, `test` folders, then run validation. No extra dataset section is needed. For unsplit pairs, pass an external folder containing `images/` and `labels/` to the optional splitter (replace the example path below):

```powershell
.\.venv\Scripts\python.exe src\split_dataset.py --source "C:\path\to\unsplit_dataset"
.\.venv\Scripts\python.exe src\validate_dataset.py
```

Splitting uses 80% train, 15% validation, 5% test with seed 42, sorted inputs before shuffling, and largest-remainder rounding. To specify these explicitly:

```powershell
.\.venv\Scripts\python.exe src\split_dataset.py --source "C:\path\to\unsplit_dataset" --ratios 0.80 0.15 0.05 --seed 42
```

The script copies pairs and retains source originals. It refuses missing pairs, malformed labels, duplicate basenames, and any populated destination split; it never silently overwrites. Back up and manually clear all destination splits before resplitting. Small datasets may produce empty splits. Keep test data held out from training. For images from the same child, room, recording, or near-duplicate frames, assign whole groups to one split manually: random per-image splitting cannot prevent that leakage.

Validation prints image/label counts, valid object counts per class, warnings, errors, and a summary for every split. Empty datasets are handled normally. Exit code 0 means no errors (warnings can remain); 1 means errors; 2 means invalid command-line arguments.

## Training later

```powershell
.\.venv\Scripts\python.exe src\train.py
.\.venv\Scripts\python.exe src\train.py --epochs 50 --imgsz 640 --batch 4 --workers 0 --device auto --name train
```

These are alternative ways to launch one run. With the supplied empty dataset, training prints instructions and exits without loading YOLO or downloading weights. Once train and val contain valid annotations, it starts from pretrained `yolo11n.pt` and downloads that weight file into `models` if needed (internet access required on first use). Results and trained weights go under `results/<experiment>/`, typically `results/train/weights/best.pt`. Existing experiment names are incremented by Ultralytics; use the actual output path for detection.

`dataset/data.yaml` is the source configuration. The launcher writes `results/data.resolved.yaml` with an absolute dataset root, avoiding Ultralytics global dataset-directory settings and PyCharm working-directory issues. Always use this launcher for the supplied relative YAML. If using the Ultralytics CLI directly, first set `path` to an absolute dataset directory. Keep the three standard split paths and class mapping unchanged.

CPU is selected automatically unless PyTorch detects CUDA. Force CPU with:

```powershell
.\.venv\Scripts\python.exe src\train.py --device cpu
```

For NVIDIA CUDA, use the official [PyTorch installer selector](https://pytorch.org/get-started/locally/) for Windows, pip, and a CUDA build supported by your GPU/driver. Run its installation command with `.\.venv\Scripts\python.exe -m pip` as the pip executable. Then check and select the first GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe src\train.py --device 0
```

An explicit unavailable CUDA device produces an explanation. If memory runs out, reduce `--batch` to 1 or 2 and optionally `--imgsz` to 416. CPU training can take a long time. Windows defaults to zero data-loader workers; increase only after a successful run.

## Detection later

### Wi-Fi / IP camera

Detection accepts RTSP, RTMP, HTTP, HTTPS, and TCP stream URLs through Ultralytics. Your camera must expose a compatible video stream; a camera accessible only through its vendor app may not provide one. Connect the PC and camera to the same reachable local network and enable streaming in the camera settings. Obtain the exact stream address from the camera manual/app; the paths below are placeholders, not universal camera addresses.

```powershell
.\.venv\Scripts\python.exe src\detect.py --source "rtsp://192.168.1.100:554/stream1"
.\.venv\Scripts\python.exe src\detect.py --source "http://192.168.1.100:8080/video"
```

For a persistent PyCharm run configuration, add `VISIONGUARD_CAMERA_URL` in Run > Edit Configurations > Environment variables. Its value is your camera stream URL. Then run detect.py without `--source`; an explicit `--source` overrides the environment variable. In PowerShell, you can set it for the current terminal session:

```powershell
$env:VISIONGUARD_CAMERA_URL = 'rtsp://192.168.1.100:554/stream1'
.\.venv\Scripts\python.exe src\detect.py
```

If the camera requires login credentials, its URL may use `rtsp://USERNAME:PASSWORD@CAMERA_IP:554/STREAM_PATH`; percent-encode special characters in credentials. Keep actual credentials out of committed files; library diagnostics may include the URL. Test the address in a network video player if connection fails. This code uses Ultralytics' stream handling and drops queued old frames instead of buffering an increasing delay. Wi-Fi interruptions can still stall or end detection; restore connectivity and restart if needed.

The model still runs on your PC. Boxes and class names are displayed there, and annotated output is saved in `results/`. A trained model is required; no camera connection is attempted if the model is missing. See [Ultralytics stream sources](https://docs.ultralytics.com/modes/predict/).

Without a camera environment variable or `--source`, the source is webcam 0. The default model path is `results/train/weights/best.pt`; if absent, the script explains how to obtain it and exits without opening a camera.

```powershell
.\.venv\Scripts\python.exe src\detect.py
.\.venv\Scripts\python.exe src\detect.py --model results\train\weights\best.pt --source 0 --conf 0.35
.\.venv\Scripts\python.exe src\detect.py --model results\train\weights\best.pt --source "C:\path\to\photo.jpg"
.\.venv\Scripts\python.exe src\detect.py --model results\train\weights\best.pt --source "C:\path\to\video.mp4" --no-show
```

Replace example media paths with existing files. Relative model/media paths start at the project root. The trained model must use the exact three-class mapping. Boxes and class names display by default; annotated output is saved under `results/detect` (or an incremented name). Use `--name` to name a run, `--device cpu` to force CPU, and Ctrl+C to stop webcam inference. `--no-show` disables display while still saving output.

Model quality requires a large, varied, accurately annotated dataset, including different rooms, lighting, viewpoints, window types, occlusions, and backgrounds. Three object detections alone do not determine danger; the proximity alerts are experimental and do not measure physical distance or establish that a scene is safe.

Reference: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) and [training documentation](https://docs.ultralytics.com/modes/train/).

Dataset contents, environments, models, and results are ignored by Git. Git does not preserve empty folders; recreate the shown folders if distributing via Git. No training or pretrained model is included in this foundation.

## Imported Roboflow dataset (2026-09-10)

Imported 55 pairs: 39 train, 11 val, 5 test. Classes already match child/window/windowsill. Two AVIF files mislabeled as JPG were re-encoded as JPEG without resizing; the original ZIP is unchanged. Import provenance is in dataset/import_manifest.json. All images decode and annotation validation passes. This small dataset is a baseline.

Training configuration: YOLO11n, 50 epochs, 640 pixels, batch 4, workers 0, CPU (installed PyTorch has no CUDA). Logs: results/training.stdout.log and results/training.stderr.log.

To explicitly use the laptop webcam after training, overriding any Wi-Fi camera environment setting:

```powershell
.\.venv\Scripts\python.exe src\detect.py --model results\train\weights\best.pt --source 0
```

## Child-near-window alerts and Telegram

No retraining is required: `detect.py` applies a geometric heuristic to the trained detector's boxes. It requires both child and window confidence >= 0.5, and a box-to-box gap <= 10% of the window box diagonal (overlap has gap zero). The same approximate pair must persist for 2 seconds and at least 3 observations. A missing detection, a different pair, or a frame interval greater than 1 second resets confirmation. Pair matching uses box overlap, not reliable identity tracking. Windowsill detections alone do not trigger alerts. The cooldown is camera-wide: one alert attempt per 60 seconds, with reminders if proximity continues.

These defaults need tuning with this camera. Apparent overlap can occur even when objects are physically far apart. Missed detections cause missed alerts; this small model is not a reliable child-safety system and cannot replace supervision. Test with an adult or recorded footage rather than placing a child near an unsafe window.

1. Create the bot with BotFather. Keep the token private.
2. Open the bot chat in Telegram and press Start (or add it to the intended group with permission to send messages).
3. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` locally. The chat ID identifies the recipient, not the bot username. You can retrieve it from a message's `chat.id` using Telegram's getUpdates API after sending the bot a message. Do not commit credentials or paste them into this project.
4. In PyCharm, use Run > Edit Configurations > Environment variables for these values, and add `--source 0 --telegram` to Parameters.

PowerShell alternative (replace placeholders locally):

```powershell
$env:TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN'
$env:TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID'
.\.venv\Scripts\python.exe src\detect.py --source 0 --telegram
```

Tune with:

```powershell
.\.venv\Scripts\python.exe src\detect.py --source 0 --telegram --near-margin 0.10 --hold-seconds 2 --cooldown 60 --alert-conf 0.5
```

Higher margin triggers at larger image gaps. Higher confirmation time reduces brief triggers but delays alerts. The detector's `--conf` also filters boxes before the alert threshold applies. Without `--telegram`, alerts print to the console only. Missing Telegram configuration produces a clear error before camera access. Messages contain text and a timestamp, not photos, credentials, or stream URLs. The bot sends the message to your configured chat; no separate bot server is needed.

Telegram HTTP requests run in a bounded background worker with a 10-second timeout so they do not freeze inference. Success/failure is printed, with credentials redacted. Failed sends are not automatically retried; continuing proximity may trigger another attempt after cooldown. Pending messages may be lost when detection exits. Processing timestamps are used: recorded videos are evaluated at processing speed, not their original playback timeline, and a single still image cannot meet the multi-frame confirmation rule. Camera disconnects are not treated as an all-clear or sent as Telegram alerts.

API reference: https://core.telegram.org/bots/api#sendmessage

Detection confidence defaults to 0.25 (25%); proximity-alert confidence defaults to 0.50 (50%). Restart detection for changed defaults to apply; explicit command-line confidence values override them.

## Default local sound alerts (updated behavior)

Local sound is now the default; Telegram is optional and requires --telegram. Use start_sound.ps1 for the laptop webcam, or run `.\.venv\Scripts\python.exe src\detect.py --source 0`. No bot token is needed. The older start_telegram.ps1 launcher explicitly enables Telegram and will still send messages.

This replaces the previous gap-based proximity rule: intersection must cover at least 15% of the child box. Both boxes must meet 50% confidence. Three consecutive one-second confirmation periods (three seconds total) trigger the alarm from assets/alarm.wav three times, with 0.4-second pauses (about 16 seconds total). A missed detection, pair change, or >1-second observation gap resets confirmation; merely touching boxes does not count. These are consecutive periods of sustained overlap, not three separate approach-and-retreat events. Persistent overlap can repeat the sound after the 30-second cooldown.

Options: --overlap 0.15, --confirmations 2 or 3, --hold-seconds 1 to 2 (per period), --cooldown 30, --volume 0.28, --mute. The old --near-margin option is removed. Volume is waveform amplitude (default 28%, maximum 50%), not a guaranteed loudness: Windows/speaker volume still applies. No system volume setting is changed. Alert sounds are written to results/alert.wav and played asynchronously. Image overlap does not establish physical proximity.
