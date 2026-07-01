# Driver Vision Review

This project reviews Euro Truck Simulator 2 driving from saved image frames. It uses a free local vision-language model with no training and no dataset.

It checks visible driving behavior such as lane crossing, traffic signals, stop/yield behavior, zebra crossings, pedestrians, cars crossing at junctions, unsafe turns, sudden impact, collision/near-miss signs, tailgating, hazards, and whether the driver forces other road users to wait.

It can only judge events that are visible in the saved frames. For best event detection, save enough frames around important moments and increase `--max-frames` if the drive is long.

## Install once

```bat
python -m pip install -r requirements.txt
```

## Run prediction

```bat
python predict_driver_review.py --frames "C:\path\to\your\frames"
```

Or:

```bat
run_driver_review.bat "C:\path\to\your\frames"
```

Outputs are written to `driver_review_output`:

- `driver_review.md`
- `driver_review.json`

The default profile is `accurate`, using `Qwen/Qwen2.5-VL-3B-Instruct`. If GPU memory or runtime fails, the code falls back to the smaller `fast` profile and CPU when needed.

Useful options:

```bat
python predict_driver_review.py --frames "C:\frames" --profile balanced
python predict_driver_review.py --frames "C:\frames" --profile fast --device cpu
python predict_driver_review.py --frames "C:\frames" --crop-mode left
python predict_driver_review.py --frames "C:\frames" --max-frames 32
```

The model cache is stored in `models_cache` so the project size can be checked easily. Keep only one large profile downloaded if you need the full folder to stay under 10 GB.
