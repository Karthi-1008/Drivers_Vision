from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "accurate": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "description": "Best default accuracy while staying practical for one 8GB GPU.",
        "max_new_tokens": 650,
        "max_pixels": 900 * 900,
        "fallback_profile": "fast",
    },
    "balanced": {
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "description": "Good quality and usually easier on an 8GB GPU.",
        "max_new_tokens": 560,
        "max_pixels": 900 * 900,
        "fallback_profile": "fast",
    },
    "fast": {
        "model_id": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "description": "Small fallback for CPU or low-memory systems.",
        "max_new_tokens": 420,
        "max_pixels": 720 * 720,
        "fallback_profile": None,
    },
}


@dataclass
class ContactSheet:
    path: str
    first_frame: int
    last_frame: int
    frame_count: int


@dataclass
class InferenceAttempt:
    profile: str
    model_id: str
    device: str
    ok: bool
    error: str | None = None
    timed_out: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review Euro Truck Simulator 2 driving from a folder of frames."
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Folder containing captured driving frames.",
    )
    parser.add_argument(
        "--output",
        default="driver_review_output",
        help="Folder where reports and temporary contact sheets are written.",
    )
    parser.add_argument(
        "--profile",
        default="accurate",
        choices=sorted(MODEL_PROFILES),
        help="Model profile. accurate is the default; fast is safest for CPU.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Use auto for GPU first and CPU fallback.",
    )
    parser.add_argument(
        "--cpu-profile",
        default="fast",
        choices=sorted(MODEL_PROFILES),
        help="Profile used when the run must fall back to CPU.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=48,
        help="Maximum sampled frames to review.",
    )
    parser.add_argument(
        "--frames-per-sheet",
        type=int,
        default=12,
        help="Frames placed on each contact sheet.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=320,
        help="Pixel size of each contact-sheet cell.",
    )
    parser.add_argument(
        "--crop-mode",
        default="auto",
        choices=["auto", "none", "left", "right", "center"],
        help="Use left/right/center for VR or stereo captures if needed.",
    )
    parser.add_argument(
        "--model-cache",
        default="models_cache",
        help="Local cache folder for downloaded models.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Inference timeout per model/device attempt, after the model is available.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only already-downloaded model files.",
    )
    parser.add_argument(
        "--no-model-fallback",
        action="store_true",
        help="Do not switch to the smaller fallback model if the selected model fails.",
    )
    parser.add_argument(
        "--keep-sheets",
        action="store_true",
        help="Keep generated contact sheets in the output folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    frames_dir = Path(args.frames).expanduser().resolve()
    output_dir = (base_dir / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    cache_dir = (base_dir / args.model_cache).resolve() if not Path(args.model_cache).is_absolute() else Path(args.model_cache).resolve()

    if not frames_dir.exists() or not frames_dir.is_dir():
        print(f"Frames folder not found: {frames_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    configure_hf_cache(cache_dir)

    frames = discover_frames(frames_dir)
    if not frames:
        print(f"No image frames found in: {frames_dir}", file=sys.stderr)
        return 2

    sampled = sample_frames(frames, max_frames=args.max_frames)
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    sheets = build_contact_sheets(
        sampled,
        sheets_dir=sheets_dir,
        frames_per_sheet=max(1, args.frames_per_sheet),
        cell_size=max(128, args.cell_size),
        crop_mode=args.crop_mode,
    )

    warnings: list[str] = []
    if len(sampled) < len(frames):
        warnings.append(
            f"Reviewed {len(sampled)} sampled frames from {len(frames)} total frames."
        )

    observations, attempts = run_inference_with_fallback(
        sheets=sheets,
        args=args,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )

    if not observations:
        error_text = "; ".join(a.error or "unknown error" for a in attempts)
        print(f"Vision model could not produce a review: {error_text}", file=sys.stderr)
        return 1

    size_gb = folder_size_gb(cache_dir)
    if size_gb > 9.5:
        warnings.append(
            f"Model cache is {size_gb:.2f} GB. Keep only one large profile downloaded to stay under 10 GB."
        )

    report = build_driver_report(
        frames_dir=frames_dir,
        output_dir=output_dir,
        total_frames=len(frames),
        sampled_frames=len(sampled),
        sheets=sheets,
        observations=observations,
        attempts=attempts,
        warnings=warnings,
        model_cache_gb=size_gb,
    )

    write_reports(report, output_dir)

    if not args.keep_sheets:
        try:
            shutil.rmtree(sheets_dir)
        except OSError:
            pass

    print(f"Driver score: {report['driver_score_100']}/100")
    print(f"Markdown report: {output_dir / 'driver_review.md'}")
    print(f"JSON report: {output_dir / 'driver_review.json'}")
    return 0


def configure_hf_cache(cache_dir: Path) -> None:
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def discover_frames(frames_dir: Path) -> list[Path]:
    files = [
        p
        for p in frames_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=natural_sort_key)


def natural_sort_key(path: Path) -> list[Any]:
    import re

    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def sample_frames(frames: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[len(frames) // 2]]
    step = (len(frames) - 1) / (max_frames - 1)
    indexes = [round(i * step) for i in range(max_frames)]
    return [frames[i] for i in indexes]


def build_contact_sheets(
    frames: list[Path],
    sheets_dir: Path,
    frames_per_sheet: int,
    cell_size: int,
    crop_mode: str,
) -> list[ContactSheet]:
    sheets: list[ContactSheet] = []
    font = ImageFont.load_default()
    columns = min(4, frames_per_sheet)
    rows = max(1, (frames_per_sheet + columns - 1) // columns)

    for sheet_number, start in enumerate(range(0, len(frames), frames_per_sheet), start=1):
        chunk = frames[start : start + frames_per_sheet]
        sheet = Image.new("RGB", (columns * cell_size, rows * cell_size), color=(18, 18, 18))
        draw = ImageDraw.Draw(sheet)

        for offset, frame_path in enumerate(chunk):
            row, col = divmod(offset, columns)
            with Image.open(frame_path) as image:
                image = ImageOps.exif_transpose(image.convert("RGB"))
                image = crop_frame(image, crop_mode)
                image.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
                x = col * cell_size + (cell_size - image.width) // 2
                y = row * cell_size + (cell_size - image.height) // 2
                sheet.paste(image, (x, y))

            frame_label = f"{start + offset + 1}"
            label_box = (col * cell_size + 5, row * cell_size + 5)
            draw.rectangle(
                [label_box[0] - 2, label_box[1] - 2, label_box[0] + 34, label_box[1] + 14],
                fill=(0, 0, 0),
            )
            draw.text(label_box, frame_label, fill=(255, 255, 255), font=font)

        sheet_path = sheets_dir / f"sheet_{sheet_number:03d}.jpg"
        sheet.save(sheet_path, quality=88, optimize=True)
        sheets.append(
            ContactSheet(
                path=str(sheet_path),
                first_frame=start + 1,
                last_frame=start + len(chunk),
                frame_count=len(chunk),
            )
        )

    return sheets


def crop_frame(image: Image.Image, crop_mode: str) -> Image.Image:
    width, height = image.size
    if crop_mode == "none":
        return image

    if crop_mode == "auto":
        if width / max(1, height) < 2.4:
            return image
        crop_mode = "left"

    if crop_mode == "left":
        return image.crop((0, 0, width // 2, height))
    if crop_mode == "right":
        return image.crop((width // 2, 0, width, height))
    if crop_mode == "center":
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))
    return image


def run_inference_with_fallback(
    sheets: list[ContactSheet],
    args: argparse.Namespace,
    cache_dir: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[InferenceAttempt]]:
    attempts_to_run = build_attempt_plan(args)
    attempts: list[InferenceAttempt] = []

    for profile_name, device in attempts_to_run:
        profile = MODEL_PROFILES[profile_name]
        attempt = InferenceAttempt(
            profile=profile_name,
            model_id=profile["model_id"],
            device=device,
            ok=False,
        )

        try:
            model_path = ensure_model_available(
                model_id=profile["model_id"],
                cache_dir=cache_dir,
                offline=args.offline,
            )
        except Exception as exc:
            attempt.error = f"Model download/cache check failed: {exc}"
            attempts.append(attempt)
            continue

        result = run_worker_attempt(
            sheets=sheets,
            profile_name=profile_name,
            device=device,
            model_path=model_path,
            cache_dir=cache_dir,
            timeout_seconds=args.timeout_seconds,
            output_dir=output_dir,
        )
        attempt.ok = result.get("ok", False)
        attempt.error = result.get("error")
        attempt.timed_out = result.get("timed_out", False)
        attempts.append(attempt)

        if attempt.ok:
            return result["observations"], attempts

    return [], attempts


def build_attempt_plan(args: argparse.Namespace) -> list[tuple[str, str]]:
    requested = args.profile
    cuda = cuda_available()
    plan: list[tuple[str, str]] = []

    if args.device == "cuda":
        plan.append((requested, "cuda"))
        if not args.no_model_fallback:
            fallback = MODEL_PROFILES[requested].get("fallback_profile")
            if fallback:
                plan.append((fallback, "cuda"))
        return unique_attempts(plan)

    if args.device == "cpu":
        plan.append((args.cpu_profile if requested != "fast" else requested, "cpu"))
        if requested == "fast":
            plan.append((requested, "cpu"))
        return unique_attempts(plan)

    if cuda:
        plan.append((requested, "cuda"))
        if not args.no_model_fallback:
            fallback = MODEL_PROFILES[requested].get("fallback_profile")
            if fallback:
                plan.append((fallback, "cuda"))
        plan.append((args.cpu_profile, "cpu"))
    else:
        plan.append((args.cpu_profile, "cpu"))

    return unique_attempts(plan)


def unique_attempts(plan: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for item in plan:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def ensure_model_available(model_id: str, cache_dir: Path, offline: bool) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir),
        local_files_only=offline,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.txt",
            "*.model",
            "*.py",
            "*.tiktoken",
            "tokenizer*",
            "processor*",
            "preprocessor*",
            "merges.txt",
            "vocab.*",
            "special_tokens_map.json",
            "generation_config.json",
        ],
    )


def run_worker_attempt(
    sheets: list[ContactSheet],
    profile_name: str,
    device: str,
    model_path: str,
    cache_dir: Path,
    timeout_seconds: int,
    output_dir: Path,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    result_queue: mp.Queue = context.Queue()
    payload = {
        "sheets": [asdict(sheet) for sheet in sheets],
        "profile_name": profile_name,
        "profile": MODEL_PROFILES[profile_name],
        "device": device,
        "model_path": model_path,
        "cache_dir": str(cache_dir),
    }
    process = context.Process(target=inference_worker, args=(payload, result_queue))
    process.start()
    process.join(timeout=max(30, timeout_seconds))

    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
        return {
            "ok": False,
            "timed_out": True,
            "error": f"{profile_name} on {device} timed out after {timeout_seconds} seconds.",
        }

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        log_path = output_dir / "last_worker_error.txt"
        log_path.write_text(
            f"Worker exited with code {process.exitcode} and returned no result.\n",
            encoding="utf-8",
        )
        return {
            "ok": False,
            "error": f"Worker exited with code {process.exitcode}. See {log_path}.",
        }


def inference_worker(payload: dict[str, Any], result_queue: mp.Queue) -> None:
    try:
        runner = VisionLanguageRunner(
            model_id=payload["profile"]["model_id"],
            model_path=payload["model_path"],
            device=payload["device"],
            cache_dir=Path(payload["cache_dir"]),
            max_pixels=payload["profile"]["max_pixels"],
        )
        observations: list[dict[str, Any]] = []
        for sheet_data in payload["sheets"]:
            sheet = ContactSheet(**sheet_data)
            prompt = build_sheet_prompt(sheet)
            raw_text = runner.generate(
                image_path=Path(sheet.path),
                prompt=prompt,
                max_new_tokens=payload["profile"]["max_new_tokens"],
            )
            if not raw_text.strip():
                raise RuntimeError("The model returned an empty response.")
            observations.append(
                {
                    "sheet": sheet_data,
                    "raw_text": raw_text,
                    "parsed": parse_json_from_text(raw_text),
                }
            )
        result_queue.put({"ok": True, "observations": observations})
    except Exception:
        result_queue.put({"ok": False, "error": traceback.format_exc()})


class VisionLanguageRunner:
    def __init__(
        self,
        model_id: str,
        model_path: str,
        device: str,
        cache_dir: Path,
        max_pixels: int,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.dtype = self._select_dtype(self.device)

        processor_kwargs: dict[str, Any] = {}
        if "qwen" in model_id.lower():
            processor_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=str(cache_dir),
            trust_remote_code=False,
            **processor_kwargs,
        )
        self.model = self._load_model(
            model_ref=model_path,
            model_id=model_id,
            cache_dir=str(cache_dir),
            torch_dtype=self.dtype if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        self.model.to(self.device)
        self.model.eval()

    def _select_dtype(self, device: str) -> Any:
        if device != "cuda":
            return self.torch.float32
        try:
            if self.torch.cuda.is_bf16_supported():
                return self.torch.bfloat16
        except Exception:
            pass
        return self.torch.float16

    def _load_model(self, model_ref: str, model_id: str, **kwargs: Any) -> Any:
        import transformers

        loader_names: list[str] = []
        lowered = model_id.lower()
        if "qwen2.5-vl" in lowered or "qwen2_5_vl" in lowered:
            loader_names.append("Qwen2_5_VLForConditionalGeneration")
        loader_names.extend(
            [
                "AutoModelForMultimodalLM",
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq",
            ]
        )

        errors: list[str] = []
        for loader_name in loader_names:
            loader = getattr(transformers, loader_name, None)
            if loader is None:
                continue
            try:
                return loader.from_pretrained(model_ref, **kwargs)
            except Exception as exc:
                errors.append(f"{loader_name}: {exc}")

        joined = "\n".join(errors) if errors else "No compatible loader was available."
        raise RuntimeError(f"Could not load {model_id}.\n{joined}")

    def generate(self, image_path: Path, prompt: str, max_new_tokens: int) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self._build_inputs(messages=messages, image=image, prompt=prompt)
        inputs = self._move_inputs(inputs)

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        if input_length and generated.shape[-1] > input_length:
            new_tokens = generated[:, input_length:]
        else:
            new_tokens = generated
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    def _build_inputs(
        self,
        messages: list[dict[str, Any]],
        image: Image.Image,
        prompt: str,
    ) -> dict[str, Any]:
        try:
            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            pass

        placeholder_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = self.processor.apply_chat_template(
                placeholder_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            text = prompt

        try:
            return self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            )
        except Exception:
            return self.processor(
                text=text,
                images=image,
                return_tensors="pt",
            )

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if key in {"pixel_values", "image_embeds"} and self.device == "cuda":
                    moved[key] = value.to(self.device, dtype=self.dtype)
                else:
                    moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved


def build_sheet_prompt(sheet: ContactSheet) -> str:
    return f"""
You are reviewing simulated Euro Truck Simulator 2 driving from a chronological contact sheet.
The sheet contains frame numbers {sheet.first_frame} to {sheet.last_frame}.

Judge only what is visible from the driving view: road position, lane keeping, traffic distance,
signals/signs/lights, junction behavior, speed safety from visual context, hazards, weather,
traffic awareness, and smoothness implied by the sampled frames.

Pay special attention to serious driving faults:
- hitting, nearly hitting, or failing to yield to pedestrians
- not waiting at zebra crossings or pedestrian crossings
- entering a junction when another car, truck, bus, bicycle, or pedestrian has priority
- running or ignoring traffic signals, stop signs, yield signs, or lane markings
- crossing lane lines, wrong-lane driving, unsafe merging, unsafe overtaking, or cutting traffic
- sudden steering, sudden impact, collision, curb strikes, road-edge departures, or loss of control
- tailgating, unsafe following distance, blocking traffic, or forcing other road users to wait

Ignore physical hardware, Quest headset details, the steering wheel device, desk setup, overlays,
menus, watermarks, game HUD text, and anything not relevant to driving quality.
Do not invent events that are not visible.

Return compact JSON only with this exact structure:
{{
  "road_summary": "one sentence",
  "positive_evidence": ["short visible driving strength"],
  "negative_evidence": ["short visible driving weakness"],
  "risk_events": [
    {{"kind": "pedestrian|crosswalk|collision|near_miss|lane|distance|speed|traffic_signal|junction|yielding|vehicle_priority|unsafe_turn|unsafe_overtake|hazard|visibility|other",
      "severity": 1,
      "evidence": "visible evidence"}}
  ],
  "driver_score_100": 0,
  "confidence_0_1": 0.0
}}
""".strip()


def parse_json_from_text(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def build_driver_report(
    frames_dir: Path,
    output_dir: Path,
    total_frames: int,
    sampled_frames: int,
    sheets: list[ContactSheet],
    observations: list[dict[str, Any]],
    attempts: list[InferenceAttempt],
    warnings: list[str],
    model_cache_gb: float,
) -> dict[str, Any]:
    parsed_items = [item["parsed"] for item in observations if isinstance(item.get("parsed"), dict)]
    raw_texts = [item.get("raw_text", "") for item in observations]
    pros = unique_phrases(flatten_list(item.get("positive_evidence") for item in parsed_items))
    cons = unique_phrases(flatten_list(item.get("negative_evidence") for item in parsed_items))
    risks = normalize_risks(flatten_list(item.get("risk_events") for item in parsed_items))
    summaries = unique_phrases(item.get("road_summary", "") for item in parsed_items)

    if not pros:
        pros = infer_phrases_from_raw(raw_texts, positive=True)
    if not cons:
        cons = infer_phrases_from_raw(raw_texts, positive=False)
    if not summaries:
        summaries = ["The review is based on visible road behavior in the sampled frames."]

    score = calculate_driver_score(parsed_items, risks, raw_texts)
    confidence = calculate_confidence(parsed_items, parsed_count=len(parsed_items), total_count=len(observations))
    overall = build_overall_review(score, summaries, risks)
    successful_attempt = next((attempt for attempt in attempts if attempt.ok), attempts[-1])

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "frames_folder": str(frames_dir),
        "output_folder": str(output_dir),
        "total_frames_found": total_frames,
        "sampled_frames_reviewed": sampled_frames,
        "contact_sheets_reviewed": [asdict(sheet) for sheet in sheets],
        "model": {
            "profile": successful_attempt.profile,
            "model_id": successful_attempt.model_id,
            "device": successful_attempt.device,
            "cache_size_gb": round(model_cache_gb, 3),
        },
        "driver_score_100": score,
        "confidence_0_1": confidence,
        "overall_review": overall,
        "pros": pros[:10],
        "cons": cons[:10],
        "risk_events": risks[:20],
        "sheet_observations": observations,
        "attempts": [asdict(attempt) for attempt in attempts],
        "warnings": warnings,
        "notes": [
            "No training or dataset is used; this is zero-shot visual review of sampled frames.",
            "The score is a practical review score, not an official driving exam result.",
        ],
    }


def flatten_list(values: Any) -> list[Any]:
    flattened: list[Any] = []
    for value in values or []:
        if isinstance(value, list):
            flattened.extend(value)
        elif value:
            flattened.append(value)
    return flattened


def unique_phrases(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values or []:
        if not isinstance(value, str):
            continue
        phrase = " ".join(value.strip().split())
        if not phrase:
            continue
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            output.append(phrase)
    return output


def normalize_risks(values: list[Any]) -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        evidence = str(value.get("evidence", "")).strip()
        if not evidence:
            continue
        severity = value.get("severity", 1)
        try:
            severity_int = max(1, min(5, int(float(severity))))
        except (TypeError, ValueError):
            severity_int = 1
        risks.append(
            {
                "kind": str(value.get("kind", "other")).strip() or "other",
                "severity": severity_int,
                "evidence": evidence,
            }
        )
    return risks


def infer_phrases_from_raw(raw_texts: list[str], positive: bool) -> list[str]:
    joined = "\n".join(raw_texts).lower()
    if positive:
        candidates = [
            ("lane", "Maintains usable lane awareness in the sampled road view."),
            ("clear", "Keeps the driving path mostly clear in visible frames."),
            ("safe", "Shows generally safe visible road behavior."),
        ]
    else:
        candidates = [
            ("close", "Possible close-distance or spacing issue appears in the model notes."),
            ("lane", "Lane discipline may need attention in some sampled frames."),
            ("hazard", "Potential hazard awareness issue appears in the model notes."),
        ]
    output = [phrase for keyword, phrase in candidates if keyword in joined]
    fallback = "No strong visible weakness was detected." if not positive else "No strong visible strength was isolated."
    return output or [fallback]


def calculate_driver_score(
    parsed_items: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    raw_texts: list[str],
) -> int:
    scores: list[float] = []
    for item in parsed_items:
        value = item.get("driver_score_100")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if 0 <= numeric <= 100:
            scores.append(numeric)

    base = sum(scores) / len(scores) if scores else 78.0
    penalty = 0.0
    for risk in risks:
        severity = float(risk.get("severity", 1))
        kind = str(risk.get("kind", "other")).lower()
        weight = {
            "pedestrian": 5.0,
            "crosswalk": 4.8,
            "collision": 5.5,
            "near_miss": 4.8,
            "hazard": 2.8,
            "distance": 2.6,
            "traffic_signal": 3.4,
            "junction": 2.8,
            "yielding": 4.2,
            "vehicle_priority": 3.8,
            "lane": 2.4,
            "speed": 2.5,
            "unsafe_turn": 3.4,
            "unsafe_overtake": 3.8,
            "visibility": 1.8,
            "other": 1.5,
        }.get(kind, 1.5)
        penalty += severity * weight

    raw_joined = " ".join(raw_texts).lower()
    severe_words = [
        "collision",
        "crash",
        "hit pedestrian",
        "hit a pedestrian",
        "hit person",
        "hit a person",
        "near miss",
        "red light",
        "zebra crossing",
        "crosswalk",
        "failed to yield",
        "did not yield",
        "wrong lane",
        "unsafe turn",
        "sudden impact",
    ]
    if any(word in raw_joined for word in severe_words):
        penalty += 10.0

    final_score = round(max(0.0, min(100.0, base - min(35.0, penalty * 0.35))))
    return int(final_score)


def calculate_confidence(
    parsed_items: list[dict[str, Any]],
    parsed_count: int,
    total_count: int,
) -> float:
    values: list[float] = []
    for item in parsed_items:
        try:
            value = float(item.get("confidence_0_1"))
        except (TypeError, ValueError):
            continue
        if 0 <= value <= 1:
            values.append(value)

    model_confidence = sum(values) / len(values) if values else 0.55
    parse_ratio = parsed_count / max(1, total_count)
    confidence = (model_confidence * 0.75) + (parse_ratio * 0.25)
    return round(max(0.0, min(1.0, confidence)), 3)


def build_overall_review(
    score: int,
    summaries: list[str],
    risks: list[dict[str, Any]],
) -> str:
    if score >= 85:
        label = "strong"
    elif score >= 70:
        label = "good but improvable"
    elif score >= 55:
        label = "mixed"
    else:
        label = "high risk"

    summary = summaries[0] if summaries else "The sampled frames were reviewed for visible road behavior."
    if risks:
        top_risk = max(risks, key=lambda item: item.get("severity", 1))
        return (
            f"Overall, the driver looks {label}. {summary} "
            f"The main visible issue is {top_risk['kind']}: {top_risk['evidence']}"
        )
    return f"Overall, the driver looks {label}. {summary}"


def write_reports(report: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "driver_review.json"
    markdown_path = output_dir / "driver_review.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(build_markdown(report), encoding="utf-8")


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Driver Review",
        "",
        f"Driver score: {report['driver_score_100']}/100",
        f"Confidence: {report['confidence_0_1']}",
        "",
        "## Overall Review",
        report["overall_review"],
        "",
        "## Pros",
    ]
    lines.extend(f"- {item}" for item in report["pros"])
    lines.extend(["", "## Cons"])
    lines.extend(f"- {item}" for item in report["cons"])
    lines.extend(["", "## Risk Events"])
    if report["risk_events"]:
        for risk in report["risk_events"]:
            lines.append(
                f"- Severity {risk['severity']} {risk['kind']}: {risk['evidence']}"
            )
    else:
        lines.append("- No clear high-risk event was detected in the sampled frames.")

    lines.extend(
        [
            "",
            "## Run Details",
            f"- Frames folder: {report['frames_folder']}",
            f"- Total frames found: {report['total_frames_found']}",
            f"- Sampled frames reviewed: {report['sampled_frames_reviewed']}",
            f"- Model profile: {report['model']['profile']}",
            f"- Model id: {report['model']['model_id']}",
            f"- Device: {report['model']['device']}",
            f"- Model cache size: {report['model']['cache_size_gb']} GB",
        ]
    )

    if report["warnings"]:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item}" for item in report["warnings"])

    lines.extend(["", "## Notes"])
    lines.extend(f"- {item}" for item in report["notes"])
    lines.append("")
    return "\n".join(lines)


def folder_size_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total / (1024 ** 3)


if __name__ == "__main__":
    raise SystemExit(main())
