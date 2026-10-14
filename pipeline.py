"""Download football-player images and generate standardized face crops."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

PLAYERS = {
    "lionel_messi": "Lionel Messi",
    "cristiano_ronaldo": "Cristiano Ronaldo",
    "zinedine_zidane": "Zinedine Zidane",
    "lamine_yamal": "Lamine Yamal",
    "kylian_mbappe": "Kylian Mbappé",
    "riyad_mahrez": "Riyad Mahrez",
    "neymar": "Neymar",
}

QUERY_VARIANTS = (
    "press conference interview headshot -watermark -stock",
    "match photo face close up -watermark -stock",
    "award ceremony solo portrait -watermark -stock",
    "high resolution profile photo -watermark -stock",
)
IMAGE_SIZE = (160, 160)
MIN_DIMENSION = 200
PADDING_RATIO = 0.20
TEXT_CONFIDENCE_THRESHOLD = 0.40
FACE_CONFIDENCE_THRESHOLD = 0.90
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
CREDIT_PATTERN = re.compile(
    r"(?:photo(?:graph)?\s*(?:by|credit|:)|photographer\s*[:\-]|credit\s*[:\-]|©|"
    r"Getty Images|Alamy|Shutterstock|Reuters|AFP|AP Images|PA Images|Imago Images)",
    re.IGNORECASE,
)
LOGGER = logging.getLogger("football_face_pipeline")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    entries.append(entry)
            except json.JSONDecodeError:
                LOGGER.warning("Ignoring malformed manifest entry %s:%d", path, line_number)
    return entries


def append_manifest(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_pillow() -> Any:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError("Pillow is required; install dependencies with `pip install -r requirements.txt`.") from exc
    return Image, UnidentifiedImageError


def decode_image(content: bytes) -> tuple[Any | None, str | None]:
    Image, UnidentifiedImageError = get_pillow()
    try:
        from io import BytesIO

        with Image.open(BytesIO(content)) as image:
            if image.format == "SVG":
                return None, "unsupported_svg"
            image.load()
            if image.width < MIN_DIMENSION or image.height < MIN_DIMENSION:
                return None, "below_minimum_resolution"
            converted = image.convert("RGB")
            return converted, None
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "corrupt_or_unsupported"


def _metadata_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00 \t\r\n")
    return str(value).strip()


def find_photographer_credit(result: dict[str, Any]) -> str | None:
    for key in ("photographer", "creator", "credit", "attribution", "author"):
        value = _metadata_text(result.get(key, ""))
        if value:
            return value
    for key in ("title", "source", "url"):
        value = _metadata_text(result.get(key, ""))
        if CREDIT_PATTERN.search(value):
            return value
    return None


def find_image_photographer_credit(content: bytes) -> str | None:
    Image, _ = get_pillow()
    from io import BytesIO

    try:
        with Image.open(BytesIO(content)) as image:
            for tag_id, value in image.getexif().items():
                if tag_id in {315, 33432, 40091}:
                    credit = _metadata_text(value)
                    if credit:
                        return credit
    except (OSError, ValueError):
        return None
    return None


def existing_download_state(
    raw_dir: Path, manifest_entries: list[dict[str, Any]]
) -> tuple[set[str], set[str], int]:
    valid_sources: set[str] = set()
    content_hashes: set[str] = set()
    excluded_filenames = {
        str(entry.get("filename"))
        for entry in manifest_entries
        if entry.get("photographer_credit") or entry.get("status") == "photographer_credit_detected"
    }
    for entry in manifest_entries:
        if entry.get("status") != "accepted":
            continue
        filename = entry.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            continue
        image_path = raw_dir / filename
        if not image_path.is_file():
            continue
        try:
            content = image_path.read_bytes()
            validated_image, validation_error = decode_image(content)
        except OSError:
            continue
        if validation_error or validated_image is None or find_image_photographer_credit(content):
            continue
        valid_sources.add(str(entry.get("image_url", "")))
        content_hashes.add(hashlib.sha256(content).hexdigest())

    if raw_dir.is_dir():
        for image_path in raw_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            if image_path.name in excluded_filenames:
                continue
            try:
                content = image_path.read_bytes()
                image, error = decode_image(content)
            except OSError:
                continue
            if image is not None and error is None and not find_image_photographer_credit(content):
                content_hashes.add(hashlib.sha256(content).hexdigest())
    return valid_sources, content_hashes, len(content_hashes)


def search_images(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("The `ddgs` package is required; install dependencies with `pip install -r requirements.txt`.") from exc
    with DDGS() as search_client:
        return list(search_client.images(query, max_results=max_results, safesearch="moderate"))


def download_candidate(image_url: str, timeout: float) -> tuple[bytes | None, str | None]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("The `requests` package is required; install dependencies with `pip install -r requirements.txt`.") from exc
    try:
        response = requests.get(
            image_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DatasetResearch/1.0)"},
            timeout=timeout,
            stream=True,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "svg" in content_type:
            return None, "unsupported_svg"
        content = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            content.extend(chunk)
            if len(content) > MAX_DOWNLOAD_BYTES:
                return None, "file_too_large"
        return bytes(content), None
    except Exception as exc:
        return None, f"download_error: {type(exc).__name__}"


def download_player(
    player_id: str,
    player_name: str,
    raw_root: Path,
    images_needed: int,
    timeout: float,
) -> int:
    from tqdm import tqdm

    raw_dir = raw_root / player_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.jsonl"
    entries = load_manifest(manifest_path)
    seen_sources, seen_hashes, _ = existing_download_state(raw_dir, entries)
    progress = tqdm(total=images_needed, desc=f"Download {player_name}", unit="image")
    accepted_this_run = 0
    per_query_goal = math.ceil(images_needed / len(QUERY_VARIANTS))

    for variant_number, variant in enumerate(QUERY_VARIANTS, start=1):
        remaining = images_needed - accepted_this_run
        if remaining <= 0:
            continue
        accepted_for_query = 0
        query = f"{player_name} {variant}"
        try:
            results = search_images(query, max_results=max(remaining * 3, 40))
        except Exception as exc:
            LOGGER.warning("Image search failed for %s (%s): %s", player_name, variant, exc)
            continue

        for result in results:
            if accepted_this_run >= images_needed or accepted_for_query >= per_query_goal:
                break
            image_url = str(result.get("image", "")).strip()
            source_url = str(result.get("url", "")).strip()
            photographer_credit = find_photographer_credit(result)
            manifest_entry: dict[str, Any] = {
                "player": player_id,
                "query": query,
                "query_variant": variant,
                "image_url": image_url,
                "source_url": source_url,
                "title": str(result.get("title", "")),
                "photographer_credit": photographer_credit,
                "status": "rejected",
            }
            if photographer_credit:
                manifest_entry["status"] = "photographer_credit_detected"
                append_manifest(manifest_path, manifest_entry)
                continue
            if not image_url:
                manifest_entry["status"] = "missing_image_url"
                append_manifest(manifest_path, manifest_entry)
                continue
            if image_url in seen_sources:
                manifest_entry["status"] = "duplicate_source"
                append_manifest(manifest_path, manifest_entry)
                continue

            content, download_error = download_candidate(image_url, timeout)
            if content is None:
                manifest_entry["status"] = download_error or "download_failed"
                append_manifest(manifest_path, manifest_entry)
                continue
            photographer_credit = find_image_photographer_credit(content)
            if photographer_credit:
                manifest_entry["photographer_credit"] = photographer_credit
                manifest_entry["status"] = "photographer_credit_detected"
                append_manifest(manifest_path, manifest_entry)
                continue
            image, validation_error = decode_image(content)
            if validation_error or image is None:
                manifest_entry["status"] = validation_error or "corrupt_or_unsupported"
                append_manifest(manifest_path, manifest_entry)
                continue

            jpeg_buffer = io.BytesIO()
            image.save(jpeg_buffer, format="JPEG", quality=95, optimize=True)
            jpeg_content = jpeg_buffer.getvalue()
            content_hash = hashlib.sha256(jpeg_content).hexdigest()
            if content_hash in seen_hashes:
                manifest_entry["status"] = "duplicate_content"
                manifest_entry["sha256"] = content_hash
                append_manifest(manifest_path, manifest_entry)
                seen_sources.add(image_url)
                continue

            filename = f"q{variant_number}_{accepted_this_run + 1:03d}_{content_hash[:10]}.jpg"
            output_path = raw_dir / filename
            try:
                output_path.write_bytes(jpeg_content)
            except OSError as exc:
                manifest_entry["status"] = f"save_error: {type(exc).__name__}"
                append_manifest(manifest_path, manifest_entry)
                continue

            manifest_entry.update(
                {
                    "filename": filename,
                    "status": "accepted",
                    "sha256": content_hash,
                    "width": image.width,
                    "height": image.height,
                }
            )
            append_manifest(manifest_path, manifest_entry)
            seen_sources.add(image_url)
            seen_hashes.add(content_hash)
            accepted_this_run += 1
            accepted_for_query += 1
            progress.update(1)
            progress.set_postfix_str(f"{accepted_this_run}/{images_needed} accepted")

    progress.close()
    if accepted_this_run < images_needed:
        LOGGER.warning("%s: downloaded %d/%d additional valid images", player_name, accepted_this_run, images_needed)
    return accepted_this_run


def get_detector(device: str) -> Any:
    try:
        import torch
        from facenet_pytorch import MTCNN
    except ImportError as exc:
        raise RuntimeError("PyTorch and facenet-pytorch are required; install dependencies with `pip install -r requirements.txt`.") from exc
    selected_device = device
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    return MTCNN(keep_all=True, device=selected_device)


def get_ocr_reader(device: str) -> Any:
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError("EasyOCR is required for watermark filtering; install dependencies with `pip install -r requirements.txt`.") from exc
    return easyocr.Reader(["en"], gpu=device == "cuda")


def has_watermark_text(image: Any, ocr_reader: Any, confidence_threshold: float) -> bool:
    import numpy as np

    for detection in ocr_reader.readtext(np.asarray(image), detail=1, paragraph=False):
        if len(detection) >= 3 and str(detection[1]).strip() and float(detection[2]) > confidence_threshold:
            return True
    return False


def crop_with_padding(image: Any, box: tuple[float, float, float, float], padding: float) -> Any:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    left = max(0, math.floor(left - width * padding))
    top = max(0, math.floor(top - height * padding))
    right = min(image.width, math.ceil(right + width * padding))
    bottom = min(image.height, math.ceil(bottom + height * padding))
    return image.crop((left, top, right, bottom))


def crop_player(
    player_id: str,
    player_name: str,
    raw_root: Path,
    crop_root: Path,
    detector: Any,
    ocr_reader: Any,
    padding: float,
    target_crops: int,
    text_confidence: float,
    face_confidence: float,
) -> int:
    from tqdm import tqdm

    Image, _ = get_pillow()
    raw_dir = raw_root / player_id
    crop_dir = crop_root / player_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    ownership_path = crop_dir / ".pipeline_crops.json"
    owned_by_raw: dict[str, str] = {}
    if ownership_path.exists():
        try:
            ownership_data = json.loads(ownership_path.read_text(encoding="utf-8"))
            if isinstance(ownership_data, dict):
                owned_by_raw = {str(name): str(raw) for name, raw in ownership_data.items()}
            elif isinstance(ownership_data, list):
                owned_by_raw = {str(name): "" for name in ownership_data if isinstance(name, str)}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            LOGGER.warning("Could not read prior crop ownership manifest: %s", ownership_path)

    Image, _ = get_pillow()
    valid_crops: list[str] = []
    for crop_path in sorted(crop_dir.iterdir()):
        if not crop_path.is_file() or not re.fullmatch(r"\d{3,}\.jpg", crop_path.name, re.IGNORECASE):
            continue
        try:
            with Image.open(crop_path) as saved_crop:
                crop = saved_crop.convert("RGB")
                is_standard_size = saved_crop.size == IMAGE_SIZE and saved_crop.mode == "RGB"
            if not is_standard_size:
                continue
            if has_watermark_text(crop, ocr_reader, text_confidence):
                LOGGER.warning("Discarded watermarked crop for %s", player_name)
                crop_path.unlink(missing_ok=True)
                owned_by_raw.pop(crop_path.name, None)
                continue
            valid_crops.append(crop_path.name)
        except Exception as exc:
            LOGGER.warning("Could not validate existing crop %s (%s): %s", crop_path, type(exc).__name__, exc)

    crop_hashes: set[str] = set()
    for crop_name in valid_crops:
        try:
            crop_hashes.add(hashlib.sha256((crop_dir / crop_name).read_bytes()).hexdigest())
        except OSError:
            continue
    if len(valid_crops) >= target_crops:
        owned_by_raw = {name: raw for name, raw in owned_by_raw.items() if name in valid_crops}
        ownership_path.write_text(json.dumps(owned_by_raw, indent=2) + "\n", encoding="utf-8")
        return len(valid_crops)

    raw_paths = sorted(
        path for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ) if raw_dir.is_dir() else []
    raw_manifest_entries = load_manifest(raw_dir / "manifest.jsonl")
    credited_raw_files = {
        str(entry.get("filename")) for entry in raw_manifest_entries
        if entry.get("photographer_credit") or entry.get("status") == "photographer_credit_detected"
    }
    processed_raw = {raw for raw in owned_by_raw.values() if raw}
    with tqdm(raw_paths, desc=f"Crop {player_name}", unit="image") as progress:
        for raw_path in progress:
            if len(valid_crops) >= target_crops:
                break
            if raw_path.name in processed_raw:
                continue
            try:
                with Image.open(raw_path) as source:
                    image = source.convert("RGB")
                    original_content = raw_path.read_bytes()
                if raw_path.name in credited_raw_files:
                    LOGGER.warning("Skipping credited raw image %s", raw_path)
                    continue
                photographer_credit = find_image_photographer_credit(original_content)
                if photographer_credit:
                    LOGGER.warning("Skipping credited raw image %s: %s", raw_path, photographer_credit)
                    continue
                detections, probabilities = detector.detect(image)
                if detections is None or len(detections) == 0:
                    LOGGER.warning("No face detected; skipping %s", raw_path)
                    continue
                boxes = []
                for index, box in enumerate(detections):
                    if (
                        box is not None and probabilities is not None
                        and index < len(probabilities) and float(probabilities[index]) >= face_confidence
                    ):
                        boxes.append((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                if not boxes:
                    LOGGER.warning("No usable face box; skipping %s", raw_path)
                    continue
                largest_box = max(boxes, key=lambda box: max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
                face = crop_with_padding(image, largest_box, padding)
                if face.width == 0 or face.height == 0:
                    LOGGER.warning("Empty face crop; skipping %s", raw_path)
                    continue
                if has_watermark_text(face, ocr_reader, text_confidence):
                    LOGGER.warning("Discarded watermarked crop for %s", player_name)
                    continue
                crop_buffer = io.BytesIO()
                face.resize(IMAGE_SIZE, Image.Resampling.LANCZOS).save(
                    crop_buffer, format="JPEG", quality=95, optimize=True
                )
                crop_content = crop_buffer.getvalue()
                crop_hash = hashlib.sha256(crop_content).hexdigest()
                if crop_hash in crop_hashes:
                    processed_raw.add(raw_path.name)
                    continue
                used_numbers = {
                    int(match.group(1)) for name in valid_crops
                    if (match := re.fullmatch(r"(\d{3,})\.jpg", name, re.IGNORECASE))
                }
                output_number = 1
                while output_number in used_numbers or (crop_dir / f"{output_number:03d}.jpg").exists():
                    output_number += 1
                output_name = f"{output_number:03d}.jpg"
                (crop_dir / output_name).write_bytes(crop_content)
                valid_crops.append(output_name)
                crop_hashes.add(crop_hash)
                owned_by_raw[output_name] = raw_path.name
                processed_raw.add(raw_path.name)
            except Exception as exc:
                LOGGER.warning("Could not crop %s (%s): %s", raw_path, type(exc).__name__, exc)

    owned_by_raw = {name: raw for name, raw in owned_by_raw.items() if name in valid_crops}
    ownership_path.write_text(json.dumps(owned_by_raw, indent=2) + "\n", encoding="utf-8")
    return len(valid_crops)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "download", "crop"), default="all")
    parser.add_argument("--player", choices=tuple(PLAYERS), help="Process only this player (directory ID).")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--target-crops", type=int, default=60)
    parser.add_argument("--max-download-rounds", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--padding", type=float, default=PADDING_RATIO)
    parser.add_argument("--text-confidence", type=float, default=TEXT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--face-confidence", type=float, default=FACE_CONFIDENCE_THRESHOLD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    if (
        args.target_crops < 1 or args.max_download_rounds < 0 or args.timeout <= 0
        or args.padding < 0 or not 0 <= args.text_confidence <= 1
        or not 0 <= args.face_confidence <= 1
    ):
        LOGGER.error("Targets/timeouts must be positive; rounds and padding cannot be negative; confidence thresholds must be between 0 and 1.")
        return 2

    selected_players = [(key, value) for key, value in PLAYERS.items() if args.player is None or key == args.player]
    raw_root = args.data_dir / "raw"
    crop_root = args.data_dir / "cropped_faces"
    raw_counts: dict[str, int] = {}
    crop_counts: dict[str, int] = {}

    if args.stage == "download":
        for player_id, player_name in selected_players:
            manifest = load_manifest(raw_root / player_id / "manifest.jsonl")
            _, _, current_raw_count = existing_download_state(raw_root / player_id, manifest)
            download_player(
                player_id, player_name, raw_root,
                max(0, args.target_crops - current_raw_count), args.timeout,
            )
            manifest = load_manifest(raw_root / player_id / "manifest.jsonl")
            _, _, raw_counts[player_id] = existing_download_state(raw_root / player_id, manifest)

    if args.stage in {"all", "crop"}:
        detector = get_detector(args.device)
        ocr_reader = get_ocr_reader(args.device)
        for player_id, player_name in selected_players:
            crop_counts[player_id] = crop_player(
                player_id, player_name, raw_root, crop_root, detector, ocr_reader,
                args.padding, args.target_crops, args.text_confidence, args.face_confidence,
            )
            if args.stage == "all":
                for _ in range(args.max_download_rounds):
                    missing_crops = args.target_crops - crop_counts[player_id]
                    if missing_crops <= 0:
                        break
                    added = download_player(
                        player_id, player_name, raw_root, missing_crops, args.timeout
                    )
                    if added == 0:
                        break
                    crop_counts[player_id] = crop_player(
                        player_id, player_name, raw_root, crop_root, detector, ocr_reader,
                        args.padding, args.target_crops, args.text_confidence, args.face_confidence,
                    )
            manifest = load_manifest(raw_root / player_id / "manifest.jsonl")
            _, _, raw_counts[player_id] = existing_download_state(raw_root / player_id, manifest)

    print("\nPlayer                 Valid raw images    Clean face crops    Crop target shortfall")
    print("-" * 83)
    for player_id, player_name in selected_players:
        raw_count = raw_counts.get(player_id, 0)
        crop_count = crop_counts.get(player_id, 0)
        crop_display = str(crop_count) if args.stage != "download" else "not run"
        shortfall = max(0, args.target_crops - crop_count) if args.stage != "download" else "not run"
        print(f"{player_name:<22} {raw_count:>8}          {crop_display:>8}          {shortfall}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
