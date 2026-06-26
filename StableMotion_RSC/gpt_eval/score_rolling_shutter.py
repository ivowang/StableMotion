#!/usr/bin/env python3
"""Batch compare rolling-shutter correction results from triptych images."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

RSC_WEIGHTS = {
    "rs_suppression": 0.30,
    "geometry_structure": 0.25,
    "content_preservation": 0.20,
    "artifact_control": 0.15,
    "naturalness": 0.10,
}

SCORE_FIELDS = tuple(RSC_WEIGHTS.keys())
METHODS = ("yang", "ours")
METHOD_LABELS = {"yang": "YANG", "ours": "OURS"}
TIE_THRESHOLD = 0.25

SYSTEM_PROMPT = """You are an expert reviewer for single-image rolling shutter correction (RSC).

Task definition:
- The input is a rolling-shutter (RS) image captured with row-wise exposure, so straight structures may lean, bend, wobble, or curve, and scene content may be skewed by camera/device motion.
- The output is one candidate corrected image. A strong output should look closer to a global-shutter (GS) image: reduced row-wise skew/wobble, straighter buildings/poles/edges, plausible object shapes, preserved scene content, and no over-correction or new artifacts.

Score only what can be inferred from the two provided images: INPUT and OUTPUT. Do not compare against another method, because no other method output is visible in this conversation. Penalize residual RS distortion, over-correction, bent straight lines, stretched local regions, ghosting, tearing, duplicated content, blur/noise, severe crop, hallucinated or missing objects, and unnatural appearance.

Return JSON only. Use integer scores from 0 to 10 for the output:
1. rs_suppression: How well are rolling-shutter skew, wobble, bending, and row-wise deformation corrected?
2. geometry_structure: Are salient straight lines, buildings, poles, road edges, object shapes, and moving objects geometrically plausible and continuous?
3. content_preservation: Does the output keep the input scene content and field of view without excessive cropping, missing regions, or hallucination?
4. artifact_control: Are local warping artifacts, tearing, seams, ghosting, blur, duplicated content, and corruption avoided?
5. naturalness: Does the corrected image look like a coherent natural global-shutter photograph?

Use the full 0-10 range. A score of 10 is near publication-quality; 7-8 is good with minor issues; 4-6 has clear but not catastrophic problems; 1-3 is poor; 0 is unusable or impossible to judge."""

USER_PROMPT_TEMPLATE = """Evaluate pair_id={pair_id}, method={method}.

Image order:
1. INPUT rolling-shutter image.
2. OUTPUT corrected image from {method}.

Score only this OUTPUT against the INPUT using the RSC rubric. Keep the rationale concise and concrete."""

SCORING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pair_id": {"type": "string"},
        "method": {"type": "string", "enum": ["yang", "ours"]},
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rs_suppression": {"type": "integer", "minimum": 0, "maximum": 10},
                "geometry_structure": {"type": "integer", "minimum": 0, "maximum": 10},
                "content_preservation": {"type": "integer", "minimum": 0, "maximum": 10},
                "artifact_control": {"type": "integer", "minimum": 0, "maximum": 10},
                "naturalness": {"type": "integer", "minimum": 0, "maximum": 10},
            },
            "required": [
                "rs_suppression",
                "geometry_structure",
                "content_preservation",
                "artifact_control",
                "naturalness",
            ],
        },
        "rationale": {"type": "string"},
        "failure_modes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "residual_rs_distortion",
                    "over_correction",
                    "geometric_distortion",
                    "line_discontinuity",
                    "motion_object_artifact",
                    "tearing_or_ghosting",
                    "blur_or_noise",
                    "content_loss_or_crop",
                    "content_hallucination",
                    "color_or_exposure_shift",
                    "low_confidence",
                    "none",
                ],
            },
        },
        "confidence": {"type": "integer", "minimum": 0, "maximum": 10},
    },
    "required": ["pair_id", "method", "scores", "rationale", "failure_modes", "confidence"],
}


@dataclass
class TriptychImages:
    input_image: Image.Image
    yang_image: Image.Image
    ours_image: Image.Image


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.stem)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_concat_images(concat_dir: Path) -> list[Path]:
    return sorted((path for path in concat_dir.iterdir() if is_image_file(path)), key=natural_sort_key)


def sample_concat_paths(paths: list[Path], limit: int | None, rng: Any | None = None) -> list[Path]:
    if limit is None:
        return list(paths)
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    if limit >= len(paths):
        return list(paths)
    sampler = rng or random.SystemRandom()
    return sorted(sampler.sample(list(paths), limit), key=natural_sort_key)


def save_sample_manifest(manifest_path: Path, concat_dir: Path, paths: list[Path]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in paths:
        try:
            lines.append(str(path.relative_to(concat_dir)))
        except ValueError:
            lines.append(str(path))
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_sample_manifest(manifest_path: Path, concat_dir: Path) -> list[Path]:
    paths: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            path = Path(value)
            paths.append(path if path.is_absolute() else concat_dir / path)
    return paths


def split_horizontal_triptych(path: Path) -> TriptychImages:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width % 3 != 0:
            raise ValueError(f"Triptych width must be divisible by 3: {path} has {width}x{height}")
        tile_width = width // 3
        return TriptychImages(
            input_image=rgb.crop((0, 0, tile_width, height)),
            yang_image=rgb.crop((tile_width, 0, 2 * tile_width, height)),
            ours_image=rgb.crop((2 * tile_width, 0, width, height)),
        )


def image_to_png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def compute_overall_score(scores: dict[str, int | float]) -> float:
    total = 0.0
    for field, weight in RSC_WEIGHTS.items():
        total += float(scores[field]) * weight
    return round(total, 4)


def build_request_payload(
    pair_id: str,
    method: str,
    triptych: TriptychImages,
    model: str,
    detail: str,
    response_format: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    method_label = METHOD_LABELS[method]
    output_image = triptych.yang_image if method == "yang" else triptych.ours_image
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT_TEMPLATE.format(pair_id=pair_id, method=method)},
                    {"type": "input_text", "text": "INPUT rolling-shutter image:"},
                    {"type": "input_image", "image_url": image_to_png_data_url(triptych.input_image), "detail": detail},
                    {"type": "input_text", "text": f"{method_label} output:"},
                    {"type": "input_image", "image_url": image_to_png_data_url(output_image), "detail": detail},
                ],
            },
        ],
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    if response_format == "schema":
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": "rolling_shutter_pair_score",
                "strict": True,
                "schema": SCORING_SCHEMA,
            }
        }
    elif response_format == "json_object":
        payload["text"] = {"format": {"type": "json_object"}}
    elif response_format != "none":
        raise ValueError(f"Unknown response format: {response_format}")
    return payload


def responses_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return f"{base_url}/responses"
    return f"{base_url}/v1/responses"


def load_provider(provider_path: Path, provider_name: str | None) -> tuple[str, str, str]:
    data = json.loads(provider_path.read_text(encoding="utf-8"))
    if provider_name is None:
        if len(data) != 1:
            names = ", ".join(sorted(data))
            raise ValueError(f"Multiple providers found; choose one with --provider-name. Available: {names}")
        provider_name = next(iter(data))
    if provider_name not in data:
        names = ", ".join(sorted(data))
        raise ValueError(f"Provider {provider_name!r} not found. Available: {names}")
    provider = data[provider_name]
    if provider.get("api") != "openai-responses":
        raise ValueError(f"Provider {provider_name!r} has unsupported api={provider.get('api')!r}")
    base_url = str(provider["baseUrl"]).rstrip("/")
    api_key = str(provider["apiKey"])
    return provider_name, base_url, api_key


def post_json(url: str, api_key: str, payload: dict[str, Any], timeout: int, retries: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "curl/8.7.1",
    }
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read()
                try:
                    return json.loads(response_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if attempt >= retries:
                        snippet = response_body[:500].decode("utf-8", errors="replace")
                        raise RuntimeError(f"Invalid JSON response: {snippet}") from exc
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            if attempt >= retries:
                raise RuntimeError(f"Request failed: {exc}") from exc
        time.sleep(min(2**attempt, 8))
    raise RuntimeError("Request failed after retries")


def extract_json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def extract_response_json(response: dict[str, Any]) -> dict[str, Any]:
    if isinstance(response.get("output_parsed"), dict):
        return response["output_parsed"]

    text = response.get("output_text")
    if not isinstance(text, str):
        chunks: list[str] = []
        for output in response.get("output", []):
            for content in output.get("content", []):
                if isinstance(content, dict):
                    value = content.get("text")
                    if isinstance(value, str) and content.get("type") in {None, "output_text", "text"}:
                        chunks.append(value)
        text = "\n".join(chunks)

    if not text and response.get("choices"):
        message = response["choices"][0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(item.get("text", "") for item in content if isinstance(item, dict))

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Could not find JSON text in response keys: {sorted(response.keys())}")
    return extract_json_from_text(text)


def normalize_method_scores(scores: dict[str, Any]) -> dict[str, int | float]:
    normalized: dict[str, int | float] = {}
    for field in SCORE_FIELDS:
        if field not in scores:
            raise ValueError(f"Model response is missing score field {field!r}")
        value = int(scores[field])
        if value < 0 or value > 10:
            raise ValueError(f"Score {field!r}={value} is outside 0..10")
        normalized[field] = value
    normalized["overall"] = compute_overall_score(normalized)
    return normalized


def winner_from_delta(delta: float) -> str:
    if delta > TIE_THRESHOLD:
        return "ours"
    if delta < -TIE_THRESHOLD:
        return "yang"
    return "tie"


def normalize_method_result(expected_method: str, parsed: dict[str, Any]) -> dict[str, Any]:
    if expected_method not in METHODS:
        raise ValueError(f"Unknown method: {expected_method}")
    method = parsed.get("method")
    if method is not None and method != expected_method:
        raise ValueError(f"Expected method {expected_method!r}, got {method!r}")
    raw_scores = parsed.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("Model response is missing object field 'scores'")
    return {
        "scores": normalize_method_scores(raw_scores),
        "rationale": parsed.get("rationale", ""),
        "failure_modes": parsed.get("failure_modes", []),
        "confidence": parsed.get("confidence"),
    }


def normalize_pair_record(pair_id: str, concat_path: Path, method: str, parsed: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_method_result(method, parsed)
    return {
        "scoring_mode": "independent_pairwise",
        "pair_id": pair_id,
        "method": method,
        "input_path": f"{concat_path}#input",
        "output_path": f"{concat_path}#{method}",
        "concat_path": str(concat_path),
        "scores": normalized["scores"],
        "rationale": normalized["rationale"],
        "failure_modes": normalized["failure_modes"],
        "confidence": normalized["confidence"],
    }


def normalize_comparison_record(
    pair_id: str,
    concat_path: Path,
    method_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    method_scores = {}
    rationales = {}
    failure_modes = {}
    confidences = {}
    for method in METHODS:
        if method not in method_results:
            raise ValueError(f"Missing independent result for {method!r}")
        normalized = normalize_method_result(method, method_results[method])
        method_scores[method] = normalized["scores"]
        rationales[method] = normalized["rationale"]
        failure_modes[method] = normalized["failure_modes"]
        confidences[method] = normalized["confidence"]

    delta = round(float(method_scores["ours"]["overall"]) - float(method_scores["yang"]["overall"]), 4)
    return {
        "scoring_mode": "independent_pairwise",
        "pair_id": pair_id,
        "concat_path": str(concat_path),
        "scores": method_scores,
        "delta_ours_minus_yang": delta,
        "winner": winner_from_delta(delta),
        "model_winner": None,
        "preference_margin": None,
        "rationale": rationales,
        "failure_modes": failure_modes,
        "confidence": confidences,
    }


def build_comparison_record(pair_id: str, concat_path: str, pair_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    method_scores = {method: pair_records[method]["scores"] for method in METHODS}
    delta = round(float(method_scores["ours"]["overall"]) - float(method_scores["yang"]["overall"]), 4)
    return {
        "scoring_mode": "independent_pairwise",
        "pair_id": pair_id,
        "concat_path": concat_path,
        "scores": method_scores,
        "delta_ours_minus_yang": delta,
        "winner": winner_from_delta(delta),
        "model_winner": None,
        "preference_margin": None,
        "rationale": {method: pair_records[method]["rationale"] for method in METHODS},
        "failure_modes": {method: pair_records[method]["failure_modes"] for method in METHODS},
        "confidence": {method: pair_records[method]["confidence"] for method in METHODS},
    }


def aggregate_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "ci95_low": round(mean - half_width, 4),
        "ci95_high": round(mean + half_width, 4),
    }


def build_comparison_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    concat_paths: dict[str, str] = {}
    for record in records:
        pair_id = record["pair_id"]
        grouped.setdefault(pair_id, {})[record["method"]] = record
        concat_paths[pair_id] = record["concat_path"]
    comparison_records = []
    for pair_id in sorted(grouped, key=lambda value: natural_sort_key(Path(value))):
        method_records = grouped[pair_id]
        if all(method in method_records for method in METHODS):
            comparison_records.append(build_comparison_record(pair_id, concat_paths[pair_id], method_records))
    return comparison_records


def build_summary(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    comparison_records = build_comparison_records(records)
    winner_counts = {"ours": 0, "yang": 0, "tie": 0}
    for record in comparison_records:
        winner_counts[record["winner"]] += 1
    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_scored_pairs": len(records),
        "num_complete_comparisons": len(comparison_records),
        "weights": RSC_WEIGHTS,
        "tie_threshold": TIE_THRESHOLD,
        "config": config,
        "winner_counts": winner_counts,
        "winner_rates": {
            key: round(value / len(comparison_records), 4) if comparison_records else None
            for key, value in winner_counts.items()
        },
        "delta_ours_minus_yang": aggregate_values(
            [float(record["delta_ours_minus_yang"]) for record in comparison_records]
        ),
        "methods": {},
    }
    for method in METHODS:
        method_records = [record for record in records if record["method"] == method]
        method_summary = {
            "num_scored_pairs": len(method_records),
            "overall": aggregate_values([float(record["scores"]["overall"]) for record in method_records]),
        }
        for field in SCORE_FIELDS:
            method_summary[field] = aggregate_values([float(record["scores"][field]) for record in method_records])
        summary["methods"][method] = method_summary
    return summary


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id",
        "method",
        "input_path",
        "output_path",
        "concat_path",
        "overall",
        *SCORE_FIELDS,
        "confidence",
        "failure_modes",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "pair_id": record["pair_id"],
                "method": record["method"],
                "input_path": record["input_path"],
                "output_path": record["output_path"],
                "concat_path": record["concat_path"],
                "overall": record["scores"]["overall"],
                "confidence": record.get("confidence"),
                "failure_modes": ";".join(record.get("failure_modes", [])),
            }
            row.update({field: record["scores"][field] for field in SCORE_FIELDS})
            writer.writerow(row)


def write_comparison_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id",
        "concat_path",
        "yang_overall",
        "ours_overall",
        "delta_ours_minus_yang",
        "winner",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "pair_id": record["pair_id"],
                    "concat_path": record["concat_path"],
                    "yang_overall": record["scores"]["yang"]["overall"],
                    "ours_overall": record["scores"]["ours"]["overall"],
                    "delta_ours_minus_yang": record["delta_ours_minus_yang"],
                    "winner": record["winner"],
                }
            )


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_run_outputs(output_dir: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "pair_scores.jsonl", records)
    write_csv(output_dir / "pair_scores.csv", records)
    comparison_records = build_comparison_records(records)
    write_jsonl(output_dir / "comparison_pairs.jsonl", comparison_records)
    write_comparison_csv(output_dir / "comparison_pairs.csv", comparison_records)
    (output_dir / "summary.json").write_text(
        json.dumps(build_summary(records, config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "rubric_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")


def default_output_dir(concat_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("scores_rsc") / f"{concat_dir.name}_{timestamp}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Yang and Ours rolling-shutter correction outputs with a GPT-5.4-compatible API."
    )
    parser.add_argument("concat_dir", type=Path, help="Folder of horizontal triptychs: input, Yang, Ours.")
    parser.add_argument("--provider", type=Path, default=Path("provider.json"), help="Provider JSON path.")
    parser.add_argument("--provider-name", default=None, help="Provider key in provider.json.")
    parser.add_argument("--model", default="gpt-5.4", help="Model name to send to the provider.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory for pair_scores and summary.")
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="high", help="Vision detail level.")
    parser.add_argument(
        "--response-format",
        choices=["schema", "json_object", "none"],
        default="schema",
        help="Responses API output formatting mode.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"], default="none")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between successful requests.")
    parser.add_argument("--limit", type=int, default=None, help="Randomly sample and score N triptychs.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible --limit sampling. Omit for system-random sampling.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip pair_ids already present in pair_scores.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Validate splitting and write metadata without API calls.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    concat_dir = args.concat_dir.resolve()
    if not concat_dir.is_dir():
        raise SystemExit(f"Concat directory does not exist: {concat_dir}")

    save_dir = (args.save_dir or default_output_dir(args.concat_dir)).resolve()
    all_concat_paths = list_concat_images(concat_dir)
    sample_manifest_path = save_dir / "sample_manifest.txt"
    if args.limit is not None and args.resume and sample_manifest_path.exists():
        concat_paths = load_sample_manifest(sample_manifest_path, concat_dir)
        sampling_source = "existing_manifest"
    else:
        rng = random.Random(args.sample_seed) if args.sample_seed is not None else None
        concat_paths = sample_concat_paths(all_concat_paths, args.limit, rng=rng)
        sampling_source = "random_sample" if args.limit is not None and args.limit < len(all_concat_paths) else "all"
    if args.limit is not None:
        save_sample_manifest(sample_manifest_path, concat_dir, concat_paths)

    config = {
        "concat_dir": str(concat_dir),
        "provider": str(args.provider),
        "provider_name": args.provider_name,
        "model": args.model,
        "detail": args.detail,
        "response_format": args.response_format,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "sampling_source": sampling_source,
        "scoring_mode": "independent_pairwise",
        "num_available_triptychs": len(all_concat_paths),
        "num_triptychs": len(concat_paths),
        "dry_run": args.dry_run,
    }

    pair_scores_path = save_dir / "pair_scores.jsonl"
    records = load_existing_records(pair_scores_path) if args.resume else []
    records = [
        record
        for record in records
        if record.get("scoring_mode") == "independent_pairwise" and record.get("method") in METHODS
    ]
    done_keys = {(record["pair_id"], record["method"]) for record in records}

    if args.dry_run:
        for path in concat_paths:
            split_horizontal_triptych(path)
        save_run_outputs(save_dir, records, config)
        print(f"Validated triptychs: {len(concat_paths)}")
        print(f"Dry-run metadata saved to: {save_dir}")
        return 0

    provider_name, base_url, api_key = load_provider(args.provider, args.provider_name)
    config["provider_name"] = provider_name
    url = responses_url(base_url)
    raw_dir = save_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for index, concat_path in enumerate(concat_paths, start=1):
        pair_id = concat_path.stem
        triptych = split_horizontal_triptych(concat_path)
        for method in METHODS:
            if (pair_id, method) in done_keys:
                print(f"[{index}/{len(concat_paths)}] skip existing {pair_id} / {method}")
                continue
            print(f"[{index}/{len(concat_paths)}] scoring {pair_id} / {method}")
            payload = build_request_payload(
                pair_id=pair_id,
                method=method,
                triptych=triptych,
                model=args.model,
                detail=args.detail,
                response_format=args.response_format,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            response = post_json(
                url=url, api_key=api_key, payload=payload, timeout=args.timeout, retries=args.retries
            )
            (raw_dir / f"{pair_id}_{method}.json").write_text(
                json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            parsed = extract_response_json(response)
            record = normalize_pair_record(pair_id=pair_id, concat_path=concat_path, method=method, parsed=parsed)
            records.append(record)
            done_keys.add((pair_id, method))
            save_run_outputs(save_dir, records, config)
            if args.sleep > 0:
                time.sleep(args.sleep)

    save_run_outputs(save_dir, records, config)
    print(f"Scored method pairs: {len(records)}")
    print(f"Complete comparisons: {len(build_comparison_records(records))}")
    print(f"Results saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
