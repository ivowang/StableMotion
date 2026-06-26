#!/usr/bin/env python3
"""Batch score stitched-image rectangling results with a vision LLM."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import mimetypes
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

RUBRIC_WEIGHTS = {
    "rectangular_boundary": 0.25,
    "content_preservation": 0.25,
    "geometry_structure": 0.25,
    "artifact_control": 0.15,
    "naturalness": 0.10,
}

SCORE_FIELDS = tuple(RUBRIC_WEIGHTS.keys())

SYSTEM_PROMPT = """You are an expert reviewer for stitched image rectangling (SIR).

Task definition:
- The input image is a stitched image with irregular boundaries, white/empty regions, and possibly geometric deformation.
- The output image is the candidate rectangling result.
- A strong result should form a clean rectangular image while preserving the input content and field of view, keeping scene geometry natural, avoiding line discontinuities/local distortions, suppressing white borders and warping artifacts, and not hallucinating unrelated new content.

Score only what can be inferred by comparing the two provided images. Penalize outputs that crop away large valid content, add unrelated objects, leave white/empty borders, bend salient structures, break continuous lines, duplicate content, blur details, or introduce seams/noise.

Return JSON only. Use integer scores from 0 to 10 for each rubric dimension:
1. rectangular_boundary: Is the output a complete, clean rectangle without visible empty/white/invalid borders?
2. content_preservation: Does the output preserve the input scene content and field of view without excessive cropping or unrelated hallucination?
3. geometry_structure: Are salient geometry, straight lines, object shapes, and non-linear structures visually plausible and continuous?
4. artifact_control: Are warping artifacts, seams, discontinuities, blur, noise, duplicated regions, and local corruption avoided?
5. naturalness: Does the final image look like a coherent natural photograph?

Use the full 0-10 range. A score of 10 is near publication-quality; 7-8 is good with minor issues; 4-6 has clear but not catastrophic problems; 1-3 is poor; 0 is unusable or impossible to judge."""

USER_PROMPT_TEMPLATE = """Evaluate pair_id={pair_id}.

Image order:
1. INPUT stitched image.
2. OUTPUT rectangling result.

Compare OUTPUT against INPUT using the SIR rubric. Keep the rationale concise and concrete."""


SCORING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pair_id": {"type": "string"},
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rectangular_boundary": {"type": "integer", "minimum": 0, "maximum": 10},
                "content_preservation": {"type": "integer", "minimum": 0, "maximum": 10},
                "geometry_structure": {"type": "integer", "minimum": 0, "maximum": 10},
                "artifact_control": {"type": "integer", "minimum": 0, "maximum": 10},
                "naturalness": {"type": "integer", "minimum": 0, "maximum": 10},
            },
            "required": [
                "rectangular_boundary",
                "content_preservation",
                "geometry_structure",
                "artifact_control",
                "naturalness",
            ],
        },
        "rationale": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rectangular_boundary": {"type": "string"},
                "content_preservation": {"type": "string"},
                "geometry_structure": {"type": "string"},
                "artifact_control": {"type": "string"},
                "naturalness": {"type": "string"},
            },
            "required": [
                "rectangular_boundary",
                "content_preservation",
                "geometry_structure",
                "artifact_control",
                "naturalness",
            ],
        },
        "failure_modes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "white_or_empty_border",
                    "excessive_cropping",
                    "content_hallucination",
                    "geometric_distortion",
                    "line_discontinuity",
                    "seam_or_blending_artifact",
                    "blur_or_noise",
                    "duplicated_content",
                    "color_or_exposure_shift",
                    "low_confidence",
                    "none",
                ],
            },
        },
        "confidence": {"type": "integer", "minimum": 0, "maximum": 10},
    },
    "required": ["pair_id", "scores", "rationale", "failure_modes", "confidence"],
}


@dataclass(frozen=True)
class ImagePair:
    pair_id: str
    input_path: Path
    output_path: Path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if is_image_file(path))


def find_pairs(input_dir: Path, output_dir: Path) -> tuple[list[ImagePair], list[Path], list[Path]]:
    input_images = list_images(input_dir)
    output_images = list_images(output_dir)
    outputs_by_stem = {path.stem: path for path in output_images}
    inputs_by_stem = {path.stem: path for path in input_images}

    pairs: list[ImagePair] = []
    missing_outputs: list[Path] = []
    for input_path in input_images:
        output_path = outputs_by_stem.get(input_path.stem)
        if output_path is None:
            missing_outputs.append(input_path)
            continue
        pairs.append(ImagePair(pair_id=input_path.stem, input_path=input_path, output_path=output_path))

    unused_outputs = [path for path in output_images if path.stem not in inputs_by_stem]
    return pairs, missing_outputs, unused_outputs


def compute_overall_score(scores: dict[str, int | float]) -> float:
    total = 0.0
    for field, weight in RUBRIC_WEIGHTS.items():
        total += float(scores[field]) * weight
    return round(total, 4)


def build_summary(
    records: list[dict[str, Any]],
    missing_outputs: list[Path],
    unused_outputs: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    overall_scores = [float(record["scores"]["overall"]) for record in records]
    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_scored_pairs": len(records),
        "num_missing_outputs": len(missing_outputs),
        "num_unused_outputs": len(unused_outputs),
        "missing_outputs": [str(path) for path in missing_outputs],
        "unused_outputs": [str(path) for path in unused_outputs],
        "weights": RUBRIC_WEIGHTS,
        "config": config,
        "overall": aggregate_values(overall_scores),
        "dimensions": {},
    }
    for field in SCORE_FIELDS:
        values = [float(record["scores"][field]) for record in records]
        summary["dimensions"][field] = aggregate_values(values)
    return summary


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


def responses_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return f"{base_url}/responses"
    return f"{base_url}/v1/responses"


def image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_request_payload(
    pair: ImagePair,
    model: str,
    detail: str,
    response_format: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT_TEMPLATE.format(pair_id=pair.pair_id)},
                    {"type": "input_text", "text": "INPUT stitched image:"},
                    {"type": "input_image", "image_url": image_to_data_url(pair.input_path), "detail": detail},
                    {"type": "input_text", "text": "OUTPUT rectangling result:"},
                    {"type": "input_image", "image_url": image_to_data_url(pair.output_path), "detail": detail},
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
                "name": "rectangle_quality_score",
                "strict": True,
                "schema": SCORING_SCHEMA,
            }
        }
    elif response_format == "json_object":
        payload["text"] = {"format": {"type": "json_object"}}
    elif response_format != "none":
        raise ValueError(f"Unknown response format: {response_format}")
    return payload


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


def normalize_score_record(pair: ImagePair, parsed: dict[str, Any]) -> dict[str, Any]:
    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("Model response is missing object field 'scores'")
    normalized_scores: dict[str, int | float] = {}
    for field in SCORE_FIELDS:
        if field not in scores:
            raise ValueError(f"Model response is missing score field {field!r}")
        value = int(scores[field])
        if value < 0 or value > 10:
            raise ValueError(f"Score {field!r}={value} is outside 0..10")
        normalized_scores[field] = value
    normalized_scores["overall"] = compute_overall_score(normalized_scores)
    return {
        "pair_id": pair.pair_id,
        "input_path": str(pair.input_path),
        "output_path": str(pair.output_path),
        "scores": normalized_scores,
        "rationale": parsed.get("rationale", {}),
        "failure_modes": parsed.get("failure_modes", []),
        "confidence": parsed.get("confidence"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id",
        "input_path",
        "output_path",
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
                "input_path": record["input_path"],
                "output_path": record["output_path"],
                "overall": record["scores"]["overall"],
                "confidence": record.get("confidence"),
                "failure_modes": ";".join(record.get("failure_modes", [])),
            }
            row.update({field: record["scores"][field] for field in SCORE_FIELDS})
            writer.writerow(row)


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


def save_run_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    missing_outputs: list[Path],
    unused_outputs: list[Path],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "pair_scores.jsonl", records)
    write_csv(output_dir / "pair_scores.csv", records)
    summary = build_summary(records, missing_outputs, unused_outputs, config)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rubric_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")


def default_output_dir(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("scores") / f"{output_dir.name}_{timestamp}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score stitched-image rectangling outputs with a GPT-5.4-compatible Responses API."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing input stitched images.")
    parser.add_argument("result_dir", type=Path, help="Folder containing rectangling output images.")
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
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high"], default="low")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between successful requests.")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N matched pairs.")
    parser.add_argument("--resume", action="store_true", help="Skip pair_ids already present in pair_scores.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Build pairs and write metadata without API calls.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    input_dir = args.input_dir.resolve()
    result_dir = args.result_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not result_dir.is_dir():
        raise SystemExit(f"Result directory does not exist: {result_dir}")

    save_dir = (args.save_dir or default_output_dir(args.result_dir)).resolve()
    pairs, missing_outputs, unused_outputs = find_pairs(input_dir, result_dir)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    config = {
        "input_dir": str(input_dir),
        "result_dir": str(result_dir),
        "provider": str(args.provider),
        "provider_name": args.provider_name,
        "model": args.model,
        "detail": args.detail,
        "response_format": args.response_format,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "limit": args.limit,
        "num_matched_pairs": len(pairs),
        "dry_run": args.dry_run,
    }

    pair_scores_path = save_dir / "pair_scores.jsonl"
    records = load_existing_records(pair_scores_path) if args.resume else []
    done_pair_ids = {record["pair_id"] for record in records}

    if args.dry_run:
        save_run_outputs(save_dir, records, missing_outputs, unused_outputs, config)
        print(f"Matched pairs: {len(pairs)}")
        print(f"Missing outputs: {len(missing_outputs)}")
        print(f"Unused outputs: {len(unused_outputs)}")
        print(f"Dry-run metadata saved to: {save_dir}")
        return 0

    provider_name, base_url, api_key = load_provider(args.provider, args.provider_name)
    config["provider_name"] = provider_name
    url = responses_url(base_url)
    raw_dir = save_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for index, pair in enumerate(pairs, start=1):
        if pair.pair_id in done_pair_ids:
            print(f"[{index}/{len(pairs)}] skip existing {pair.pair_id}")
            continue
        print(f"[{index}/{len(pairs)}] scoring {pair.pair_id}")
        payload = build_request_payload(
            pair=pair,
            model=args.model,
            detail=args.detail,
            response_format=args.response_format,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        response = post_json(url=url, api_key=api_key, payload=payload, timeout=args.timeout, retries=args.retries)
        (raw_dir / f"{pair.pair_id}.json").write_text(
            json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        parsed = extract_response_json(response)
        record = normalize_score_record(pair, parsed)
        records.append(record)
        save_run_outputs(save_dir, records, missing_outputs, unused_outputs, config)
        if args.sleep > 0:
            time.sleep(args.sleep)

    save_run_outputs(save_dir, records, missing_outputs, unused_outputs, config)
    print(f"Scored pairs: {len(records)}")
    print(f"Results saved to: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
