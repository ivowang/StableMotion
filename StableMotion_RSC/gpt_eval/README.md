# GPT Rule-Based Evaluation — RSC

Rule-based perceptual evaluation of **Rolling Shutter Correction (RSC)** results
with a vision LLM (GPT). `score_rolling_shutter.py` splits each horizontal triptych
`[input | Yang | Ours]`, scores every method independently against the input on a
fixed rubric, then derives per-pair winners and win rates. This is the script used
for the GPT-judge RSC experiments in the paper, and it is self-contained (no
cross-folder imports).

Each output is scored 0–10 on five dimensions — rs suppression, geometry/structure,
content preservation, artifact control, naturalness — combined with fixed weights
into a per-image `overall` score. The per-pair `winner` (ours / yang / tie) comes
from the overall-score delta with a small tie threshold.

## Requirements
Python 3.10+ and `pillow` (already in the repo's `requirements.txt`).

## Configure the provider
```bash
cp provider.example.json provider.json   # provider.json is gitignored
```
```json
{
  "my-provider": {
    "api": "openai-responses",
    "baseUrl": "https://api.openai.com",
    "apiKey": "sk-..."
  }
}
```
`baseUrl` may point at any OpenAI-Responses-compatible endpoint. With a single
provider entry the key is auto-selected; otherwise pass `--provider-name`.

## Usage
`concat_dir` holds horizontal triptychs, each `[input | Yang | Ours]` (width
divisible by 3).
```bash
python score_rolling_shutter.py <concat_dir> \
    --provider provider.json --model gpt-5.4 --save-dir scores_rsc/rsc_run
```
Useful flags: `--dry-run` (validate triptych splitting and write metadata without
any API call), `--resume`, `--limit N` with `--sample-seed` (reproducible random
subset), `--detail {low,high,auto}`,
`--reasoning-effort {none,minimal,low,medium,high}`, `--max-output-tokens`,
`--sleep`. The `--model` default (`gpt-5.4`) matches the paper; change it to
whatever your provider serves.

## Outputs (in `--save-dir`)
- `pair_scores.jsonl` / `pair_scores.csv` — per-method scores, rationales, failure modes.
- `comparison_pairs.jsonl` / `comparison_pairs.csv` — per-pair Ours-vs-Yang delta and winner.
- `summary.json` — rubric weights, win counts/rates, per-method mean, std, 95% CI.
- `rubric_prompt.txt` — the exact system prompt used.
- `raw_responses/` — the unmodified API response for every method.
- `sample_manifest.txt` — the sampled triptychs when `--limit` is used.

Runs are incremental: each response is saved immediately, and `--resume` skips
`(pair, method)` combinations already present in `pair_scores.jsonl`.

## Tests
```bash
python -m pytest          # or: python -m unittest discover -p 'test_*.py'
```
