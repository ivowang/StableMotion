# GPT Rule-Based Evaluation — SIR

Rule-based perceptual evaluation of **Stitched Image Rectangling (SIR)** results
with a vision LLM (GPT). `score_rectangle.py` scores each `(input, rectangling
output)` pair on a fixed rubric, so the model acts as a rule-following judge rather
than a free-form critic. This is the script used for the GPT-judge SIR experiments
in the paper.

Each output is scored 0–10 on five dimensions — rectangular boundary, content
preservation, geometry/structure, artifact control, naturalness — which are
combined with fixed weights into a per-image `overall` score.

## Requirements
Python 3.10+, standard library only (no extra packages).

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
Input images and rectangling outputs are matched by filename stem.
```bash
python score_rectangle.py <input_dir> <result_dir> \
    --provider provider.json --model gpt-5.4 --save-dir scores/sir_run
```
Useful flags: `--dry-run` (build pairs and write metadata without any API call),
`--resume`, `--limit N`, `--detail {low,high,auto}`,
`--reasoning-effort {none,minimal,low,medium,high}`, `--max-output-tokens`,
`--sleep`. The `--model` default (`gpt-5.4`) matches the paper; change it to
whatever your provider serves.

## Outputs (in `--save-dir`)
- `pair_scores.jsonl` / `pair_scores.csv` — per-image scores, rationales, failure modes.
- `summary.json` — rubric weights and per-dimension/overall mean, std, 95% CI.
- `rubric_prompt.txt` — the exact system prompt used.
- `raw_responses/` — the unmodified API response for every pair.

Runs are incremental: each response is saved immediately, and `--resume` skips
pairs already present in `pair_scores.jsonl`.

## Tests
```bash
python -m pytest          # or: python -m unittest discover -p 'test_*.py'
```
