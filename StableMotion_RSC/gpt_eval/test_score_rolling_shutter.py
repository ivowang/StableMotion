import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import score_rolling_shutter as rsc


class ScoreRollingShutterTest(unittest.TestCase):
    def test_split_horizontal_triptych_returns_three_equal_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concat.png"
            image = Image.new("RGB", (9, 2))
            for x in range(3):
                for y in range(2):
                    image.putpixel((x, y), (255, 0, 0))
            for x in range(3, 6):
                for y in range(2):
                    image.putpixel((x, y), (0, 255, 0))
            for x in range(6, 9):
                for y in range(2):
                    image.putpixel((x, y), (0, 0, 255))
            image.save(path)

            crops = rsc.split_horizontal_triptych(path)

            self.assertEqual(crops.input_image.size, (3, 2))
            self.assertEqual(crops.yang_image.size, (3, 2))
            self.assertEqual(crops.ours_image.size, (3, 2))
            self.assertGreater(crops.input_image.getpixel((1, 1))[0], 200)
            self.assertGreater(crops.yang_image.getpixel((1, 1))[1], 200)
            self.assertGreater(crops.ours_image.getpixel((1, 1))[2], 200)

    def test_compute_overall_score_uses_rsc_weights(self):
        scores = {
            "rs_suppression": 8,
            "geometry_structure": 6,
            "content_preservation": 4,
            "artifact_control": 10,
            "naturalness": 2,
        }

        overall = rsc.compute_overall_score(scores)

        self.assertAlmostEqual(overall, 6.4)

    def test_build_request_payload_contains_only_input_and_one_method_output(self):
        triptych = rsc.TriptychImages(
            input_image=Image.new("RGB", (2, 2), (255, 0, 0)),
            yang_image=Image.new("RGB", (2, 2), (0, 255, 0)),
            ours_image=Image.new("RGB", (2, 2), (0, 0, 255)),
        )

        payload = rsc.build_request_payload(
            pair_id="RE_frame-1",
            method="yang",
            triptych=triptych,
            model="gpt-5.4",
            detail="low",
            response_format="none",
            max_output_tokens=100,
            reasoning_effort="none",
        )

        content = payload["input"][1]["content"]
        image_items = [item for item in content if item["type"] == "input_image"]
        text = "\n".join(item["text"] for item in content if item["type"] == "input_text")
        self.assertEqual(len(image_items), 2)
        self.assertIn("YANG output", text)
        self.assertNotIn("OURS output", text)

    def test_normalize_comparison_record_computes_delta_and_winner_from_independent_results(self):
        yang_result = {
            "pair_id": "RE_frame-1",
            "method": "yang",
            "scores": {
                "rs_suppression": 4,
                "geometry_structure": 4,
                "content_preservation": 8,
                "artifact_control": 5,
                "naturalness": 5,
            },
            "rationale": "still skewed",
            "failure_modes": ["residual_rs_distortion"],
            "confidence": 8,
        }
        ours_result = {
            "pair_id": "RE_frame-1",
            "method": "ours",
            "scores": {
                "rs_suppression": 8,
                "geometry_structure": 7,
                "content_preservation": 8,
                "artifact_control": 7,
                "naturalness": 7,
            },
            "rationale": "straighter",
            "failure_modes": ["none"],
            "confidence": 8,
        }

        record = rsc.normalize_comparison_record(
            pair_id="RE_frame-1",
            concat_path=Path("RS_Yang_Ours/RE_frame-1.jpg"),
            method_results={"yang": yang_result, "ours": ours_result},
        )

        self.assertEqual(record["scoring_mode"], "independent_pairwise")
        self.assertEqual(record["winner"], "ours")
        self.assertGreater(record["delta_ours_minus_yang"], 0)
        self.assertIn("overall", record["scores"]["yang"])
        self.assertIn("overall", record["scores"]["ours"])

    def test_normalize_pair_record_matches_rectangle_like_shape(self):
        parsed = {
            "pair_id": "RE_frame-1",
            "method": "ours",
            "scores": {
                "rs_suppression": 8,
                "geometry_structure": 7,
                "content_preservation": 8,
                "artifact_control": 7,
                "naturalness": 7,
            },
            "rationale": "straighter geometry with minor crop",
            "failure_modes": ["content_loss_or_crop"],
            "confidence": 8,
        }

        record = rsc.normalize_pair_record(
            pair_id="RE_frame-1",
            concat_path=Path("RS_Yang_Ours/RE_frame-1.jpg"),
            method="ours",
            parsed=parsed,
        )

        self.assertEqual(record["pair_id"], "RE_frame-1")
        self.assertEqual(record["method"], "ours")
        self.assertIn("input_path", record)
        self.assertIn("output_path", record)
        self.assertIn("overall", record["scores"])
        self.assertEqual(record["rationale"], "straighter geometry with minor crop")

    def test_extract_response_json_reuses_responses_output_text(self):
        response = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"winner": "tie"}),
                        }
                    ]
                }
            ]
        }

        parsed = rsc.extract_response_json(response)

        self.assertEqual(parsed["winner"], "tie")

    def test_sample_concat_paths_uses_random_sample_instead_of_prefix(self):
        paths = [Path(f"frame-{index}.jpg") for index in range(10)]

        class ReverseSampler:
            def sample(self, population, k):
                return list(reversed(population))[:k]

        selected = rsc.sample_concat_paths(paths, limit=3, rng=ReverseSampler())

        self.assertEqual([path.name for path in selected], ["frame-7.jpg", "frame-8.jpg", "frame-9.jpg"])
        self.assertNotEqual(selected, paths[:3])

    def test_load_sample_manifest_reuses_previous_limit_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / f"frame-{index}.jpg" for index in range(5)]
            for path in paths:
                path.write_bytes(b"image")
            manifest = root / "sample_manifest.txt"
            manifest.write_text("frame-3.jpg\nframe-1.jpg\n", encoding="utf-8")

            selected = rsc.load_sample_manifest(manifest, root)

            self.assertEqual([path.name for path in selected], ["frame-3.jpg", "frame-1.jpg"])


if __name__ == "__main__":
    unittest.main()
