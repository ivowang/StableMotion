import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_rectangle


class ScoreRectangleTest(unittest.TestCase):
    def test_find_pairs_matches_by_stem_and_reports_missing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            (input_dir / "000001.jpg").write_bytes(b"a")
            (input_dir / "000002.jpeg").write_bytes(b"b")
            (input_dir / "notes.txt").write_text("ignore")
            (output_dir / "000001.png").write_bytes(b"c")
            (output_dir / "unused.png").write_bytes(b"d")

            pairs, missing_outputs, unused_outputs = score_rectangle.find_pairs(input_dir, output_dir)

            self.assertEqual([pair.pair_id for pair in pairs], ["000001"])
            self.assertEqual([path.name for path in missing_outputs], ["000002.jpeg"])
            self.assertEqual([path.name for path in unused_outputs], ["unused.png"])

    def test_compute_weighted_overall_score_uses_rubric_weights(self):
        scores = {
            "rectangular_boundary": 8,
            "content_preservation": 6,
            "geometry_structure": 4,
            "artifact_control": 10,
            "naturalness": 2,
        }

        overall = score_rectangle.compute_overall_score(scores)

        self.assertAlmostEqual(overall, 6.2)

    def test_build_summary_includes_mean_std_and_pair_count(self):
        records = [
            {
                "pair_id": "a",
                "scores": {
                    "overall": 8.0,
                    "rectangular_boundary": 8,
                    "content_preservation": 8,
                    "geometry_structure": 8,
                    "artifact_control": 8,
                    "naturalness": 8,
                },
            },
            {
                "pair_id": "b",
                "scores": {
                    "overall": 6.0,
                    "rectangular_boundary": 6,
                    "content_preservation": 6,
                    "geometry_structure": 6,
                    "artifact_control": 6,
                    "naturalness": 6,
                },
            },
            {
                "pair_id": "c",
                "scores": {
                    "overall": 4.0,
                    "rectangular_boundary": 4,
                    "content_preservation": 4,
                    "geometry_structure": 4,
                    "artifact_control": 4,
                    "naturalness": 4,
                },
            },
        ]

        summary = score_rectangle.build_summary(
            records=records,
            missing_outputs=[Path("missing.jpg")],
            unused_outputs=[Path("unused.png")],
            config={"model": "gpt-5.4"},
        )

        self.assertEqual(summary["num_scored_pairs"], 3)
        self.assertEqual(summary["num_missing_outputs"], 1)
        self.assertEqual(summary["num_unused_outputs"], 1)
        self.assertEqual(summary["overall"]["mean"], 6.0)
        self.assertAlmostEqual(summary["overall"]["std"], 2.0)
        self.assertIn("ci95_low", summary["overall"])
        self.assertEqual(summary["config"]["model"], "gpt-5.4")

    def test_extract_json_from_response_output_text(self):
        response = {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"scores": {"rectangular_boundary": 7}}),
                        }
                    ]
                }
            ]
        }

        parsed = score_rectangle.extract_response_json(response)

        self.assertEqual(parsed["scores"]["rectangular_boundary"], 7)

    def test_build_request_payload_can_omit_reasoning_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jpg"
            output_path = root / "output.png"
            input_path.write_bytes(b"input")
            output_path.write_bytes(b"output")
            pair = score_rectangle.ImagePair("sample", input_path, output_path)

            payload = score_rectangle.build_request_payload(
                pair=pair,
                model="gpt-5.4",
                detail="low",
                response_format="none",
                max_output_tokens=100,
                reasoning_effort="none",
            )

            self.assertNotIn("reasoning", payload)

    def test_post_json_sets_gateway_friendly_headers(self):
        class FakeResponse:
            def __init__(self, body=b"{}"):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return self.body

        captured = {}

        def fake_urlopen(request, timeout):
            captured["headers"] = dict(request.header_items())
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            score_rectangle.post_json(
                url="https://example.test/v1/responses",
                api_key="test-key",
                payload={"model": "gpt-5.4", "input": "ok"},
                timeout=1,
                retries=0,
            )

        self.assertEqual(captured["headers"]["User-agent"], "curl/8.7.1")
        self.assertEqual(captured["headers"]["Accept"], "application/json")

    def test_post_json_retries_transient_non_json_response(self):
        responses = [b"", b'{"ok": true}']

        def fake_urlopen(request, timeout):
            return type(
                "FakeResponse",
                (),
                {
                    "__enter__": lambda self: self,
                    "__exit__": lambda self, exc_type, exc, traceback: False,
                    "read": lambda self: responses.pop(0),
                },
            )()

        with (
            mock.patch("urllib.request.urlopen", side_effect=fake_urlopen),
            mock.patch("time.sleep"),
        ):
            result = score_rectangle.post_json(
                url="https://example.test/v1/responses",
                api_key="test-key",
                payload={"model": "gpt-5.4", "input": "ok"},
                timeout=1,
                retries=1,
            )

        self.assertEqual(result, {"ok": True})


if __name__ == "__main__":
    unittest.main()
