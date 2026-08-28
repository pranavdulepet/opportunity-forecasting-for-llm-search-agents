import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from opportunity_forecasting.data.merge_labels import MergeError, merge_data_shards


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(", ", ": ")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class MergeDataShardsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_globbed_named_shards_restore_order_and_keep_duplicate_states(self):
        rows = [
            {"goal": "same", "step": 2, "label": "first"},
            {"goal": "other", "step": 1, "label": "x"},
            {"goal": "same", "step": 2, "label": "second"},
            {"goal": "third", "step": 4, "label": "y"},
            {"goal": "same", "step": 2, "label": "third"},
        ]
        for index in range(3):
            _write_jsonl(
                self.tmp_path / f"labels_shard_idx={index}_num_shards=3.jsonl",
                rows[index::3],
            )
        output = self.tmp_path / "merged.jsonl"

        report = merge_data_shards(
            [str(self.tmp_path / "labels_shard_idx=*_num_shards=3.jsonl")],
            output,
        )

        self.assertEqual(_read_jsonl(output), rows)
        duplicate_labels = [
            row["label"] for row in _read_jsonl(output) if row["goal"] == "same"
        ]
        self.assertEqual(duplicate_labels, ["first", "second", "third"])
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["rows"], 5)
        self.assertEqual(
            output.read_text(encoding="utf-8").splitlines()[0],
            '{"goal":"same","label":"first","step":2}',
        )

    def test_explicit_indices_and_reference_via_cli(self):
        rows = [{"position": index, "nested": {"value": index}} for index in range(7)]
        paths = []
        for index, name in enumerate(("zero.jsonl", "one.jsonl", "two.jsonl")):
            path = self.tmp_path / name
            _write_jsonl(path, rows[index::3])
            paths.append(path)
        reference = self.tmp_path / "reference.jsonl"
        output = self.tmp_path / "merged.jsonl"
        _write_jsonl(reference, rows)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "opportunity_forecasting.data.merge_labels",
                str(paths[2]),
                str(paths[0]),
                str(paths[1]),
                "--shard-indices",
                "2,0,1",
                "--num-shards",
                "3",
                "--reference",
                str(reference),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(_read_jsonl(output), rows)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "ok")
        self.assertEqual([shard["index"] for shard in report["shards"]], [0, 1, 2])

    def test_reference_mismatch_does_not_replace_output(self):
        shard = self.tmp_path / "sh0-of-1.jsonl"
        reference = self.tmp_path / "reference.jsonl"
        output = self.tmp_path / "merged.jsonl"
        _write_jsonl(shard, [{"value": 1}])
        _write_jsonl(reference, [{"value": 2}])
        output.write_text("existing\n", encoding="utf-8")

        with self.assertRaisesRegex(MergeError, "reference mismatch at row 0"):
            merge_data_shards([shard], output, reference_path=reference)

        self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")

    def test_malformed_shards_are_rejected(self):
        cases = [
            ('{"valid": true}\nnot-json\n', "invalid JSON"),
            ('{"valid": true}\n[]\n', "expected a JSON object"),
            ('{"valid": true}\n\n', "blank JSONL row"),
        ]
        for contents, message in cases:
            with self.subTest(contents=contents):
                shard = self.tmp_path / "sh0-of-1.jsonl"
                shard.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(MergeError, message):
                    merge_data_shards([shard], self.tmp_path / "merged.jsonl")

    def test_missing_and_duplicate_shard_indices_are_rejected(self):
        shard_zero = self.tmp_path / "part_sh0-of-3.jsonl"
        shard_two = self.tmp_path / "part_sh2-of-3.jsonl"
        duplicate_zero = self.tmp_path / "copy_sh0-of-3.jsonl"
        for path in (shard_zero, shard_two, duplicate_zero):
            _write_jsonl(path, [])

        with self.assertRaisesRegex(MergeError, r"missing shard indices: \[1\]"):
            merge_data_shards(
                [shard_zero, shard_two], self.tmp_path / "missing.jsonl"
            )

        with self.assertRaisesRegex(MergeError, "duplicate shard index 0"):
            merge_data_shards(
                [shard_zero, duplicate_zero, shard_two],
                self.tmp_path / "duplicate.jsonl",
            )
