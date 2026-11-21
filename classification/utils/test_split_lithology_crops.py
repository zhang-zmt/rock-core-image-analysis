"""验证划分数量、可复现性、来源记录和文件保护；不运行模型训练。"""

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("split_lithology_crops.py")
SPEC = importlib.util.spec_from_file_location("crop_split", SCRIPT_PATH)
splitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splitter)


class SplitTests(unittest.TestCase):
    """使用临时目录构造输入，测试结束后只清理这些临时测试文件。"""

    def make_fixture(self, root, counts):
        """构造不同字节内容的测试文件；本脚本校验文件指纹，不解码图像。"""
        input_dir = root / "data"
        manifest_path = root / "source.jsonl"
        records = []
        number = 0
        for category, count in counts.items():
            (input_dir / category).mkdir(parents=True)
            for index in range(count):
                number += 1
                path = input_dir / category / f"image_{index}.png"
                data = f"测试字节-{category}-{index}".encode("utf-8")
                path.write_bytes(data)
                records.append({
                    "case_id": f"FULL-{number:06d}", "category": category,
                    "final_decision": "保留", "group_id": category + "-原图一",
                    "original_source_sha256": hashlib.sha256(category.encode()).hexdigest(),
                    "crop_sha256": hashlib.sha256(data).hexdigest(),
                    "source_crop_relpath": category + "/" + path.name,
                    "bbox_on_source": [index, 0, index + 250, 350],
                    "decision_source": "人工确认", "human_confirmed": True,
                })
        with manifest_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return input_dir, manifest_path

    def run_cli(self, input_dir, manifest_path, output_dir, *extra):
        """收集控制台信息，使测试结果便于阅读。"""
        arguments = ["--input_dir", str(input_dir), "--source_manifest", str(manifest_path),
                     "--output_dir", str(output_dir)] + list(extra)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return splitter.main(arguments)

    def test_stratification_and_group_overlap(self):
        """同一原图的不同裁片允许跨集合，但必须被明确记录。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, manifest = self.make_fixture(root, {"煤": 10, "灰岩": 10})
            samples = splitter.read_samples(inputs, manifest)
            rows, counts = splitter.split_samples(samples, (0.7, 0.2, 0.1), 42)
            self.assertEqual(counts["煤"], {"train": 7, "val": 2, "test": 1})
            self.assertEqual(counts["灰岩"], counts["煤"])
            audit = splitter.audit_split(rows)
            self.assertEqual(audit["overlapping_group_count"], 2)
            self.assertEqual(audit["unique_images"], 20)

    def test_seed_is_independent_of_file_order(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, manifest = self.make_fixture(Path(directory), {"煤": 50})
            samples = splitter.read_samples(inputs, manifest)
            first, _ = splitter.split_samples(samples, (0.7, 0.2, 0.1), 42)
            second, _ = splitter.split_samples(list(reversed(samples)), (0.7, 0.2, 0.1), 42)
            third, _ = splitter.split_samples(samples, (0.7, 0.2, 0.1), 43)
            self.assertEqual(first, second)
            self.assertNotEqual(first, third)

    def test_rounding_and_small_class_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs, manifest = self.make_fixture(Path(directory), {"少量": 4, "边界": 5, "普通": 17})
            samples = splitter.read_samples(inputs, manifest)
            _, counts = splitter.split_samples(samples, (0.7, 0.2, 0.1), 42)
            self.assertEqual(counts["少量"], {"train": 4, "val": 0, "test": 0})
            self.assertEqual(counts["边界"], {"train": 3, "val": 1, "test": 1})
            self.assertEqual(sum(counts["普通"].values()), 17)

    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, manifest = self.make_fixture(root, {"煤": 10})
            output = root / "new_parent" / "output"
            self.assertEqual(self.run_cli(inputs, manifest, output, "--dry-run"), 0)
            self.assertFalse(output.parent.exists())

    def test_copy_and_manifest_are_compatible_and_traceable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, manifest = self.make_fixture(root, {"煤": 10, "灰岩": 10})
            before = {}
            for path in inputs.rglob("*.png"):
                before[str(path)] = splitter.fingerprint(path)
            for mode in ("copy", "manifest"):
                output = root / mode
                self.assertEqual(self.run_cli(inputs, manifest, output, "--mode", mode), 0)
                summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], "completed")
                self.assertTrue(summary["source_integrity_unchanged"])
                records = []
                for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines():
                    records.append(json.loads(line))
                self.assertEqual(len(records), 20)
                for record in records:
                    target = Path(record["training_path"])
                    self.assertEqual(splitter.fingerprint(target)["sha256"], record["crop_sha256"])
                    self.assertIn("bbox_on_source", record)
                # 按原训练程序的读取方式检查两列格式和类别编号。
                labels = set()
                for line in (output / "train.txt").read_text(encoding="utf-8").splitlines():
                    path_text, category = line.strip().split("\t")
                    self.assertTrue(Path(path_text).is_file())
                    labels.add(category)
                mapping = json.loads((output / "class_to_idx.json").read_text(encoding="utf-8"))
                self.assertEqual(list(mapping), sorted(labels))
                if mode == "manifest":
                    self.assertFalse((output / "train").exists())
            for path, file_info in before.items():
                self.assertEqual(splitter.fingerprint(Path(path)), file_info)

    def test_existing_output_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, manifest = self.make_fixture(root, {"煤": 10})
            output = root / "existing"
            output.mkdir()
            marker = output / "保留.txt"
            marker.write_text("不能覆盖", encoding="utf-8")
            self.assertEqual(self.run_cli(inputs, manifest, output), 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "不能覆盖")

    def test_nested_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, manifest = self.make_fixture(root, {"煤": 10})
            self.assertEqual(self.run_cli(inputs, manifest, inputs / "output"), 2)
            self.assertFalse((inputs / "output").exists())

    def test_missing_changed_and_unknown_files_are_rejected(self):
        for fault in ("missing", "changed", "unknown"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, manifest = self.make_fixture(root, {"煤": 10})
                first = inputs / "煤" / "image_0.png"
                if fault == "missing":
                    first.unlink()
                elif fault == "changed":
                    first.write_bytes(b"changed")
                else:
                    (inputs / "煤" / "unknown.png").write_bytes(b"unknown")
                self.assertEqual(self.run_cli(inputs, manifest, root / "output"), 2)
                self.assertFalse((root / "output").exists())

    def test_invalid_ratios_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, manifest = self.make_fixture(root, {"煤": 10})
            for value in ("nan", "-0.1", "0.8"):
                self.assertEqual(self.run_cli(inputs, manifest, root / "output", "--train_ratio", value), 2)

    def test_ascii_alias_and_manual_exclusion(self):
        """英文目录保留原始类别身份；主动排除的记录不会重新进入训练。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, manifest = self.make_fixture(root, {"煤": 10})
            original = splitter.read_samples(inputs, manifest)
            (inputs / "煤").rename(inputs / "class_00")
            records = []
            for line in manifest.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                record["training_category"] = "class_00"
                record["training_filename"] = Path(record["source_crop_relpath"]).name
                record["exclude_from_training"] = record["case_id"] == "FULL-000001"
                records.append(record)
            with manifest.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            renamed = splitter.read_samples(inputs, manifest)
            self.assertEqual(len(renamed), 9)
            self.assertEqual({row["category"] for row in renamed}, {"class_00"})
            self.assertEqual({row["original_category"] for row in renamed}, {"煤"})
            original = [row for row in original if row["case_id"] != "FULL-000001"]
            old_split, _ = splitter.split_samples(original, (0.7, 0.2, 0.1), 42)
            new_split, _ = splitter.split_samples(renamed, (0.7, 0.2, 0.1), 42)
            self.assertEqual({r["case_id"]: r["split"] for r in old_split},
                             {r["case_id"]: r["split"] for r in new_split})
            self.assertEqual(self.run_cli(inputs, manifest, root / "output", "--mode", "manifest"), 0)

    def test_duplicate_manifest_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, manifest = self.make_fixture(root, {"煤": 10})
            first = manifest.read_text(encoding="utf-8").splitlines()[0]
            with manifest.open("a", encoding="utf-8") as stream:
                stream.write(first + "\n")
            self.assertEqual(self.run_cli(inputs, manifest, root / "output"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
