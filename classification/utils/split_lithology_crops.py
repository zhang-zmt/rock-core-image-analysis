"""按岩性对单张裁片进行分层随机划分，并保存可追溯的训练清单。

默认训练、验证、测试比例为70%、20%、10%；少于5张的类别全部用于训练。
同一原图的不同裁片可以跨集合，程序记录这种情况，但不阻止划分。
原图只读；复制模式不缩放、不重编码、不移动图片，也不覆盖已有输出。
本文件只使用标准库，要求Python 3.9及以上版本。
"""

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_DIR / "training_metadata" / "source_manifest.jsonl"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def parse_args(argv=None):
    """保留参考脚本的输入、输出和模式参数，补充本工程所需的溯源参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=PROJECT_DIR / "data",
                        help="图片根目录，其直接子目录为岩性类别")
    parser.add_argument("--output_dir", type=Path,
                        default=PROJECT_DIR / "data_split_lithology",
                        help="尚不存在的输出目录")
    parser.add_argument("--source_manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="最终合并产生的逐图JSONL清单，用于核对身份和来源")
    parser.add_argument("--mode", choices=("copy", "manifest"), default="copy",
                        help="复制图片或只生成引用现有图片的清单")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="固定划分的随机种子")
    parser.add_argument("--dry-run", action="store_true",
                        help="完成身份校验并显示计划，不创建输出目录")
    return parser.parse_args(argv)


def fingerprint(path):
    """流式计算文件指纹，同时检查读取期间文件是否被其他程序改动。"""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"读取期间文件发生变化：{path}")
    return {"sha256": digest.hexdigest(), "size": after.st_size,
            "mtime_ns": after.st_mtime_ns}


def write_json(path, content):
    """统一使用UTF-8保存中文日志；仅用于本次新建目录中的文件。"""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_samples(input_dir, manifest_path):
    """用类别、文件名和内容指纹核对现有图片，不根据相似名称猜测来源。"""
    expected = {}
    case_ids = set()
    with manifest_path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            decision = record.get("final_decision")
            if decision == "排除":
                continue
            if decision != "保留":
                raise ValueError(f"清单第{line_number}行缺少有效的最终结论")
            for field in ("case_id", "category", "group_id", "crop_sha256",
                          "source_crop_relpath", "original_source_sha256"):
                if not isinstance(record.get(field), str) or not record[field]:
                    raise ValueError(f"清单第{line_number}行缺少字段：{field}")
            category = record.get("training_category", record["category"])
            if category != category.strip() or any(c in category for c in '<>:"/\\|?*\t\r\n'):
                raise ValueError(f"类别名称不能安全用于目录或清单：{category!r}")
            name = Path(record["source_crop_relpath"].replace("\\", "/")).name
            name = record.get("training_filename", name)
            key = (category, name)
            if key in expected or record["case_id"] in case_ids:
                raise ValueError(f"保留清单包含重复身份或同类别同名图片：{key}")
            expected[key] = record
            case_ids.add(record["case_id"])
    if not expected:
        raise ValueError("溯源清单没有确认保留的图片")

    samples = []
    matched = set()
    content_hashes = set()
    group_sources = {}
    # 先排序再打乱，避免磁盘遍历顺序影响固定种子的结果。
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if input_dir not in path.resolve().parents:
            raise ValueError(f"图片链接指向输入目录之外：{path}")
        relative_path = path.relative_to(input_dir)
        if len(relative_path.parts) < 2:
            raise ValueError(f"图片必须放在岩性类别目录内：{path}")
        if any(c in str(path) for c in "\t\r\n"):
            raise ValueError(f"路径含有清单无法表达的制表符或换行：{path}")
        key = (relative_path.parts[0], path.name)
        if key not in expected:
            raise ValueError(f"图片不在最终保留清单中，或类别/文件名已改变：{path}")
        if key in matched:
            raise ValueError(f"输入目录存在同类别同名的多份图片：{key}")
        record = expected[key]
        file_info = fingerprint(path)
        if file_info["sha256"] != record["crop_sha256"]:
            raise ValueError(f"图片内容与最终清单不一致：{path}")
        if file_info["sha256"] in content_hashes:
            raise ValueError(f"发现内容完全相同的重复裁片，需先核对，不重复划分：{path}")
        content_hashes.add(file_info["sha256"])
        matched.add(key)
        group_id = record["group_id"]
        original_sha = record["original_source_sha256"]
        if group_id in group_sources and group_sources[group_id] != original_sha:
            raise ValueError(f"同一原图组对应不同原图指纹：{group_id}")
        group_sources[group_id] = original_sha
        # 核对所有源文件，但用户明确排除的记录不进入训练清单。
        if record.get("exclude_from_training", False):
            continue
        sample = {
            "case_id": record["case_id"], "category": key[0],
            "original_category": record["category"],
            "source_path": str(path), "source_relpath": relative_path.as_posix(),
            "group_id": group_id, "original_source_sha256": original_sha,
            "crop_sha256": record["crop_sha256"], "source_fingerprint": file_info,
        }
        # 质量审核结论与岩性标签分开记录，避免把质量复核误写成岩性鉴定。
        for field in ("source_crop_relpath", "original_path_metadata_only",
                      "acquisition_group_id", "acquisition_identity_status",
                      "bbox_on_source", "size", "decision_source", "human_confirmed"):
            sample[field] = record.get(field)
        samples.append(sample)
        if len(samples) % 5000 == 0:
            print(f"已核对图片：{len(samples)}张", flush=True)
    missing = set(expected) - matched
    if missing:
        examples = sorted(missing)[:3]
        raise ValueError(f"输入目录比最终保留清单少{len(missing)}张，例如：{examples}")
    return samples


def split_samples(samples, ratios, seed):
    """每个类别独立打乱单张裁片，按比例分配整数张数，不按原图分组。"""
    by_category = {}
    for sample in samples:
        by_category.setdefault(sample["category"], []).append(sample)
    rows = []
    class_counts = {}
    for category in sorted(by_category):
        category_samples = sorted(by_category[category], key=lambda row: row["case_id"])
        # 每类使用独立随机序列，其他类别增删不会改变该类的划分。
        # 使用原类别名称保留改名前的随机序列含义。
        seed_category = category_samples[0].get("original_category", category)
        generator = random.Random(f"{seed}:{seed_category}")
        generator.shuffle(category_samples)
        count = len(category_samples)
        if count < 5:
            counts = [count, 0, 0]
        else:
            targets = []
            counts = []
            for ratio in ratios:
                targets.append(count * ratio)
                counts.append(math.floor(count * ratio))
            # 用最大小数余数分配剩余张数，三个集合之和始终等于原数量。
            while sum(counts) < count:
                index = max(range(3), key=lambda i: targets[i] - counts[i])
                counts[index] += 1
            # 图片数足够时保证每个集合至少一张；极端自定义比例也明确记账。
            for index in range(3):
                if counts[index] != 0:
                    continue
                donor = max(range(3), key=lambda i: counts[i])
                counts[donor] -= 1
                counts[index] = 1
        class_counts[category] = dict(zip(SPLITS, counts))
        start = 0
        for split, split_count in zip(SPLITS, counts):
            for sample in category_samples[start:start + split_count]:
                row = dict(sample)
                row["split"] = split
                rows.append(row)
            start += split_count
    rows.sort(key=lambda row: (row["split"], row["category"], row["case_id"]))
    return rows, class_counts


def audit_split(rows):
    """检查裁片不重复分配，并如实统计跨集合原图组；不把它当独立组测试。"""
    seen_cases = set()
    seen_hashes = set()
    group_splits = {}
    original_splits = {}
    split_counts = dict.fromkeys(SPLITS, 0)
    for row in rows:
        if row["case_id"] in seen_cases or row["crop_sha256"] in seen_hashes:
            raise ValueError("同一裁片身份或相同内容被重复分配")
        seen_cases.add(row["case_id"])
        seen_hashes.add(row["crop_sha256"])
        split_counts[row["split"]] += 1
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
        original_splits.setdefault(row["original_source_sha256"], set()).add(row["split"])
    overlapping_groups = {}
    for group_id, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            overlapping_groups[group_id] = sorted(splits)
    pair_counts = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        pair_counts[first + "_" + second] = 0
        for splits in group_splits.values():
            if first in splits and second in splits:
                pair_counts[first + "_" + second] += 1
    return {
        "split_unit": "cropped_image", "group_isolation_enforced": False,
        "split_counts": split_counts, "unique_images": len(seen_cases),
        "original_group_count": len(group_splits),
        "original_sha_count": len(original_splits),
        "overlapping_group_count": len(overlapping_groups),
        "overlapping_original_sha_count": sum(len(s) > 1 for s in original_splits.values()),
        "group_overlap_by_pair": pair_counts, "overlapping_groups": overlapping_groups,
        "interpretation": "裁片级分层随机划分；不能据此声称对未见原图或独立采集组的泛化性能。",
    }


def export_dataset(rows, output_dir, mode, summary):
    """只创建新结果目录；失败时保留失败记录，不把部分结果标为成功。"""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    summary_path = output_dir / "run_summary.json"
    summary["status"] = "writing"
    write_json(summary_path, summary)
    try:
        classes = sorted(summary["class_counts"])
        class_to_idx = {}
        for index, category in enumerate(classes):
            class_to_idx[category] = index
        write_json(output_dir / "class_to_idx.json", class_to_idx)
        write_json(output_dir / "source_overlap.json", summary.pop("overlap_detail"))
        shutil.copy2(Path(__file__), output_dir / Path(__file__).name)

        # 同时输出旧训练程序可读的两列清单，以及包含来源坐标的完整记录。
        with (output_dir / "samples.jsonl").open("x", encoding="utf-8", newline="\n") as detail:
            copied = 0
            for split in SPLITS:
                manifest_path = output_dir / (split + ".txt")
                with manifest_path.open("x", encoding="utf-8", newline="\n") as manifest:
                    for row in rows:
                        if row["split"] != split:
                            continue
                        source = Path(row["source_path"])
                        before = source.stat()
                        expected = row["source_fingerprint"]
                        if (before.st_size, before.st_mtime_ns) != (expected["size"], expected["mtime_ns"]):
                            raise ValueError(f"预检后源图片发生变化：{source}")
                        training_path = source
                        row["output_relpath"] = None
                        if mode == "copy":
                            relative_path = Path(split) / row["source_relpath"]
                            training_path = output_dir / relative_path
                            training_path.parent.mkdir(parents=True, exist_ok=True)
                            # 独占创建文件，杜绝覆盖；二进制复制保持像素和编码不变。
                            with source.open("rb") as reader, training_path.open("xb") as writer:
                                shutil.copyfileobj(reader, writer)
                            shutil.copystat(source, training_path)
                            if fingerprint(training_path)["sha256"] != row["crop_sha256"]:
                                raise ValueError(f"复制后内容校验失败：{training_path}")
                            row["output_relpath"] = relative_path.as_posix()
                            copied += 1
                            if copied % 5000 == 0:
                                print(f"已复制并校验：{copied}张", flush=True)
                        row["training_path"] = str(training_path)
                        row["class_index"] = class_to_idx[row["category"]]
                        manifest.write(str(training_path) + "\t" + row["category"] + "\n")
                        detail.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 再读源文件，确认本次操作前后其内容、大小和修改时间均未改变。
        print("正在核对全部源图片及划分依据是否保持不变。", flush=True)
        for row in rows:
            if fingerprint(Path(row["source_path"])) != row["source_fingerprint"]:
                raise ValueError(f"输出期间源图片发生变化：{row['source_path']}")
        if fingerprint(Path(summary["source_manifest"])) != summary["manifest_fingerprint"]:
            raise ValueError("输出期间最终处理清单发生变化")
        expected_paths = set()
        for row in rows:
            if row["output_relpath"] is not None:
                expected_paths.add(row["output_relpath"])
        actual_paths = set()
        for split in SPLITS:
            for path in (output_dir / split).rglob("*"):
                if path.is_file():
                    actual_paths.add(path.relative_to(output_dir).as_posix())
        if actual_paths != expected_paths:
            raise ValueError("复制结果存在缺失或额外文件")
        summary.update(status="completed", copied_images=copied,
                       source_integrity_unchanged=True, errors=[])
        write_json(summary_path, summary)
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="failed_incomplete", errors=[str(error)])
        write_json(summary_path, summary)
        raise


def main(argv=None):
    """按参数检查、身份核对、计划划分、输出核验的顺序执行。"""
    args = parse_args(argv)
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = args.source_manifest.resolve()
    try:
        if any(not math.isfinite(r) or r <= 0 for r in ratios):
            raise ValueError("三个比例必须都是大于0的有限数")
        if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9, rel_tol=0):
            raise ValueError("训练、验证、测试比例之和必须为1")
        if not input_dir.is_dir() or not manifest_path.is_file():
            raise ValueError("图片目录或最终处理清单不存在，请检查输入参数")
        if output_dir == input_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
            raise ValueError("输入和输出必须分离，不能互相包含")
        if output_dir.exists():
            raise ValueError(f"输出目录已存在，拒绝覆盖：{output_dir}")
        manifest_info = fingerprint(manifest_path)
        samples = read_samples(input_dir, manifest_path)
        rows, class_counts = split_samples(samples, ratios, args.seed)
        audit = audit_split(rows)
        if audit["unique_images"] != len(samples):
            raise ValueError("划分后图片数与输入图片数不一致")
        if fingerprint(manifest_path) != manifest_info:
            raise ValueError("预检期间最终处理清单发生变化")
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"预检失败：{error}", file=sys.stderr)
        return 2

    print("\n类别 | 训练 | 验证 | 测试")
    small_classes = []
    for category, counts in class_counts.items():
        print(f"{category} | {counts['train']} | {counts['val']} | {counts['test']}")
        if counts["val"] == 0 or counts["test"] == 0:
            small_classes.append(category)
    print(f"总计：{len(rows)}张，{len(class_counts)}类；{audit['split_counts']}")
    print(f"按单张裁片划分，跨集合原图组数：{audit['overlapping_group_count']}")
    if small_classes:
        print("以下类别不足5张，按文档规则全部用于训练：" + "、".join(small_classes))
    if args.dry_run:
        print("预检完成；未创建目录、清单或图片副本。")
        return 0

    summary = {
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "source_manifest": str(manifest_path), "manifest_fingerprint": manifest_info,
        "script_sha256": fingerprint(Path(__file__))["sha256"],
        "python_version": sys.version, "mode": args.mode, "seed": args.seed,
        "ratios": dict(zip(SPLITS, ratios)), "split_unit": "cropped_image",
        "group_isolation_enforced": False, "class_counts": class_counts,
        "split_counts": audit["split_counts"], "total_images": len(rows),
        "class_count": len(class_counts), "train_only_small_classes": small_classes,
        "overlap_detail": audit, "overlapping_group_count": audit["overlapping_group_count"],
        "interpretation": audit["interpretation"], "images_decoded": False,
        "integrity_scope": "全量文件内容SHA256、大小、修改时间；不逐张解码或复核岩性标签",
        "rounding_rule": "各类按最大余数法取整；至少5张时各集合至少1张；不足5张全部训练",
    }
    try:
        export_dataset(rows, output_dir, args.mode, summary)
    except (Exception, KeyboardInterrupt) as error:
        print(f"输出未完成：{error}。请查看新输出目录中的运行摘要。", file=sys.stderr)
        return 1
    print(f"完成：{output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
