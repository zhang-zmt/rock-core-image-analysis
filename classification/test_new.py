"""使用现有21类岩性权重进行推理，不训练、不更新权重。

【改动说明】原程序的三模型级联已替换为单个ResNet101。
默认读取同目录的test文件夹；250×350图片按裁片处理，其余按原图分块处理。
原图坐标沿用既有裁切流程，质量结论按内容指纹匹配，不通过文件名猜测。
"""

import argparse
import hashlib
import io
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_args():
    """【修改1】图片、单份权重和类别映射分开配置，取消旧的三权重参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img", type=Path, default=ROOT / "test", help="单张图片或图片目录，支持递归")
    parser.add_argument("--weights", type=Path, default=ROOT / "best_ore_sub.pth", help="现有21类权重")
    parser.add_argument("--class_map", type=Path, default=ROOT / "training_metadata/class_labels.json", help="训练时的类别顺序")
    parser.add_argument("--quality_manifest", type=Path, default=ROOT / "training_metadata/inference_quality.jsonl", help="冻结的质量审核清单")
    parser.add_argument("--split_manifest", type=Path, default=ROOT / "data_split/samples.jsonl", help="仅用于标记训练来源重合，不参与预测")
    parser.add_argument("--sample_manifest", type=Path, help="可选的抽样溯源清单，默认自动查找")
    parser.add_argument("--output", type=Path, help="必须不存在的新结果目录，默认使用时间命名")
    parser.add_argument("--arch", choices=["resnet101"], default="resnet101", help="当前权重已核实为ResNet101")
    parser.add_argument("--mode", choices=["auto", "patch", "tiled"], default="auto", help="自动判断、单裁片、原图分块")
    parser.add_argument("--quality", choices=["reviewed", "all"], default="reviewed", help="默认只汇总已有保留结论；all仅用于未筛选诊断")
    parser.add_argument("--device", default="cuda", help="cuda或cpu；无CUDA时回退CPU")
    parser.add_argument("--batch", type=int, default=16, help="一次送入模型的裁片数量")
    parser.add_argument("--limit", type=int, default=0, help="只处理排序前N张，0表示全部")
    return parser.parse_args()


def fingerprint(path):
    """记录内容、大小和修改时间，检测读取期间变化并用于前后完整性校验。"""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"文件在读取时变化：{path}")
    return {"sha256": digest.hexdigest(), "size": after.st_size, "mtime_ns": after.st_mtime_ns}


def write_json(path, value):
    """统一保存中文和严格JSON，避免写出不可解析的无穷值。"""
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_classes(path):
    """【修改2】沿用训练编号，不重新按中文名称排序。"""
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    rows.sort(key=lambda row: row["class_index"])
    indices = []
    names = set()
    originals = set()
    for row in rows:
        indices.append(row["class_index"])
        names.add(row["class_name"])
        originals.add(row["original_name"])
    if indices != list(range(21)) or len(names) != 21 or len(originals) != 21:
        raise ValueError("类别映射必须包含互不重复的21类，编号连续为0—20")
    return rows


def load_quality(path):
    """【修改3】只读取已有审核结论，不在推理时重新猜测黑底或模糊阈值。"""
    by_box = {}
    by_crop = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            box = row["bbox_on_source"]
            if len(box) != 4 or any(type(value) is not int for value in box):
                raise ValueError(f"质量清单第{line_number}行坐标不是四个整数")
            if box[0] < 0 or box[1] < 0 or box[2] - box[0] != 250 or box[3] - box[1] != 350:
                raise ValueError(f"质量清单第{line_number}行不是有效的250×350裁框")
            if row["final_decision"] not in ("保留", "排除"):
                raise ValueError(f"质量清单第{line_number}行没有明确结论")
            key = (row["original_source_sha256"], tuple(box))
            if key in by_box:
                raise ValueError(f"质量清单重复记录同一原图坐标：{key}")
            by_box[key] = row
            by_crop.setdefault(row["crop_sha256"], []).append(row)
    return by_box, by_crop


def plan_patches(width, height):
    """【修改4】复现500×700二维全覆盖、右下补窗及四象限，不拉伸原图。

    原图尺寸可以是奇数；固定窗口始终为偶数，因此四象限仍为250×350。
    坐标是方向校正后的原图坐标，右边界和下边界不包含在裁框内。
    """
    if width < 500 or height < 700:
        raise ValueError("原图小于500×700；不自动放大或补黑，请检查输入模式")
    xs = list(range(0, width - 500 + 1, 500))
    ys = list(range(0, height - 700 + 1, 700))
    regular_columns = len(xs)
    regular_rows = len(ys)
    if xs[-1] != width - 500:
        xs.append(width - 500)
    if ys[-1] != height - 700:
        ys.append(height - 700)
    quadrants = [("top_left", 0, 0), ("top_right", 250, 0),
                 ("bot_left", 0, 350), ("bot_right", 250, 350)]
    patches = []
    for row, y in enumerate(ys):
        for column, x in enumerate(xs):
            overlap_x = 0
            overlap_y = 0
            if column:
                overlap_x = max(0, xs[column - 1] + 500 - x)
            if row:
                overlap_y = max(0, ys[row - 1] + 700 - y)
            for quadrant, dx, dy in quadrants:
                patches.append({"grid_row": row, "grid_column": column, "quadrant": quadrant,
                                "bbox_on_source": [x + dx, y + dy, x + dx + 250, y + dy + 350],
                                "parent_bbox_on_source": [x, y, x + 500, y + 700],
                                "tail_right": column >= regular_columns, "tail_bottom": row >= regular_rows,
                                "parent_overlap_x_px": overlap_x, "parent_overlap_y_px": overlap_y})
    return patches


def quality_decision(record, policy):
    """未匹配不是合格；诊断模式可计算全部区域，但必须保留原审核状态。"""
    state = "unknown"
    if record is not None:
        state = "keep"
        if record["final_decision"] == "排除":
            state = "exclude"
        if record.get("exclude_from_training", False):
            state = "manual_dataset_exclusion"
    if policy == "all":
        return state, True
    return state, state == "keep"


def load_net(weights, device):
    """【修改5】只构造一个21类模型；严格加载，不下载预训练参数、不更新权重。"""
    model = models.resnet101(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 21)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def predict_patches(image, patches, model, device, batch_size, classes):
    """【修改6】逐裁片计算21类分数；文件夹名称和参考标签不传入模型。"""
    # 255来自当前训练脚本的round(224×1.14)，不沿用旧脚本的256。
    transform = transforms.Compose([
        transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    usable = []
    for patch in patches:
        if patch["used_for_inference"]:
            usable.append(patch)
    probability_sum = torch.zeros(21, dtype=torch.float64)
    votes = Counter()
    forward_seconds = 0.0
    with torch.inference_mode():
        for start in range(0, len(usable), batch_size):
            batch_rows = usable[start:start + batch_size]
            tensors = []
            for patch in batch_rows:
                # 裁片仅在内存中生成，避免额外写出数千张重复图片。
                crop = image.crop(patch["bbox_on_input"])
                tensors.append(transform(crop))
                crop.close()
            batch = torch.stack(tensors).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            logits = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - tick
            if not torch.isfinite(logits).all():
                raise ValueError("模型输出包含非有限数值")
            probabilities = logits.softmax(dim=1).cpu()
            for patch, probability in zip(batch_rows, probabilities):
                scores, indices = probability.topk(3)
                top3 = []
                for score, index in zip(scores.tolist(), indices.tolist()):
                    top3.append({"class_index": index, "class_name": classes[index]["class_name"],
                                 "original_name": classes[index]["original_name"], "score": score})
                patch["top3"] = top3
                patch["predicted_index"] = top3[0]["class_index"]
                probability_sum += probability.to(torch.float64)
                votes[patch["predicted_index"]] += 1
    if not usable:
        return None, votes, forward_seconds
    # 平均分数是汇总规则，并不是对“地质结论正确概率”的校准。
    return probability_sum / len(usable), votes, forward_seconds


def infer_image(path, model, device, args, classes, by_box, by_crop, train_groups, train_crops):
    """【修改7】每张原图单独汇总；同一目录内不同原图之间不混合投票。"""
    started = time.perf_counter()
    before = fingerprint(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != before["sha256"]:
        raise ValueError(f"读取图片期间内容变化：{path}")
    with Image.open(io.BytesIO(data)) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("多帧图片不支持，不能静默只取第一帧")
        orientation = int(opened.getexif().get(274, 1))
        raw_size = list(opened.size)
        with ImageOps.exif_transpose(opened) as oriented:
            image = oriented.convert("RGB")
    try:
        mode = args.mode
        if mode == "auto":
            mode = "tiled"
            if image.size == (250, 350):
                mode = "patch"
        if mode == "patch" and image.size != (250, 350):
            raise ValueError("patch模式要求250×350，不会把大原图误作训练裁片")
        groups = set()
        patches = []
        if mode == "tiled":
            groups.add(before["sha256"])
            patches = plan_patches(*image.size)
            for patch in patches:
                patch["bbox_on_input"] = patch["bbox_on_source"]
                record = by_box.get((before["sha256"], tuple(patch["bbox_on_source"])))
                patch["quality_record"] = record
        else:
            matches = by_crop.get(before["sha256"], [])
            record = None
            # 同内容有多个溯源记录时不任取一个，避免把来源与质量结论配错。
            if len(matches) == 1:
                record = matches[0]
                groups.add(record["original_source_sha256"])
            patch = {"bbox_on_input": [0, 0, 250, 350], "bbox_on_source": None,
                     "quality_record": record, "quality_match_candidates": len(matches)}
            if record is not None:
                patch["bbox_on_source"] = record["bbox_on_source"]
            patches.append(patch)
        quality_counts = Counter()
        for number, patch in enumerate(patches, 1):
            record = patch.pop("quality_record")
            state, usable = quality_decision(record, args.quality)
            patch.update(patch_number=number, quality_state=state, used_for_inference=usable,
                         case_id=None, quality_decision_source=None, quality_human_confirmed=None)
            if record is not None:
                patch.update(case_id=record["case_id"], crop_sha256=record["crop_sha256"],
                             quality_decision_source=record.get("decision_source"),
                             quality_human_confirmed=record.get("human_confirmed"),
                             also_in_train_crop=record["crop_sha256"] in train_crops)
            quality_counts[state] += 1
        probability, votes, forward_seconds = predict_patches(image, patches, model, device, args.batch, classes)
        result = {"input_path": str(path), "source_fingerprint": before, "mode": mode,
                  "raw_size": raw_size, "oriented_size": list(image.size), "exif_orientation": orientation,
                  "group_ids": sorted(groups), "also_in_train_source": bool(groups & train_groups),
                  "also_in_train_crop": before["sha256"] in train_crops,
                  "total_patch_count": len(patches), "used_patch_count": sum(votes.values()),
                  "quality_counts": dict(quality_counts), "quality_policy": args.quality,
                  "status": "needs_review", "predicted_index": None, "top3": [],
                  "aggregation": "mean_probability_over_eligible_patches_equal_weight_including_tail_overlap",
                  "forward_seconds": forward_seconds}
        if probability is not None:
            scores, indices = probability.topk(3)
            top3 = []
            for score, index in zip(scores.tolist(), indices.tolist()):
                top3.append({"class_index": index, "class_name": classes[index]["class_name"],
                             "original_name": classes[index]["original_name"], "score": score})
            result.update(status="predicted", predicted_index=top3[0]["class_index"], top3=top3,
                          probabilities=probability.tolist(), patch_top1_votes=dict(votes),
                          agreement_with_aggregate=votes[top3[0]["class_index"]] / sum(votes.values()))
            if args.quality == "all":
                result["status"] = "diagnostic_unfiltered"
        result["elapsed_seconds"] = time.perf_counter() - started
        result["source_unchanged"] = before == fingerprint(path)
        if not result["source_unchanged"]:
            raise ValueError(f"推理前后图片发生变化：{path}")
        return result, patches
    finally:
        image.close()


def summarize(rows, classes):
    """报告真实分母，未得到预测的样本计入覆盖率，不悄悄从总数中消失。"""
    confusion = [[0] * len(classes) for _ in classes]
    labelled = 0
    covered = 0
    correct = 0
    errors = 0
    for row in rows:
        if row.get("status") == "error":
            errors += 1
        truth = row.get("reference_index")
        prediction = row.get("predicted_index")
        if truth is None:
            continue
        labelled += 1
        if prediction is None:
            continue
        covered += 1
        confusion[truth][prediction] += 1
        if prediction == truth:
            correct += 1
    per_class = []
    for index, label in enumerate(classes):
        support = sum(confusion[index])
        predicted_count = sum(line[index] for line in confusion)
        tp = confusion[index][index]
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        if predicted_count:
            precision = tp / predicted_count
        if support:
            recall = tp / support
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        all_support = sum(row.get("reference_index") == index for row in rows)
        per_class.append(dict(label, labelled_support=all_support, predicted_support=support,
                              precision=precision, recall=recall, f1=f1))
    result = {"images": len(rows), "labelled_images": labelled, "predicted_labelled_images": covered,
              "correct": correct, "processing_errors": errors, "confusion_matrix": confusion,
              "per_class": per_class, "accuracy_on_predicted": None, "correct_over_all_labelled": None,
              "coverage_of_labelled": None, "macro_f1_over_supported_classes": None}
    if labelled:
        result.update(correct_over_all_labelled=correct / labelled, coverage_of_labelled=covered / labelled)
    if covered:
        result["accuracy_on_predicted"] = correct / covered
        supported = [row["f1"] for row in per_class if row["predicted_support"]]
        result["macro_f1_over_supported_classes"] = sum(supported) / len(supported)
    return result


def main():
    args = get_args()
    # 【修改8】相对默认路径均基于脚本所在目录，不依赖IDE当前工作目录。
    if args.batch < 1 or args.limit < 0:
        raise ValueError("batch必须大于0，limit不能小于0")
    args.img = args.img.resolve()
    if not args.img.exists():
        raise FileNotFoundError(f"输入不存在：{args.img}")
    files = []
    if args.img.is_file():
        if args.img.suffix.lower() not in EXTENSIONS:
            raise ValueError("输入文件不是支持的图片格式")
        files = [args.img]
    else:
        for path in sorted(args.img.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                files.append(path)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise ValueError("没有找到图片，请先准备test目录")
    if args.output is None:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        args.output = ROOT / ("inference_results_" + stamp)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有结果目录：{output}")
    if args.img.is_dir() and output.is_relative_to(args.img):
        raise ValueError("结果目录不能放进输入图片目录，避免下次递归重复读取")
    if args.sample_manifest is None:
        start = args.img
        if start.is_file():
            start = start.parent
        for parent in [start] + list(start.parents):
            candidate = parent / "sampling_manifest.jsonl"
            if candidate.is_file():
                args.sample_manifest = candidate
                break
    protected_paths = [args.weights, args.class_map, args.quality_manifest, args.split_manifest, Path(__file__)]
    if args.sample_manifest is not None:
        protected_paths.append(args.sample_manifest)
    snapshots = {}
    for path in protected_paths:
        snapshots[str(path.resolve())] = fingerprint(path)
    classes = load_classes(args.class_map)
    by_box, by_crop = load_quality(args.quality_manifest)
    train_groups = set()
    train_crops = set()
    with args.split_manifest.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] == "train":
                train_groups.add(row["original_source_sha256"])
                train_crops.add(row["crop_sha256"])
    samples = {}
    if args.sample_manifest is not None:
        with args.sample_manifest.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if not line.strip():
                    continue
                sample = json.loads(line)
                key = sample["copy_relpath"].replace("\\", "/")
                if key in samples:
                    raise ValueError(f"抽样清单含重复路径：{key}")
                if type(sample["class_index"]) is not int or not 0 <= sample["class_index"] < 21:
                    raise ValueError(f"抽样清单类别编号无效：{key}")
                samples[key] = sample
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，已回退CPU。", flush=True)
        device = torch.device("cpu")
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = load_net(args.weights, device)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "class_labels.json", classes)
    summary = {"status": "running", "args": {}, "snapshots_before": snapshots,
               "model": "resnet101", "device": str(device), "torch": torch.__version__,
               "preprocessing": "RGB; Resize短边255; CenterCrop224; ImageNet归一化",
               "tta": False, "amp": False, "weight_updates": False,
               "limitations": ["参考岩性来自目录或抽样清单，未重新地质鉴定",
                               "同源训练图片只能用于内部复查，不能冒充独立验收",
                               "补尾重叠裁片等权汇总，面积重叠会影响权重；不视为独立观测",
                               "历史质量结论包括人工确认与授权采纳的机器建议，分别记录",
                               "低分不能证明未知岩性或矿区外来源；本程序不作矿区归属判断",
                               "耗时含读图校验、分块、预处理和推理；不含模型加载，无预热，另记前向耗时"]}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value.resolve())
        summary["args"][key] = value
    if device.type == "cuda":
        summary["device_name"] = torch.cuda.get_device_name(device)
    write_json(output / "run_summary.json", summary)
    rows = []
    alias = {}
    for label in classes:
        alias[label["class_name"]] = label["class_index"]
        alias[label["original_name"]] = label["class_index"]
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions, (output / "patch_predictions.jsonl").open("w", encoding="utf-8") as patch_file:
        for number, path in enumerate(files, 1):
            row = {"input_path": str(path), "reference_index": None}
            sample = None
            reference = None
            for parent in path.parents:
                name = re.sub(r"[（(].*$", "", parent.name).strip()
                if name in alias:
                    reference = alias[name]
                    break
            try:
                if args.sample_manifest is not None:
                    sample_root = args.sample_manifest.resolve().parent
                    if path.is_relative_to(sample_root):
                        sample = samples.get(path.relative_to(sample_root).as_posix())
                    if sample is None:
                        raise ValueError("图片不在提供的抽样清单中，不能静默丢失溯源")
                    reference = sample["class_index"]
                row, patches = infer_image(path, model, device, args, classes, by_box, by_crop, train_groups, train_crops)
                if sample is not None:
                    if sample["sha256"] != row["source_fingerprint"]["sha256"]:
                        raise ValueError("图片SHA与抽样清单不一致")
                    reference = sample["class_index"]
                    row.update(sample_id=sample["sample_id"], original_path=sample["original_path"])
                # 参考标签不传给推理函数，只用于完成后的计分和错误记录。
                row["reference_index"] = reference
                row["reference_label_source"] = "directory_or_sampling_manifest_not_geologically_reverified"
                row["correct"] = None
                if reference is not None:
                    row["reference_name"] = classes[reference]["original_name"]
                    if row["predicted_index"] is not None:
                        row["correct"] = row["predicted_index"] == reference
                for patch in patches:
                    patch["input_path"] = str(path)
                    patch["input_sha256"] = row["source_fingerprint"]["sha256"]
                    patch_file.write(json.dumps(patch, ensure_ascii=False, allow_nan=False) + "\n")
                text = "无可用审核裁片，待核查"
                if row["top3"]:
                    text = row["top3"][0]["original_name"]
                print(f"[{number}/{len(files)}] {path.name} → {text}；使用{row['used_patch_count']}/{row['total_patch_count']}片", flush=True)
            except Exception as exc:
                # 单图失败保留错误记录和非零退出码，其他图片仍可继续检查。
                if sample is not None:
                    reference = sample["class_index"]
                row = {"input_path": str(path), "status": "error", "error": str(exc),
                       "reference_index": reference, "predicted_index": None}
                print(f"[{number}/{len(files)}] 失败：{path.name}：{exc}", flush=True)
            rows.append(row)
            predictions.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            predictions.flush()
    summary["metrics"] = summarize(rows, classes)
    summary["train_source_overlap_images"] = sum(bool(row.get("also_in_train_source")) for row in rows)
    summary["unique_input_sha256_count"] = len({row["source_fingerprint"]["sha256"] for row in rows if "source_fingerprint" in row})
    summary["independent_acquisition_groups_confirmed"] = None
    summary["source_integrity"] = []
    for row in rows:
        if "source_fingerprint" in row:
            summary["source_integrity"].append({"path": row["input_path"], "unchanged": row["source_unchanged"]})
    summary["protected_files_unchanged"] = {}
    for path, before in snapshots.items():
        summary["protected_files_unchanged"][path] = before == fingerprint(Path(path))
    failed = summary["metrics"]["processing_errors"] > 0
    if not all(summary["protected_files_unchanged"].values()):
        failed = True
    summary["status"] = "completed"
    if failed:
        summary["status"] = "completed_with_errors"
    write_json(output / "run_summary.json", summary)
    # 【修改9】同时提供可直接阅读的文本，JSON保留完整字段供后续程序复核。
    metrics = summary["metrics"]
    report = ["岩性单图与原图分块推理结果", "",
              f"图片数：{len(rows)}；处理失败：{metrics['processing_errors']}。",
              f"有参考标签：{metrics['labelled_images']}；其中得到预测：{metrics['predicted_labelled_images']}。",
              f"与训练来源重合的图片：{summary['train_source_overlap_images']}。",
              "这些参考标签来自文件目录或抽样记录，未重新进行地质鉴定。"]
    if metrics["accuracy_on_predicted"] is not None:
        report.append(f"有预测且有参考标签的准确率：{metrics['accuracy_on_predicted']:.2%}。")
        report.append(f"有参考标签图片的预测覆盖率：{metrics['coverage_of_labelled']:.2%}。")
        report.append(f"正确数除以全部有参考标签图片：{metrics['correct_over_all_labelled']:.2%}。")
    report.extend(["", "方法与限制："] + summary["limitations"] + ["", "逐图结果："])
    for number, row in enumerate(rows, 1):
        report.append(f"\n{number}. {row['input_path']}")
        if row["status"] == "error":
            report.append("处理失败：" + row["error"])
            continue
        report.append("参考岩性：" + row.get("reference_name", "未知"))
        report.append(f"使用裁片：{row['used_patch_count']}/{row['total_patch_count']}；状态：{row['status']}")
        for position, prediction in enumerate(row["top3"], 1):
            report.append(f"Top-{position}：{prediction['original_name']}；分数：{prediction['score']:.6f}")
    (output / "inference_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"结果保存到：{output}", flush=True)
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"参数、路径或预检失败：{error}", file=sys.stderr)
        sys.exit(2)
